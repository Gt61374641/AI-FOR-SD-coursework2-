"""
Training script for 4-band Amazon deforestation detection dataset.

This script handles multi-spectral satellite imagery (R, G, B, NIR bands)
for binary segmentation of deforested areas.
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from train_pytorch import (
    UNet, AttentionUNet, ResNet50SegNet, FCN32VGG16, ResUNet,
    ConvBlock, AttentionBlock, CombinedLoss, compute_metrics, set_seed
)


class Amazon4BandDataset(Dataset):
    """
    Dataset for 4-band GeoTIFF satellite imagery.
    Supports loading multi-spectral data with rasterio or PIL fallback.
    """
    def __init__(self, image_paths, mask_paths, img_size=512, in_channels=4):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.in_channels = in_channels
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Try loading with rasterio (preferred for GeoTIFF)
        try:
            import rasterio
            with rasterio.open(img_path) as src:
                image = src.read()  # Shape: (C, H, W)
                # Adjust number of channels
                if image.shape[0] > self.in_channels:
                    image = image[:self.in_channels]
                elif image.shape[0] < self.in_channels:
                    pad = np.zeros((self.in_channels - image.shape[0], *image.shape[1:]))
                    image = np.concatenate([image, pad], axis=0)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Single band mask
        except ImportError:
            # Fallback to PIL
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            # Handle channel dimension
            if image.ndim == 2:
                image = np.stack([image] * self.in_channels, axis=0)
            elif image.ndim == 3:
                if image.shape[-1] <= 8:  # HWC format
                    image = np.transpose(image, (2, 0, 1))
                if image.shape[0] > self.in_channels:
                    image = image[:self.in_channels]
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        
        # Min-max normalization
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 0:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        
        # Resize if needed
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            resized = []
            for c in range(image.shape[0]):
                ch = (image[c] * 255).astype(np.uint8)
                ch = np.array(Image.fromarray(ch).resize((self.img_size, self.img_size), Image.BILINEAR)) / 255.0
                resized.append(ch)
            image = np.stack(resized, axis=0)
        
        if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (self.img_size, self.img_size), Image.NEAREST))
        
        # Binarize mask (original labels: 1=forest, 2=deforestation -> 0/1)
        unique_vals = np.unique(mask)
        if len(unique_vals) <= 3 and mask.max() <= 2 and mask.max() > 1:
            mask = np.clip(mask - 1, 0, 1)
        elif mask.max() > 1:
            mask = (mask > 0).astype(np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        
        return image, mask


def get_amazon_4band_loaders(data_dir="./AMAZON", batch_size=8, val_split=0.2):
    """Create data loaders for 4-band Amazon dataset."""
    train_img_dir = os.path.join(data_dir, "Training", "image")
    train_mask_dir = os.path.join(data_dir, "Training", "label")
    
    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.tif")))
    print(f"Found {len(train_images)} training images")
    
    # Match images with masks
    pairs = []
    for img_path in train_images:
        mask_path = os.path.join(train_mask_dir, os.path.basename(img_path))
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    
    # Limit to 250 as per original paper
    if len(pairs) > 250:
        pairs = pairs[:250]
    
    # Split into train/val
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_split, random_state=42)
    print(f"Training: {len(train_pairs)} | Validation: {len(val_pairs)}")
    
    train_dataset = Amazon4BandDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs])
    val_dataset = Amazon4BandDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def get_model_4band(model_name, in_channels=4):
    """Factory function for models with 4-channel input."""
    models = {
        'unet': UNet(in_ch=in_channels, out_ch=1, base=32),
        'attention_unet': AttentionUNet(in_ch=in_channels, out_ch=1, base=16),
        'resnet50_segnet': ResNet50SegNet(in_ch=in_channels, out_ch=1),
        'fcn32_vgg16': FCN32VGG16(in_ch=in_channels, out_ch=1),
        'resunet': ResUNet(in_ch=in_channels, out_ch=1),
    }
    return models[model_name]


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            metrics = compute_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        pbar.set_postfix({'loss': loss.item(), 'f1': metrics['f1_score']})
    
    return total_loss / len(loader.dataset), {k: np.mean(v) for k, v in all_metrics.items()}


def validate(model, loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            metrics = compute_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    return total_loss / len(loader.dataset), {k: np.mean(v) for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Train on 4-band Amazon dataset')
    parser.add_argument('--model', type=str, default='attention_unet', 
                        choices=['unet', 'attention_unet', 'resnet50_segnet', 'fcn32_vgg16', 'resunet'])
    parser.add_argument('--data_dir', type=str, default='./AMAZON')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() 
                          else args.device if args.device != 'auto' else 'cpu')
    
    print(f"\n4-Band Amazon Dataset | Model: {args.model} | Device: {device}\n")
    
    # Load data and create model
    train_loader, val_loader = get_amazon_4band_loaders(args.data_dir, args.batch_size)
    model = get_model_4band(args.model, in_channels=4).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    save_dir = "./checkpoints_4band"
    os.makedirs(save_dir, exist_ok=True)
    best_f1 = 0.0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | F1: {val_metrics['f1_score']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, os.path.join(save_dir, f"{args.model}_4band_best.pt"))
            print(f"Best model saved! F1: {best_f1:.4f}")
    
    # Load best model and evaluate on test set
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(save_dir, f"{args.model}_4band_best.pt"), 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test set evaluation
    test_img_dir = os.path.join(args.data_dir, "Test", "image")
    test_mask_dir = os.path.join(args.data_dir, "Test", "mask")
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.tif")))
    test_pairs = [(img, os.path.join(test_mask_dir, os.path.basename(img))) 
                  for img in test_images if os.path.exists(os.path.join(test_mask_dir, os.path.basename(img)))]
    
    test_dataset = Amazon4BandDataset([p[0] for p in test_pairs], [p[1] for p in test_pairs])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    model.eval()
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            metrics = compute_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    test_m = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_m['accuracy']*100:.2f}%")
    print(f"  Precision: {test_m['precision']*100:.2f}%")
    print(f"  Recall:    {test_m['recall']*100:.2f}%")
    print(f"  F1 Score:  {test_m['f1_score']*100:.2f}%")
    print(f"  IoU:       {test_m['iou']*100:.2f}%")
    
    # Save results to file
    results_dir = "./evaluation_results_4band"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{args.model}_4band_results.txt"), 'w') as f:
        f.write(f"Model: {args.model} (4-band)\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Random Seed: {args.seed}\n\n")
        f.write(f"Best Val F1: {best_f1*100:.2f}% (Epoch {best_epoch})\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Accuracy:  {test_m['accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {test_m['precision']*100:.2f}%\n")
        f.write(f"  Recall:    {test_m['recall']*100:.2f}%\n")
        f.write(f"  F1 Score:  {test_m['f1_score']*100:.2f}%\n")
        f.write(f"  IoU:       {test_m['iou']*100:.2f}%\n")
    
    print(f"\nTraining complete!")
    print(f"Best Val F1: {best_f1*100:.2f}% | Test F1: {test_m['f1_score']*100:.2f}%")
    print(f"Results saved: {results_dir}/{args.model}_4band_results.txt")


if __name__ == '__main__':
    main()
