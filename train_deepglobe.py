"""
Training script for DeepGlobe Land Cover dataset - Binary forest segmentation.

Converts 7-class land cover classification to binary forest/non-forest segmentation
for contextual adaptation experiments.
"""

import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

from train_pytorch import AttentionUNet, UNet, CombinedLoss, compute_metrics, set_seed


class DeepGlobeDataset(Dataset):
    """
    Dataset for DeepGlobe forest segmentation.
    Supports optional data augmentation (flip, rotate).
    """
    def __init__(self, image_dir, mask_dir, img_size=512, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.images)
    
    def _augment(self, img, mask):
        """Apply random augmentations to image and mask."""
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('L')
        
        # Resize to target size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Apply augmentation if enabled
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # Convert to tensors and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        mask_array = (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        
        return img_tensor, mask_tensor


def get_data_loaders(data_dir="./deepglobe_processed", batch_size=8, augment=False):
    """Create training and validation data loaders."""
    train_img = os.path.join(data_dir, "training", "images")
    train_mask = os.path.join(data_dir, "training", "masks")
    val_img = os.path.join(data_dir, "validation", "images")
    val_mask = os.path.join(data_dir, "validation", "masks")
    
    for d in [train_img, train_mask, val_img, val_mask]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    
    # Augmentation only for training set
    train_dataset = DeepGlobeDataset(train_img, train_mask, augment=augment)
    val_dataset = DeepGlobeDataset(val_img, val_mask, augment=False)
    
    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def get_test_loader(data_dir="./deepglobe_processed", batch_size=8):
    """Create test data loader."""
    test_dataset = DeepGlobeDataset(
        os.path.join(data_dir, "test", "images"),
        os.path.join(data_dir, "test", "masks")
    )
    print(f"Test samples: {len(test_dataset)}")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
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


def evaluate_test(model, loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            metrics = compute_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    return {k: np.mean(v) for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Train on DeepGlobe dataset')
    parser.add_argument('--data_dir', type=str, default='./deepglobe_processed')
    parser.add_argument('--model', type=str, default='attention_unet', choices=['attention_unet', 'unet'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pos_weight', type=float, default=3.0, help='Weight for positive class (forest)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--base', type=int, default=16, help='Base filter count for model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() 
                          else args.device if args.device != 'auto' else 'cpu')
    
    print(f"\nDeepGlobe Forest Segmentation")
    print(f"Model: {args.model} | Base: {args.base} | Device: {device}")
    print(f"Augment: {args.augment} | Scheduler: {args.scheduler}\n")
    
    # Load data
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, args.augment)
    
    # Create model
    if args.model == 'attention_unet':
        model = AttentionUNet(in_ch=3, out_ch=1, base=args.base).to(device)
    else:
        model = UNet(in_ch=3, out_ch=1, base=args.base).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = CombinedLoss(pos_weight=args.pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Optional learning rate scheduler
    scheduler = None
    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    save_dir = "./checkpoints_deepglobe"
    os.makedirs(save_dir, exist_ok=True)
    
    best_f1, best_epoch, no_improve = 0.0, 0, 0
    
    # Training loop with early stopping
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_m = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1_score']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1_score']:.4f} | IoU: {val_m['iou']:.4f}")
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_m['f1_score'])
        
        # Save best model
        if val_m['f1_score'] > best_f1:
            best_f1, best_epoch, no_improve = val_m['f1_score'], epoch, 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_m
            }, os.path.join(save_dir, f"{args.model}_deepglobe_best.pt"))
            print(f"Best model saved! F1: {best_f1:.4f}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s)")
        
        # Early stopping
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Evaluate on test set
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(save_dir, f"{args.model}_deepglobe_best.pt"), 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = get_test_loader(args.data_dir, args.batch_size)
    test_m = evaluate_test(model, test_loader, device)
    
    # Print results
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_m['accuracy']*100:.2f}%")
    print(f"  Precision: {test_m['precision']*100:.2f}%")
    print(f"  Recall:    {test_m['recall']*100:.2f}%")
    print(f"  F1 Score:  {test_m['f1_score']*100:.2f}%")
    print(f"  IoU:       {test_m['iou']*100:.2f}%")
    
    # Save results to file
    results_dir = "./evaluation_results_deepglobe"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"deepglobe_{args.model}_results.txt"), 'w') as f:
        f.write(f"Model: {args.model}\nBase: {args.base}\nBatch Size: {args.batch_size}\n")
        f.write(f"LR: {args.lr}\nPos Weight: {args.pos_weight}\nAugment: {args.augment}\n\n")
        f.write(f"Best Val F1: {best_f1*100:.2f}% (Epoch {best_epoch})\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Accuracy:  {test_m['accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {test_m['precision']*100:.2f}%\n")
        f.write(f"  Recall:    {test_m['recall']*100:.2f}%\n")
        f.write(f"  F1 Score:  {test_m['f1_score']*100:.2f}%\n")
        f.write(f"  IoU:       {test_m['iou']*100:.2f}%\n")
    
    print(f"\nBest Val F1: {best_f1*100:.2f}% | Test F1: {test_m['f1_score']*100:.2f}%")


if __name__ == '__main__':
    main()
