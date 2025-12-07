"""
Prediction Script for Deforestation Detection - PyTorch Version
================================================================
Use trained models to predict deforestation masks on new images.

Usage:
    python predict_pytorch.py --model attention_unet --checkpoint ./checkpoints/attention_unet_best.pt --input image.tiff --output mask.png
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Try to import rasterio for GeoTIFF support
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Import models from training script
from train_pytorch import (
    UNet, AttentionUNet, ResNet50SegNet, FCN32VGG16, ResUNet, get_model
)


def load_image(path, in_channels=3):
    """Load and preprocess input image (supports GeoTIFF and regular images)"""
    # Load image
    if path.endswith('.npy'):
        img = np.load(path)
        if img.ndim == 4:  # (1, H, W, C) or (1, C, H, W)
            img = img.squeeze(0)
    elif path.lower().endswith(('.tif', '.tiff')) and HAS_RASTERIO:
        # Use rasterio for GeoTIFF files
        with rasterio.open(path) as src:
            img = src.read()  # Returns (C, H, W)
    else:
        # Try PIL for regular images
        try:
            img = np.array(Image.open(path))
        except Exception as e:
            # If PIL fails, try rasterio
            if HAS_RASTERIO:
                with rasterio.open(path) as src:
                    img = src.read()
            else:
                raise RuntimeError(f"Cannot open {path}. Install rasterio: pip install rasterio")
    
    # Normalize
    if img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Handle channel dimension
    if img.ndim == 2:
        img = np.stack([img] * in_channels, axis=0)
    elif img.ndim == 3:
        if img.shape[-1] <= 8 and img.shape[0] > 8:  # HWC format
            img = np.transpose(img, (2, 0, 1))
    
    # Select channels
    if img.shape[0] > in_channels:
        img = img[:in_channels]
    elif img.shape[0] < in_channels:
        # Pad with zeros
        pad = np.zeros((in_channels - img.shape[0], *img.shape[1:]), dtype=img.dtype)
        img = np.concatenate([img, pad], axis=0)
    
    return img.astype(np.float32)


def predict(model, image, device, threshold=0.5):
    """Run prediction on a single image"""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        # Forward pass
        output = model(image)
        
        # Apply sigmoid and threshold
        prob = torch.sigmoid(output)
        mask = (prob > threshold).float()
        
        # Remove batch dimension
        mask = mask.squeeze(0).squeeze(0)
        prob = prob.squeeze(0).squeeze(0)
        
        return mask.cpu().numpy(), prob.cpu().numpy()


def save_mask(mask, output_path, colormap=True):
    """Save prediction mask as image"""
    if colormap:
        # Create colored output (green for forest, red for deforestation)
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[mask == 0] = [34, 139, 34]   # Forest green
        colored[mask == 1] = [255, 0, 0]      # Red for deforestation
        Image.fromarray(colored).save(output_path)
    else:
        # Binary mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(output_path)


def predict_batch(model, image_paths, output_dir, device, in_channels=3, threshold=0.5):
    """Predict on multiple images"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    results = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name = os.path.splitext(filename)[0]
        
        print(f"Processing: {filename}")
        
        try:
            # Load and predict
            image = load_image(img_path, in_channels=in_channels)
            mask, prob = predict(model, image, device, threshold=threshold)
            
            # Save outputs
            save_mask(mask, os.path.join(output_dir, f"{name}_mask.png"), colormap=True)
            save_mask(mask, os.path.join(output_dir, f"{name}_binary.png"), colormap=False)
            np.save(os.path.join(output_dir, f"{name}_prob.npy"), prob)
            
            results.append({
                'filename': filename,
                'deforestation_ratio': mask.mean(),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'filename': filename,
                'status': 'error',
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict deforestation masks')
    
    parser.add_argument('--model', type=str, default='attention_unet',
                        choices=['unet', 'attention_unet', 'resnet50_segnet', 'fcn32_vgg16', 'resunet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./predictions',
                        help='Output path (file or directory)')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--base', type=int, default=16,
                        help='Base filter count (must match training)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model} (base={args.base})")
    if args.model == 'attention_unet':
        model = AttentionUNet(in_ch=args.in_channels, out_ch=1, base=args.base)
    elif args.model == 'unet':
        model = UNet(in_ch=args.in_channels, out_ch=1, base=args.base)
    else:
        model = get_model(args.model, in_channels=args.in_channels)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best F1: {checkpoint.get('best_f1', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Process input
    if os.path.isdir(args.input):
        # Batch prediction
        import glob
        image_paths = (
            glob.glob(os.path.join(args.input, '*.tiff')) +
            glob.glob(os.path.join(args.input, '*.tif')) +
            glob.glob(os.path.join(args.input, '*.png')) +
            glob.glob(os.path.join(args.input, '*.jpg')) +
            glob.glob(os.path.join(args.input, '*.npy'))
        )
        
        print(f"Found {len(image_paths)} images")
        
        results = predict_batch(
            model, image_paths, args.output, device,
            in_channels=args.in_channels, threshold=args.threshold
        )
        
        # Print summary
        success = sum(1 for r in results if r['status'] == 'success')
        print(f"\nCompleted: {success}/{len(results)} images")
        
        avg_ratio = np.mean([r['deforestation_ratio'] for r in results if r['status'] == 'success'])
        print(f"Average deforestation ratio: {avg_ratio:.2%}")
        
    else:
        # Single image
        print(f"Processing: {args.input}")
        
        image = load_image(args.input, in_channels=args.in_channels)
        mask, prob = predict(model, image, device, threshold=args.threshold)
        
        # Determine output path
        if args.output.endswith('.png') or args.output.endswith('.npy'):
            output_path = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{name}_mask.png")
        
        save_mask(mask, output_path, colormap=True)
        
        print(f"Saved mask to: {output_path}")
        print(f"Deforestation ratio: {mask.mean():.2%}")


if __name__ == '__main__':
    main()

