"""
Semantic Segmentation Models for Satellite Image Analysis

Models implemented:
- UNet: Standard encoder-decoder with skip connections
- AttentionUNet: UNet with attention gates for feature refinement
- ResNet50SegNet: ResNet encoder with SegNet decoder
- FCN32VGG16: Fully Convolutional Network with VGG16 backbone
- ResUNet: UNet with residual blocks
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import GradScaler, autocast


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Attention gate for focusing on relevant features.
    g: gating signal from decoder, x: skip connection from encoder
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # Resize gating signal if dimensions don't match
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # Attention coefficients
        return x * psi


# =============================================================================
# UNet
# =============================================================================

class UNet(nn.Module):
    """Standard U-Net with encoder-decoder structure and skip connections."""
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.bottleneck = ConvBlock(base*8, base*16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
        
        self.out = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        # Decoder path with skip connections
        d4 = self.up4(b)
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out(d1)


# =============================================================================
# Attention U-Net
# =============================================================================

class AttentionUNet(nn.Module):
    """U-Net with attention gates for improved feature selection."""
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.bottleneck = ConvBlock(base*8, base*16)
        
        # Attention gates
        self.att4 = AttentionBlock(F_g=base*8, F_l=base*8, F_int=base*4)
        self.att3 = AttentionBlock(F_g=base*4, F_l=base*4, F_int=base*2)
        self.att2 = AttentionBlock(F_g=base*2, F_l=base*2, F_int=base)
        self.att1 = AttentionBlock(F_g=base, F_l=base, F_int=max(base//2, 1))
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
        
        self.out = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention-weighted skip connections
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        return self.out(d1)


# =============================================================================
# ResNet50-SegNet
# =============================================================================

class ResNet50SegNet(nn.Module):
    """ResNet-style encoder with SegNet-style decoder."""
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        # Encoder (ResNet-style)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.enc2 = self._make_layer(base, base*2, 2)
        self.enc3 = self._make_layer(base*2, base*4, 2)
        self.enc4 = self._make_layer(base*4, base*8, 2)
        self.enc5 = self._make_layer(base*8, base*16, 2)
        
        # Decoder (SegNet-style upsampling)
        self.dec5 = self._make_decoder_layer(base*16, base*8)
        self.dec4 = self._make_decoder_layer(base*8, base*4)
        self.dec3 = self._make_decoder_layer(base*4, base*2)
        self.dec2 = self._make_decoder_layer(base*2, base)
        self.dec1 = self._make_decoder_layer(base, base//2)
        self.out = nn.Conv2d(base//2, out_ch, 1)
    
    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder with bilinear upsampling to match sizes
        d5 = F.interpolate(self.dec5(e5), size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.dec4(d5), size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.dec3(d4), size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = F.interpolate(self.dec2(d3), size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = F.interpolate(self.dec1(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return self.out(d1)


# =============================================================================
# FCN32-VGG16
# =============================================================================

class FCN32VGG16(nn.Module):
    """Fully Convolutional Network with VGG16-style encoder."""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # VGG16-style feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        # FC layers converted to convolutions
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, out_ch, 1),
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        x = self.features(x)
        x = self.classifier(x)
        # Upsample to original resolution
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)


# =============================================================================
# ResUNet
# =============================================================================

class ResBlock(nn.Module):
    """Residual block with shortcut connection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        # Shortcut for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.relu(out)


class ResUNet(nn.Module):
    """U-Net with residual blocks for better gradient flow."""
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        # Encoder
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base*2, stride=2)
        self.enc3 = ResBlock(base*2, base*4, stride=2)
        self.enc4 = ResBlock(base*4, base*8, stride=2)
        self.bridge = ResBlock(base*8, base*16, stride=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ResBlock(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ResBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ResBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ResBlock(base*2, base)
        
        self.out = nn.Conv2d(base, out_ch, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bridge(e4)
        
        # Decoder with skip connections
        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out(d1)


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss with optional class weighting."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if pos_weight != 1.0:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        if hasattr(self.bce, 'pos_weight') and self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(pred, target, threshold=0.5):
    """Compute segmentation metrics: accuracy, precision, recall, F1, IoU."""
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        pred_flat = pred_binary.cpu().numpy().flatten()
        target_flat = target.cpu().numpy().flatten()
        
        # Handle edge case where all pixels are same class
        if len(np.unique(target_flat)) == 1 and len(np.unique(pred_flat)) == 1:
            if target_flat[0] == pred_flat[0]:
                return {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'iou': 1.0}
        
        acc = accuracy_score(target_flat, pred_flat)
        prec = precision_score(target_flat, pred_flat, zero_division=0)
        rec = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        
        # IoU calculation
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'iou': iou}


# =============================================================================
# Dataset
# =============================================================================

class DeforestationDataset(Dataset):
    """Dataset for loading image-mask pairs for segmentation."""
    def __init__(self, image_dir, mask_dir, img_size=512, in_channels=3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.images = sorted([f for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Try to find mask with different extension if not found
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.tif', '.tiff']:
                test_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(test_path):
                    mask_path = test_path
                    break
        
        # Load and preprocess image
        img = Image.open(img_path)
        if self.in_channels == 3:
            img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Load and preprocess mask
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_array = np.array(mask, dtype=np.float32)
        if mask_array.max() > 1:
            mask_array = mask_array / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        # Convert to tensors (C, H, W)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * self.in_channels, axis=0)
        else:
            img_array = img_array.transpose(2, 0, 1)
        
        return torch.from_numpy(img_array), torch.from_numpy(mask_array).unsqueeze(0)


def get_data_loaders(data_dir, batch_size=8, in_channels=3):
    """Create training and validation data loaders."""
    # Try different directory structures
    train_img_dir = os.path.join(data_dir, "train", "images")
    train_mask_dir = os.path.join(data_dir, "train", "masks")
    val_img_dir = os.path.join(data_dir, "val", "images")
    val_mask_dir = os.path.join(data_dir, "val", "masks")
    
    if not os.path.exists(train_img_dir):
        train_img_dir = os.path.join(data_dir, "training", "images")
        train_mask_dir = os.path.join(data_dir, "training", "masks")
        val_img_dir = os.path.join(data_dir, "validation", "images")
        val_mask_dir = os.path.join(data_dir, "validation", "masks")
    
    train_dataset = DeforestationDataset(train_img_dir, train_mask_dir, in_channels=in_channels)
    val_dataset = DeforestationDataset(val_img_dir, val_mask_dir, in_channels=in_channels)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, val_loader


def get_model(model_name, in_channels=3):
    """Factory function to create model by name."""
    models = {
        'unet': lambda: UNet(in_ch=in_channels, out_ch=1, base=32),
        'attention_unet': lambda: AttentionUNet(in_ch=in_channels, out_ch=1, base=16),
        'resnet50_segnet': lambda: ResNet50SegNet(in_ch=in_channels, out_ch=1),
        'fcn32_vgg16': lambda: FCN32VGG16(in_ch=in_channels, out_ch=1),
        'resunet': lambda: ResUNet(in_ch=in_channels, out_ch=1, base=32),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]()


# =============================================================================
# Training Functions
# =============================================================================

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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train segmentation models')
    parser.add_argument('--model', type=str, default='attention_unet', 
                        choices=['unet', 'attention_unet', 'resnet50_segnet', 'fcn32_vgg16', 'resunet'])
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() 
                          else args.device if args.device != 'auto' else 'cpu')
    
    print(f"\nModel: {args.model} | Device: {device} | Epochs: {args.epochs}\n")
    
    # Load data and create model
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, args.in_channels)
    model = get_model(args.model, args.in_channels).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_f1 = 0.0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | F1: {val_metrics['f1_score']:.4f}")
        
        # Save best model
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_f1': best_f1}, 
                      os.path.join(save_dir, f"{args.model}_best.pt"))
            print(f"Best model saved! F1: {best_f1:.4f}")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
