"""
DeepGlobe dataset preprocessing script.

Converts 7-class land cover classification to binary forest segmentation:
- Forest class (RGB: 0, 255, 0) -> 1
- All other classes -> 0

Note: Original validation/test sets don't have public labels,
so we split the training data into train/val/test (70/15/15).
"""

import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# Forest class RGB value in original masks
FOREST_RGB = (0, 255, 0)
IMG_SIZE = 512

# Set random seed for reproducibility
random.seed(42)


def rgb_to_binary_mask(mask_rgb):
    """Convert RGB mask to binary forest mask."""
    forest_mask = (
        (mask_rgb[:, :, 0] == FOREST_RGB[0]) &
        (mask_rgb[:, :, 1] == FOREST_RGB[1]) &
        (mask_rgb[:, :, 2] == FOREST_RGB[2])
    )
    return forest_mask.astype(np.uint8)


def process_training_data(src_dir, min_forest_ratio=0.01):
    """
    Process training data and filter samples with minimum forest content.
    
    Args:
        src_dir: Source directory containing *_sat.jpg and *_mask.png files
        min_forest_ratio: Minimum percentage of forest pixels to include sample
    
    Returns:
        List of (image_id, image, binary_mask, forest_ratio) tuples
    """
    sat_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_sat.jpg')])
    valid_samples = []
    
    print(f"Scanning {len(sat_files)} images...")
    
    for sat_file in tqdm(sat_files, desc="Processing"):
        img_id = sat_file.replace('_sat.jpg', '')
        mask_file = f"{img_id}_mask.png"
        mask_path = os.path.join(src_dir, mask_file)
        
        if not os.path.exists(mask_path):
            continue
        
        # Load and resize images
        sat_img = Image.open(os.path.join(src_dir, sat_file)).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')
        
        sat_img = sat_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        
        # Convert to binary forest mask
        binary_mask = rgb_to_binary_mask(np.array(mask_img))
        forest_ratio = binary_mask.sum() / binary_mask.size
        
        # Filter by minimum forest ratio
        if forest_ratio >= min_forest_ratio:
            valid_samples.append((img_id, sat_img, binary_mask, forest_ratio))
    
    print(f"Found {len(valid_samples)} samples with >= {min_forest_ratio*100:.1f}% forest")
    return valid_samples


def save_split(samples, output_dir, split_name):
    """Save samples to output directory."""
    img_dir = os.path.join(output_dir, split_name, "images")
    mask_dir = os.path.join(output_dir, split_name, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for img_id, sat_img, binary_mask, _ in tqdm(samples, desc=f"Saving {split_name}"):
        sat_img.save(os.path.join(img_dir, f"{img_id}.png"))
        # Save mask with values 0 and 255 for visualization
        Image.fromarray(binary_mask * 255).save(os.path.join(mask_dir, f"{img_id}.png"))


def main():
    print("DeepGlobe -> Binary Forest Segmentation\n")
    
    archive_dir = "./archive"
    output_dir = "./deepglobe_processed"
    train_dir = os.path.join(archive_dir, "train")
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found")
        return
    
    # Clean previous output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Process and filter samples
    samples = process_training_data(train_dir, min_forest_ratio=0.01)
    
    if not samples:
        print("No samples with forest found!")
        return
    
    # Shuffle and split: 70% train, 15% val, 15% test
    random.shuffle(samples)
    n = len(samples)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"\nSplit: {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test")
    
    # Save each split
    save_split(train_samples, output_dir, "training")
    save_split(val_samples, output_dir, "validation")
    save_split(test_samples, output_dir, "test")
    
    print(f"\nDone! Output: {output_dir}")
    print(f"Avg forest ratio: {np.mean([s[3] for s in samples])*100:.2f}%")


if __name__ == '__main__':
    main()
