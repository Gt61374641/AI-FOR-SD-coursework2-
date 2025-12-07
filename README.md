# Attention U-Net for Forest Segmentation

PyTorch implementation of "An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery".

## Project Overview

This project consists of two parts:

### 1. Baseline Reproduction
Reproduced the original paper's experiments using PyTorch (converted from TensorFlow/Keras). Trained Attention U-Net on the 4-band Amazon deforestation dataset.

### 2. Contextual Adaptation
Extended the model to the DeepGlobe Land Cover dataset for binary forest segmentation, demonstrating the model's applicability to different geographical contexts.

## Results

| Task | Dataset | Model | F1 Score | IoU |
|------|---------|-------|----------|-----|
| Baseline | Amazon 4-band | Attention U-Net | 94.85% | 90.25% |
| Extension | DeepGlobe | Attention U-Net | 86.03% | 75.53% |

## File Structure

| File | Description |
|------|-------------|
| `train_pytorch.py` | Core module with model architectures (UNet, AttentionUNet, etc.) and utilities |
| `train_4band_amazon.py` | Training script for 4-band Amazon dataset (baseline) |
| `train_deepglobe.py` | Training script for DeepGlobe dataset (extension) |
| `preprocess_deepglobe.py` | Preprocess DeepGlobe: convert 7-class to binary forest segmentation |
| `predict_pytorch.py` | Generate predictions using trained models |
| `requirements_pytorch.txt` | Python dependencies |

## Installation

```bash
# Install PyTorch (CUDA 12.8)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install numpy pillow tqdm scikit-learn rasterio matplotlib
```

## Usage

### Baseline Reproduction (Amazon 4-band)

```bash
# Train
python train_4band_amazon.py --model attention_unet --epochs 50 --batch_size 16 --device cuda

# Predict
python predict_pytorch.py --model attention_unet --checkpoint ./checkpoints_4band/attention_unet_4band_best.pt --input ./AMAZON/Test/image --output ./predictions_4band/ --in_channels 4 --base 16 --device cuda
```

### Contextual Adaptation (DeepGlobe)

```bash
# Step 1: Preprocess data
python preprocess_deepglobe.py

# Step 2: Train
python train_deepglobe.py --model attention_unet --base 32 --augment --scheduler --epochs 50 --batch_size 16 --patience 10 --device cuda

# Step 3: Predict
python predict_pytorch.py --model attention_unet --checkpoint ./checkpoints_deepglobe/attention_unet_deepglobe_best.pt --input ./deepglobe_processed/test/images --output ./predictions_deepglobe/ --in_channels 3 --base 32 --device cuda
```

## References

### Original Paper & Code
- Paper: [An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery](https://www.sciencedirect.com/science/article/pii/S0303243422000113)
- Original Code (TensorFlow): https://github.com/davej23/attention-mechanism-unet

### Datasets
- Amazon 4-band Dataset: https://zenodo.org/record/4498086
- DeepGlobe Land Cover: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
