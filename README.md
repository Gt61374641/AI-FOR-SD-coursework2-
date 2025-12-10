# Attention U-Net for Forest Segmentation

PyTorch implementation of "An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery".

## Project Overview

This project consists of two parts:

### 1. Baseline Reproduction
Reproduced the original paper's experiments using PyTorch (converted from TensorFlow/Keras). Implemented and trained **all five models** from the original paper on the 4-band Amazon deforestation dataset:

- **Attention U-Net** (Main contribution of the paper)
- U-Net
- ResNet50-SegNet
- FCN32-VGG16
- ResUNet

### 2. Contextual Adaptation
Extended the Attention U-Net model to the DeepGlobe Land Cover dataset for binary forest segmentation, demonstrating the model's applicability to different geographical contexts.

## Results

| Task | Dataset | Model | F1 Score | IoU |
|------|---------|-------|----------|-----|
| Baseline | Amazon 4-band | Attention U-Net | 94.85% | 90.25% |
| Extension | DeepGlobe | Attention U-Net | 86.03% | 75.53% |

## Implemented Models

| Model | Description |
|-------|-------------|
| Attention U-Net | U-Net with attention gates for improved feature selection |
| U-Net | Standard encoder-decoder with skip connections |
| ResNet50-SegNet | ResNet-style encoder with SegNet-style decoder |
| FCN32-VGG16 | Fully Convolutional Network with VGG16 backbone |
| ResUNet | U-Net with residual blocks for better gradient flow |

## File Structure

| File | Description |
|------|-------------|
| `train_pytorch.py` | Core module with all model architectures and utilities |
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

### Training on Amazon 4-band Dataset (Baseline)

```bash
# Attention U-Net (main model)
python train_4band_amazon.py --model attention_unet --epochs 50 --batch_size 16 --device cuda

# Other models
python train_4band_amazon.py --model unet --epochs 50 --batch_size 16 --device cuda
python train_4band_amazon.py --model resnet50_segnet --epochs 50 --batch_size 16 --device cuda
python train_4band_amazon.py --model fcn32_vgg16 --epochs 50 --batch_size 16 --device cuda
python train_4band_amazon.py --model resunet --epochs 50 --batch_size 16 --device cuda
```

### Training on DeepGlobe Dataset (Contextual Adaptation)

```bash
# Step 1: Preprocess data
python preprocess_deepglobe.py

# Step 2: Train
python train_deepglobe.py --model attention_unet --base 32 --augment --scheduler --epochs 50 --batch_size 16 --patience 10 --device cuda
```

### Prediction

```bash
# Amazon dataset
python predict_pytorch.py --model attention_unet --checkpoint ./checkpoints_4band/attention_unet_4band_best.pt --input ./AMAZON/Test/image --output ./predictions_amazon/ --in_channels 4 --base 16 --device cuda

# DeepGlobe dataset
python predict_pytorch.py --model attention_unet --checkpoint ./checkpoints_deepglobe/attention_unet_deepglobe_best.pt --input ./deepglobe_processed/test/images --output ./predictions_deepglobe/ --in_channels 3 --base 32 --device cuda
```

## References

### Original Paper & Code
- Paper: [An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery](https://www.sciencedirect.com/science/article/pii/S0303243422000113)
- Original Code (TensorFlow): https://github.com/davej23/attention-mechanism-unet

### Datasets
- Amazon 4-band Dataset: https://zenodo.org/record/4498086
- DeepGlobe Land Cover: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
