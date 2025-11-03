# GastroVisionNet - Advanced Endoscopic Image Classification

Copyright (c) 2025 Qatar University

GastroVisionNet is a state-of-the-art deep learning framework for endoscopic image classification with two model variants: GastroVisionNet (full model) and MobileGastroVisionNet (lightweight mobile model).

## Architecture

The framework consists of two main model variants:

1. **GastroVisionNet**: MobileNetV2 backbone with TriadAttention mechanisms, ConvNeXt blocks, and enhanced CBAM attention
2. **MobileGastroVisionNet**: Lightweight MobileNetV2 with MSCA (Mobile Spatial Channel Attention)

Both models are designed for 8-class endoscopic image classification.

## Key Features

-   **Multi-attention mechanisms**: TriadAttention, MSCA, CBAM, and Self-Attention
-   **Advanced preprocessing**: Image normalization and augmentation
-   **Comprehensive evaluation**: Accuracy, Precision, Recall, F1-score, MCC, Kappa, and ROC-AUC with confidence intervals
-   **Cross-validation support**: Built-in 5-fold cross-validation
-   **XAI-ready**: Extensible structure for Grad-CAM and Saliency Map visualization
-   **Production-ready**: Modular codebase with proper error handling and logging

## Model Performance

| Model                 | Precision (%) | Recall (%)   | F1-score (%) | Accuracy (%) | Model Size (MB) | Inference Time (ms) |
| --------------------- | ------------- | ------------ | ------------ | ------------ | --------------- | ------------------- |
| MobileGastroVisionNet | 93.89 ± 1.23  | 93.84 ± 1.23 | 93.87 ± 1.23 | 93.84 ± 1.23 | 30.67           | 0.27                |
| GastroVisionNet       | 94.23 ± 1.14  | 94.19 ± 1.15 | 94.21 ± 1.14 | 94.19 ± 1.15 | 110.14          | 0.32                |

## Project Structure

```
GastroVisionNet/
├── Load_Dataset/
│   ├── __init__.py
│   └── dataset.py               # Dataset loading and preprocessing
├── Preprocessing/
│   ├── __init__.py
│   └── preprocessing.py          # Image transformations
├── Model/
│   ├── __init__.py
│   ├── config.py                # Model configuration
│   ├── gastrovisionnet.py       # Full GastroVisionNet model
│   ├── mobilegastrovisionnet.py # MobileGastroVisionNet model
│   ├── triad_attention.py       # TriadAttention mechanism
│   ├── convnext_block.py        # ConvNeXt blocks
│   ├── channel_spatial_attention.py # CBAM attention
│   └── model.py                 # Model factory
├── Training/
│   ├── __init__.py
│   ├── train_val.py             # Training and validation loops
│   └── optimizer_scheduler.py   # Optimizer and scheduler setup
├── Evaluation/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics
│   ├── plotting.py              # Visualization utilities
│   └── evaluation.py            # Evaluation wrapper
├── XAI/
│   ├── __init__.py
│   ├── Grad-Cam/
│   │   └── __init__.py
│   └── Saliency-Map/
│       └── __init__.py
├── main.py                      # Main training script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Installation

### Requirements

-   Python 3.8+
-   PyTorch 2.0+
-   CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:

```bash
cd GastroVisionNet
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Dataset Structure

Prepare your dataset in the following format:

```
kvasir-v2-5folds/
├── kvasir-v2-dataset-run-1/
│   ├── train/
│   │   ├── normal-cecum/
│   │   ├── dyed-lifted-polyps/
│   │   ├── dyed-resection-margins/
│   │   ├── esophagitis/
│   │   ├── normal-pylorus/
│   │   ├── polyps/
│   │   ├── ulcerative-colitis/
│   │   └── normal-z-line/
│   └── test/
│       ├── normal-cecum/
│       ├── dyed-lifted-polyps/
│       ├── dyed-resection-margins/
│       ├── esophagitis/
│       ├── normal-pylorus/
│       ├── polyps/
│       ├── ulcerative-colitis/
│       └── normal-z-line/
├── kvasir-v2-dataset-run-2/
│   └── ...
├── kvasir-v2-dataset-run-3/
│   └── ...
├── kvasir-v2-dataset-run-4/
│   └── ...
└── kvasir-v2-dataset-run-5/
    └── ...
```


## Usage

### Training

Run the main training script with your desired model:

```bash
python main.py \
    --model gastrovisionnet \
    --base_path /path/to/kvasir-v2-5folds \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --num_workers 0 \
    --image_size 224 224
```

For the mobile model:

```bash
python main.py \
    --model mobilegastrovisionnet \
    --base_path /path/to/kvasir-v2-5folds \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --num_workers 0 \
    --image_size 224 224
```

### Key Parameters

-   `--model`: Model architecture to use (`gastrovisionnet` or `mobilegastrovisionnet`)
-   `--base_path`: Path to the kvasir-v2-5folds directory
-   `--batch_size`: Batch size for training (default: 16)
-   `--num_epochs`: Number of training epochs (default: 100)
-   `--learning_rate`: Learning rate (default: 1e-4)
-   `--weight_decay`: Weight decay for regularization (default: 1e-5)
-   `--num_workers`: Number of data loader workers (default: 0)
-   `--image_size`: Image dimensions as height width (default: 224 224)

### Evaluation

The model automatically evaluates on the test set after training. Results are saved to:

-   `kvasir_v2_classification_results_{model_name}.csv`: Comprehensive evaluation metrics
-   `Best_States/`: Best model weights for each fold
-   `fold_X_plots/`: Training curves, confusion matrix, and ROC curves for each fold

### Output Files

After training, the following files are generated:

```
Best_States/
├── best_model_fold_1.pth
├── best_model_fold_2.pth
├── best_model_fold_3.pth
├── best_model_fold_4.pth
└── best_model_fold_5.pth

fold_1_plots/
├── fold_1_accuracy_curve.pdf
├── fold_1_confusion_matrix.pdf
├── fold_1_loss_curve.pdf
└── fold_1_roc_curve.pdf

... (similar for folds 2-5)

kvasir_v2_classification_results_{model_name}.csv
```

## Model Architecture Details

### GastroVisionNet

-   **Backbone**: MobileNetV2 (ImageNet pretrained)
-   **Attention**: TriadAttention at multiple stages
-   **Enhancements**: ConvNeXt blocks, Channel Attention, Spatial Attention
-   **Classifier**: Fully connected layer with dropout (0.2)

### MobileGastroVisionNet

-   **Backbone**: MobileNetV2 (ImageNet pretrained)
-   **Attention**: MSCA (Mobile Spatial Channel Attention)
-   **Classifier**: Fully connected layers with dropout (0.4)

### Attention Mechanisms

-   **TriadAttention**: Three-way attention (channel, height, spatial)
-   **MSCA**: Lightweight channel attention with global average and max pooling
-   **CBAM**: Convolutional Block Attention Module (channel + spatial)
-   **ConvNeXt Blocks**: Modern architectural patterns with layer scaling

## Preprocessing

The models use standard ImageNet normalization:

-   Mean: [0.485, 0.456, 0.406]
-   Std: [0.229, 0.224, 0.225]
-   Resize: 224x224 with BICUBIC interpolation

## Evaluation Metrics

The framework computes the following metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Matthews Correlation Coefficient (MCC)
-   Cohen's Kappa
-   ROC-AUC (One-vs-Rest)
-   Bootstrap confidence intervals
-   Model size
-   Inference time

## License

Copyright (c) 2025 Qatar University. All rights reserved.

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact:

**Lead By**: Dr. Amith Khandakar  
**Institution**: Qatar University  
**Email**: amitk@qu.edu.qa

## Acknowledgments

This work was supported by Qatar University.
