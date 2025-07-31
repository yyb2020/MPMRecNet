# MPMRecNet - Recurrence Risk Prediction for Colorectal Cancer

## Overview

**MPMRecNet** is a deep learning model designed to predict recurrence risk in stage I–III colorectal cancer (CRC) patients using multiphoton microscopy (MPM) images. It combines two imaging modalities—TPEF and SHG—to extract structural and cellular features from unstained tissue, using MaxViT encoders and cross-modal attention fusion.

## Key Features

- Dual-modality MPM input (TPEF + SHG)
- MaxViT-based deep learning with attention pooling
- Focal Loss for class imbalance
- External validation AUC = **0.849**
- Nomogram integration with clinical features

## Training

### 1. **Standard Training**

```
python train/train.py \
  --train_csv  \
  --pt_dir \
  --model_output \
  --device cuda
```

### 2. **10-Fold Cross-Validation**

```
python train/crossval_train.py \
  --train_csv \
  --pt_dir \
  --output_dir \
  --device cuda \
  --batch_size 8
```

## Inference

```
python train/inference.py \
  --file_path  \
  --model_path  \
  --device cuda
```

## Note

This project is for research use only and not for clinical deployment. For more details or collaboration, please contact the authors.
