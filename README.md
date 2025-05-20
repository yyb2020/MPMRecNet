# MPMRecNet - Recurrence Risk Prediction for Colorectal Cancer

## Overview

**MPMRecNet** is a deep learning model designed to predict recurrence risk in stage I–III colorectal cancer (CRC) patients using multiphoton microscopy (MPM) images. It combines two imaging modalities—TPEF and SHG—to extract structural and cellular features from unstained tissue, using MaxViT encoders and cross-modal attention fusion.

## Key Features

- 🔬 Dual-modality MPM input (TPEF + SHG)
- 🧠 MaxViT-based deep learning with attention pooling
- ⚖️ Focal Loss for class imbalance
- 📈 External validation AUC = **0.849**
- 📊 Nomogram integration with clinical features

## Environment

- Python 3.12 + PyTorch 2.5.1
- Mixed precision training (AMP)
- Key libraries: `torch`, `transformers`, `numpy`, `sklearn`, `matplotlib`

## Dataset

- 1,071 patients (stage I–III CRC)
- MPM images from FFPE samples (2 hospitals in China)
- Data not publicly available — contact authors for access

## Note

This project is for research use only and not for clinical deployment. For more details or collaboration, please contact the authors.
