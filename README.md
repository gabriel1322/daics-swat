# Implementation of DAICS on SWaT

A university project implementing **DAICS** for anomaly detection on the **SWaT** dataset, as described by Abdelaty et al. in their 2021 IEEE paper "DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems".

## Overview

This repository provides a **paper-aligned reproduction** of the DAICS architecture for industrial anomaly detection on the SWaT dataset.

The complete pipeline includes:

1. **WDNN (Weighted Deep Neural Network)**  
   → Multi-step time-series prediction model trained on *normal-only* data.

2. **TTNN (Threshold Tuning Neural Network)**  
   → Learns the dynamics of prediction errors on validation data.

3. **Adaptive threshold tuning (Algorithm 1)**  
   → Computes section-wise decision thresholds.

4. **Window-level detection with W<sub>anom</sub> logic**  
   → Reduces false alarms by enforcing anomaly persistence.

The implementation follows the notation and structure of the paper.

---

## Installation & Requirements

*Note: project was tested on Python 3.10.12*

### 1️⃣ Clone this repository
```bash
git clone https://github.com/gabriel1322/daics-swat.git
```
Navigate to the project directory:
```bash
cd daics-swat
```

### 2️⃣ Create a Virtual Environment (recommended)
```bash
python -m venv .venv
```

Activate it:
#### Windows
```bash
.\.venv\Scripts\activate
```

#### Linux / macOS
```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

# A. Full Pipeline
*Important: if you only want to evaluate detection, you can skip to section 6. The models are already trained with default values and saved in .pt files under the /runs folder.*

**Hyperparameters values can manually be changed in /configs/base.yaml.**
## Data Protocol Used

- **Training set** → Normal data only  
- **Validation set** → Normal data only  
- **Test set** → Merged data

The model is trained and validated exclusively on the normal dataset, following the one-class learning setup described in the paper.
For evaluation, we use the official SWaT merged dataset, which contains both normal and attack periods in chronological order (≈3.8 attack ratio).
This ensures a realistic test scenario where the model is exposed to long normal behavior segments followed by real attack sequences.

## 1. Preprocessing the SWaT dataset (optional)
The code already contains the processed parquet files, **so you can skip preprocessing**.

If you want to do it by yourself, you should download the official SWaT CSV files (see *References*):

- `normal.csv`
- `merged.csv`

Place them inside the */data* folder and run the following command: 
```bash
python scripts/preprocess_swat.py --config configs/base.yaml
```

## 2. Sanity Check Windowing
```bash
python scripts/sanity_check_windows.py --config configs/base.yaml
```
You might see *"UserWarning: 'pin_memory' argument is set as true but no accelerator is found"*, it simply means CUDA is not available and it is safe to ignore.

## 3. Train WDNN Model
```bash
python scripts/train_wdnn.py --config configs/base.yaml
```
*Number of epochs:* **100**

*Outputs:*
- *runs/wdnn/best.pt*
- *runs/wdnn/last.pt*

*best.pt* is the checkpoint with best validation loss while *last.p*t is the final epoch model. You should use *best.pt* for detection.

## 4. Train TTNN Model
```bash
python scripts/train_ttnn.py --config configs/base.yaml --wdnn_ckpt runs/wdnn/best.pt
```

*Number of epochs:* **1**

*Output: runs/ttnn/section_0.pt*

We use *G = 1* section (all sensors grouped).

## 5. Tune Thresholds (Algorithm 1)
```bash
python scripts/tune_thresholds.py --config configs/base.yaml --wdnn_ckpt runs/wdnn/best.pt --ttnn_ckpt runs/ttnn/section_0.pt
```

*Output: runs/thresholds.json*

This script implements Algorithm 1 from the paper by computing the adaptive decision threshold T<sub>g</sub>, using the validation prediction error dynamics modeled by the TTNN.
The Few-Steps Learning Algorithm (Algorithm 2) can also be enabled during threshold tuning with an additional flag *"--few_steps N"* for N adaptation steps. Here, Few-Steps produces only marginal variations of T<sub>g</sub> and does not affect final detection metrics.

## 6. Evaluate Detection
```bash
python scripts/eval_detect.py --config configs/base.yaml --wdnn_ckpt runs/wdnn/best.pt --thresholds runs/thresholds.json
```

# B. Results

## 1. Model Hyperparameters and Evaluation Metrics
*Note: we reuse the sames values as defined in the paper, except for W<sub>anom</sub>*
<table>
<tr>
<td valign="top" width="50%">

<h3>Window Parameters</h3>

| Parameter | Value |
|------------|--------|
| W<sub>in</sub> | 60 |
| W<sub>out</sub> | 4 |
| H | 50 |
| S | 1 |
| W<sub>anom</sub> | 200 |
| W<sub>grace</sub> | 0 |

</td>
<td valign="top" width="50%">

<h3>WDNN Hyperparameters</h3>

| Parameter | Value |
|------------|--------|
| Optimizer | SGD |
| Learning rate | 0.001 |
| Epochs | 100 |
| Loss | MSE |
| DL1 neurons | 3 × W<sub>in</sub> |
| DL2 neurons | 3 × m |
| DL4 neurons | 80 |
| CL1 channels | 64 |
| CL1 kernel size | 2 |
| CL2 channels | 128 |
| CL2 kernel size | 2 |
| Activation | LeakyReLU (0.01) |

</td>
</tr>

<tr>
<td valign="top" width="50%">

<h3>TTNN Hyperparameters</h3>

| Parameter | Value |
|------------|--------|
| Optimizer | SGD |
| Learning rate | 0.01 |
| Epochs | 1 |
| Batch size | 32 |
| Median kernel | 59 |
| Activation | LeakyReLU (0.01) |

</td>
<td valign="top" width="50%">

<h3>Detection Performance (Window-Level)</h3>

| Model | Precision | Recall | F1-score |
|------|----------|--------|----------|
| Our model (after W<sub>anom</sub>) | 0.9162 | 0.6044 | 0.7283 |
| Paper's model | 0.9185 | 0.8616 | 0.8892 |

</td>
</tr>
</table>

## 2. Plots

To generate analysis plots:
```bash
python scripts/plot_detection_analysis.py --config configs/base.yaml --wdnn_ckpt runs/wdnn/best.pt --thresholds runs/thresholds.json --tag paper_default --out_dir runs/plots
```
*Outputs: runs/plots/*

## Project Structure
```
daics-swat/
│
├── configs/
│   └── base.yaml                      # Global configuration file
│
├── data/
│   ├── normal.csv                     # Original SWaT normal CSV
│   ├── merged.csv                     # Original SWaT merged CSV
│   ├── processed_swat_normal.parquet
│   └── processed_swat_merged.parquet
│
├── scripts/
│   ├── preprocess_swat.py             # CSV → Parquet preprocessing
│   ├── sanity_check_windows.py        # Validates window shapes and splits
│   ├── train_wdnn.py                  # WDNN training
│   ├── train_ttnn.py                  # TTNN training
│   ├── tune_thresholds.py             # Threshold computation (Algorithm 1)
│   ├── eval_detect.py                 # Final detection evaluation
│   └── plot_detection_analysis.py     # Plot generation
│
├── src/daics/
│   ├── models/
│   │   ├── wdnn.py                    # WDNN architecture
│   │   └── ttnn.py                    # TTNN architecture
│   ├── train/
│   │   ├── wdnn_trainer.py            # WDNN training loop
│   │   └── ttnn_trainer.py            # TTNN training loop
│   ├── eval/
│   │   ├── mse.py                     # MSE computation
│   │   ├── thresholds.py              # Threshold tuning logic
│   │   └── detect.py                  # Wanom detection logic
│   └── data/
│       ├── swat.py                    # SWaT-specific utilities
│       ├── dataloaders.py             # Paper-strict splits
│       └── windowing.py               # Sliding window logic
│
├── runs/                              # Checkpoints & outputs
│   ├── wdnn/
│   ├── ttnn/
│   └── thresholds.json
│
├── requirements.txt
└── README.md
```

## References

**Primary paper**

- M. Abdelaty, R. Doriguzzi-Corin, and D. Siracusa,
“DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems,”
IEEE Transactions on Emerging Topics in Computing, 2021.

**Dataset**

- [SWaT Dataset: Secure Water Treatment System (Normal & Attack Scenarios)](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) - Uploader: Vishal Agrawal

**Related Implementations / Public Notebooks (SWaT-Based Work)**

These public notebooks were reviewed for contextual understanding and alternative approaches to SWaT anomaly detection.

- ngoclesydney,
[Anomaly Detection with SWaT Dataset](https://github.com/ngoclesydney/Anomaly-Detection-with-Swat-Dataset/blob/master/Anomaly_3_attacks_25_01.ipynb)

- scarss,
[Anomaly Detection for Industrial Control Systems](https://www.kaggle.com/code/scarss/anomaly-detection-for-industrial-control-systems)

- kashifNazirntuf,
[Revised Version](https://www.kaggle.com/code/kashifnazirntuf/revised-version)

- muhammadabrar78,
[Blockchain Waseem](https://www.kaggle.com/code/muhammadabrar78/blockchain-waseem)

- Vishal Agrawal,
[notebook772eb39999](https://www.kaggle.com/code/vishala28/notebook772eb39999)

- Đoàn Ngọc Bảo,
[gdnv3](https://www.kaggle.com/code/baodoandev/gdnv3)
