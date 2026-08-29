# LOB-JEPA: Self-Supervised Joint-Embedding Predictive Architecture for Limit Order Book Representation Learning

This repository contains the official implementation of the baseline benchmarks, reconstruction models, evaluation pipelines, and research suite for **JEPA on Limit Order Book (LOB)** data.

---

## 📊 Phase 1 Baseline Benchmark Results

All 30 baseline experiments (6 model architectures $\times$ 5 stocks) were trained under a 100-timestep reconstruction objective on Level-10 LOB data (40 features). Checkpoints were selected strictly via global minimum validation loss ($\arg\min \mathcal{L}_{\text{val}}$).

### Test Mean Squared Error (MSE)
| Model | sz000001 | sz000002 | sz000858 | sz300147 | sz002415 | Mean (excl. 300147) | Full Mean | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimLOB** | **0.0253** | **0.0924** | **0.0809** | **5.2036** | **0.0503** | **0.0622** | **1.0905** | 🥇 **1st** |
| **Transformer** | 0.0335 | 0.1197 | 0.1132 | 6.0798 | 0.0673 | 0.0834 | 1.2827 | 🥈 **2nd** |
| **LSTM** | 0.0345 | 0.1346 | 0.1275 | 5.8752 | 0.0776 | 0.0936 | 1.2499 | 🥉 **3rd** |
| **TransLOB** | 0.0470 | 0.1880 | 0.1258 | 5.8000 | 0.0907 | 0.1129 | 1.2503 | 4th |
| **DeepLOB** | 0.0623 | 0.2236 | 0.1769 | 5.8972 | 0.1149 | 0.1444 | 1.2950 | 5th |
| **CNN** | 0.0864 | 0.2532 | 0.1893 | 10.2817 | 0.1651 | 0.1735 | 2.1951 | 6th |

### Test Mean Absolute Error (MAE)
| Model | sz000001 | sz000002 | sz000858 | sz300147 | sz002415 | Mean (excl. 300147) | Full Mean | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimLOB** | 0.0778 | **0.1350** | 0.1167 | 0.3611 | **0.0937** | **0.1058** | **0.1569** | 🥇 **1st** |
| **Transformer** | **0.0768** | 0.1501 | **0.1004** | **0.3586** | 0.1068 | 0.1085 | 0.1585 | 🥈 **2nd** |
| **LSTM** | 0.0830 | 0.1659 | 0.1057 | 0.3705 | 0.1180 | 0.1182 | 0.1686 | 🥉 **3rd** |
| **TransLOB** | 0.1041 | 0.2036 | 0.1060 | 0.3812 | 0.1263 | 0.1350 | 0.1842 | 4th |
| **DeepLOB** | 0.1234 | 0.2257 | 0.1236 | 0.3958 | 0.1416 | 0.1536 | 0.2020 | 5th |
| **CNN** | 0.1475 | 0.2341 | 0.1234 | 0.7666 | 0.1661 | 0.1678 | 0.2875 | 6th |

---

## 📁 Repository Structure

```
├── CNN.ipynb                       # Frozen baseline notebook for 1D CNN Encoder
├── DeepLOB.ipynb                   # Frozen baseline notebook for DeepLOB (Zhang et al.)
├── LSTM.ipynb                      # Frozen baseline notebook for Multi-layer LSTM
├── SimLOB.ipynb                    # Frozen baseline notebook for SimLOB (Transformer + FCN)
├── TransLOB.ipynb                  # Frozen baseline notebook for TransLOB (Causal Conv + Transformer)
├── Transformer.ipynb               # Frozen baseline notebook for Pure Transformer Encoder
├── EvaluateAll.ipynb               # Colab execution runner for evaluate_all.py
├── common.py                       # Common data loaders, reconstruction module & architecture utils
├── evaluate_all.py                 # Standalone 30-cell evaluation script (argmin val loss)
├── diagnose_all.py                 # Forensic metadata auditor and raw data inspector
├── evaluation_results.csv          # Raw 30-experiment test metrics (MSE, MAE, ckpt, epoch, val_loss)
├── audit_and_reevaluation_report.md# Formal checkpoint & data audit report
└── .gitignore                      # Excludes heavy datasets and binary checkpoints
```

---

## 🔬 Reproducibility

To re-evaluate all 30 experiments on Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/JEPA_LOB/baselines')
!pip install -q lightning pandas numpy torch
!python evaluate_all.py
```
