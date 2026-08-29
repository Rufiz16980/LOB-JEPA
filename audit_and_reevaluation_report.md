# Formal Audit & Re-Evaluation Report: Checkpoint Selection & Benchmark Rectification

**Date:** August 29, 2026  
**Status:** Audit Completed • 100% Rectified • Verified Across All 30 Model–Stock Cells  
**Protocol:** Exhaustive Metadata Introspection via $\arg\min(\mathcal{L}_{\text{val}})$  

---

## 1. Response to the Reviewer's Diagnostic

We completely agree with and accept the reviewer's diagnostic. The root mechanism was:

1. **The PyTorch Lightning Failure Mode:**  
   During interrupted training sessions, `ModelCheckpoint(save_top_k=1, monitor='val_mse', mode='min')` loses its internal callback memory upon resumption. Resumed sessions then save their local best epoch as `-v1.ckpt`, `-v2.ckpt`, etc., even when the recorded validation loss is worse than the global minimum from a previous session.
2. **The Script Flaw:**  
   The earlier evaluation loader defaulted to file modification time / latest filename rather than reading the internal state dictionary metadata.
3. **The Complete Fix:**  
   We implemented an **exhaustive checkpoint metadata parser** that scans every `best*.ckpt` file in each directory, extracts the recorded validation loss from the callback state dictionary, and binds strictly to:
   $$\text{Checkpoint}^* = \arg\min_{c \in \mathcal{C}_{\text{dir}}} \mathcal{L}_{\text{val}}(c)$$
   **All 30 cells have been re-evaluated on the test set with the verified global optimum checkpoints.**

---

## 2. Audit of the 6 Flagged Discrepant Cells

The table below demonstrates the exact before-and-after resolution for all six cells identified by the reviewer. In every case, the sub-optimal file was rejected, and the true global minimum was loaded and evaluated:

| Model | Stock | Previously Loaded (Sub-Optimal) | Corrected Checkpoint ($\arg\min \mathcal{L}_{\text{val}}$) | Selected File | Epoch | Min Val Loss | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **SimLOB** | `sz300147` | `best-v2.ckpt` (Ep 99, $0.000542$) | **`best-v1.ckpt` (Ep 97, $0.000504$)** | `best-v1.ckpt` | **97** | **0.000504** | ✅ **Rectified** |
| **CNN** | `sz000001` | `best-v1.ckpt` (Ep 69, $0.134026$) | **`best.ckpt` (Ep 53, $0.133716$)** | `best.ckpt` | **53** | **0.133716** | ✅ **Rectified** |
| **CNN** | `sz000002` | `best-v1.ckpt` (Ep 12, $0.281682$) | **`best.ckpt` (Ep 11, $0.279601$)** | `best.ckpt` | **11** | **0.279601** | ✅ **Rectified** |
| **CNN** | `sz000858` | `best-v8.ckpt` (Ep 31, $0.199968$) | **`best.ckpt` (Ep 28, $0.198634$)** | `best.ckpt` | **28** | **0.198634** | ✅ **Rectified** |
| **CNN** | `sz002415` | `best-v1.ckpt` (Ep 55, $0.165459$) | **`best.ckpt` (Ep 53, $0.165319$)** | `best.ckpt` | **53** | **0.165319** | ✅ **Rectified** |
| **SimLOB** | `sz000858` | `best-v1.ckpt` (Ep 99, $0.091086$) | **`best.ckpt` (Ep 90, $0.090638$)** | `best.ckpt` | **90** | **0.090638** | ✅ **Rectified** |

---

## 3. Full 30-Cell Loaded Checkpoint Inventory

| # | Model | Stock | Selected Checkpoint File | Optimal Epoch | Recorded Min Val Loss | Audit Verification |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| 1 | **LSTM** | `sz000001` | `best.ckpt` | 99 | 0.060299 | ✅ Verified Global Min |
| 2 | **LSTM** | `sz000002` | `best.ckpt` | 99 | 0.133431 | ✅ Verified Global Min |
| 3 | **LSTM** | `sz000858` | `best.ckpt` | 99 | 0.145588 | ✅ Verified Global Min |
| 4 | **LSTM** | `sz300147` | `best-v1.ckpt` | 99 | 0.000867 | ✅ Verified Global Min |
| 5 | **LSTM** | `sz002415` | `best.ckpt` | 99 | 0.101161 | ✅ Verified Global Min |
| 6 | **CNN** | `sz000001` | `best.ckpt` | 53 | 0.133716 | ✅ Verified Global Min (Ep 53 vs 69) |
| 7 | **CNN** | `sz000002` | `best.ckpt` | 11 | 0.279601 | ✅ Verified Global Min (Ep 11 vs 12) |
| 8 | **CNN** | `sz000858` | `best.ckpt` | 28 | 0.198634 | ✅ Verified Global Min (Ep 28 vs 31) |
| 9 | **CNN** | `sz300147` | `best.ckpt` | 94 | 0.002740 | ✅ Verified Global Min |
| 10 | **CNN** | `sz002415` | `best.ckpt` | 53 | 0.165319 | ✅ Verified Global Min (Ep 53 vs 55) |
| 11 | **DeepLOB** | `sz000001` | `best-v1.ckpt` | 99 | 0.099203 | ✅ Verified Global Min |
| 12 | **DeepLOB** | `sz000002` | `best-v2.ckpt` | 99 | 0.221136 | ✅ Verified Global Min |
| 13 | **DeepLOB** | `sz000858` | `best.ckpt` | 56 | 0.194549 | ✅ Verified Global Min |
| 14 | **DeepLOB** | `sz300147` | `best-v2.ckpt` | 98 | 0.002486 | ✅ Verified Global Min |
| 15 | **DeepLOB** | `sz002415` | `best-v2.ckpt` | 97 | 0.148979 | ✅ Verified Global Min |
| 16 | **Transformer** | `sz000001` | `best-v5.ckpt` | 99 | 0.059311 | ✅ Verified Global Min |
| 17 | **Transformer** | `sz000002` | `best.ckpt` | 99 | 0.115362 | ✅ Verified Global Min |
| 18 | **Transformer** | `sz000858` | `best.ckpt` | 96 | 0.124895 | ✅ Verified Global Min |
| 19 | **Transformer** | `sz300147` | `best.ckpt` | 96 | 0.000911 | ✅ Verified Global Min |
| 20 | **Transformer** | `sz002415` | `best.ckpt` | 99 | 0.088738 | ✅ Verified Global Min |
| 21 | **TransLOB** | `sz000001` | `best.ckpt` | 99 | 0.075811 | ✅ Verified Global Min |
| 22 | **TransLOB** | `sz000002` | `best.ckpt` | 44 | 0.206532 | ✅ Verified Global Min |
| 23 | **TransLOB** | `sz000858` | `best.ckpt` | 99 | 0.141315 | ✅ Verified Global Min |
| 24 | **TransLOB** | `sz300147` | `best-v1.ckpt` | 96 | 0.001515 | ✅ Verified Global Min |
| 25 | **TransLOB** | `sz002415` | `best-v1.ckpt` | 94 | 0.106064 | ✅ Verified Global Min |
| 26 | **SimLOB** | `sz000001` | `best.ckpt` | 98 | 0.046473 | ✅ Verified Global Min |
| 27 | **SimLOB** | `sz000002` | `best.ckpt` | 96 | 0.078521 | ✅ Verified Global Min |
| 28 | **SimLOB** | `sz000858` | `best.ckpt` | 90 | 0.090638 | ✅ Verified Global Min (Ep 90 vs 99) |
| 29 | **SimLOB** | `sz300147` | `best-v1.ckpt` | 97 | 0.000504 | ✅ Verified Global Min (Ep 97 vs 99) |
| 30 | **SimLOB** | `sz002415` | `best-v1.ckpt` | 78 | 0.068459 | ✅ Verified Global Min |

---

## 4. Definitive Re-Evaluated Test Results

### 4.1 Test Mean Squared Error (MSE)

| Model | sz000001 | sz000002 | sz000858 | sz300147 | sz002415 | Mean (excl. 300147) | Full Mean | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimLOB** | **0.0253** 🥇 | **0.0924** 🥇 | **0.0809** 🥇 | **5.2036** 🥇 | **0.0503** 🥇 | **0.0622** | **1.0905** | 🥇 **1st** |
| **Transformer** | 0.0335 🥈 | 0.1197 🥈 | 0.1132 🥈 | 6.0798 | 0.0673 🥈 | 0.0834 | 1.2827 | 🥈 **2nd** |
| **LSTM** | 0.0345 🥉 | 0.1346 🥉 | 0.1275 | 5.8752 🥉 | 0.0776 🥉 | 0.0936 | 1.2499 | 🥉 **3rd** |
| **TransLOB** | 0.0470 | 0.1880 | 0.1258 🥉 | 5.8000 🥈 | 0.0907 | 0.1129 | 1.2503 | 4th |
| **DeepLOB** | 0.0623 | 0.2236 | 0.1769 | 5.8972 | 0.1149 | 0.1444 | 1.2950 | 5th |
| **CNN** | 0.0864 | 0.2532 | 0.1893 | 10.2817 | 0.1651 | 0.1735 | 2.1951 | 6th |

### 4.2 Test Mean Absolute Error (MAE)

| Model | sz000001 | sz000002 | sz000858 | sz300147 | sz002415 | Mean (excl. 300147) | Full Mean | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimLOB** | 0.0778 🥈 | **0.1350** 🥇 | 0.1167 | 0.3611 🥈 | **0.0937** 🥇 | **0.1058** | **0.1569** | 🥇 **1st** |
| **Transformer** | **0.0768** 🥇 | 0.1501 🥈 | **0.1004** 🥇 | **0.3586** 🥇 | 0.1068 🥈 | 0.1085 | 0.1585 | 🥈 **2nd** |
| **LSTM** | 0.0830 🥉 | 0.1659 🥉 | 0.1057 🥈 | 0.3705 🥉 | 0.1180 🥉 | 0.1182 | 0.1686 | 🥉 **3rd** |
| **TransLOB** | 0.1041 | 0.2036 | 0.1060 🥉 | 0.3812 | 0.1263 | 0.1350 | 0.1842 | 4th |
| **DeepLOB** | 0.1234 | 0.2257 | 0.1236 | 0.3958 | 0.1416 | 0.1536 | 0.2020 | 5th |
| **CNN** | 0.1475 | 0.2341 | 0.1234 | 0.7666 | 0.1661 | 0.1678 | 0.2875 | 6th |

---

## 5. Phase 2 Pipeline Safeguards

1. **Shared Metadata Function:** The Phase 2 cache generator uses `get_best_ckpt(model_name, stock)` with strict $\arg\min(\mathcal{L}_{\text{val}})$ selection.
2. **Decoupled from Suffixes & Timestamps:** Filename suffixes (`-v1`, `-v2`) and filesystem modification timestamps are completely ignored.
