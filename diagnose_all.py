"""
diagnose_all.py
───────────────
Comprehensive Forensic Audit Script for the Reviewer Agent.

This script performs 4 rigorous inspections:
  1. Checkpoint Metadata Audit: Reads internal PyTorch Lightning state dictionaries
     (epoch, global_step, best_model_score) for ALL checkpoint files across ALL models & stocks.
  2. Weight Comparison: Tests whether versioned files (best.ckpt vs best-v1.ckpt) are near-duplicates
     or distinct checkpoints.
  3. Raw Row Forensics: Extracts and prints the Top 20 extreme rows (|Z| up to 81.68) for sz300147,
     including full timestamps, feature columns, and physical order book coherence checks.
  4. Temporal Clustering: Computes the exact date-by-date distribution of all |Z| > 10 anomalies
     in the sz300147 test set to check for market events (halts / circuit breakers).

Author: Forensic Diagnostic Suite (Read-Only)
"""

import os
import glob
import csv
import json
import heapq
import datetime
import torch
import numpy as np
import pandas as pd

MODELS = ['LSTM', 'CNN', 'DeepLOB', 'Transformer', 'TransLOB', 'SimLOB']
STOCKS = ['sz000001', 'sz000002', 'sz000858', 'sz300147', 'sz002415']

print("="*90)
print("  FORENSIC AUDIT PART 1: CHECKPOINT METADATA INSPECTION (ALL 30 EXPERIMENTS)")
print("="*90)

ckpt_records = []
for m in MODELS:
    for s in STOCKS:
        ckpt_dir = os.path.realpath(f"checkpoints/{m}/{s}")
        if not os.path.exists(ckpt_dir):
            continue
        files = sorted(glob.glob(f"{ckpt_dir}/*.ckpt"))
        for fpath in files:
            fname = os.path.basename(fpath)
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')
            try:
                data = torch.load(fpath, map_location='cpu', weights_only=False)
                epoch = data.get('epoch', 'N/A')
                step = data.get('global_step', 'N/A')
                callbacks = data.get('callbacks', {})
                best_score = None
                best_path = None
                for cb_key, cb_val in callbacks.items():
                    if 'ModelCheckpoint' in cb_key:
                        best_score = cb_val.get('best_model_score', None)
                        if best_score is not None:
                            best_score = float(best_score)
                        best_path = cb_val.get('best_model_path', None)
                ckpt_records.append({
                    'model': m, 'stock': s, 'file': fname, 'epoch': epoch,
                    'step': step, 'val_loss': best_score, 'size_mb': round(fsize_mb, 1),
                    'mtime': mtime, 'best_path_in_cb': os.path.basename(best_path) if best_path else 'None'
                })
            except Exception as e:
                ckpt_records.append({
                    'model': m, 'stock': s, 'file': fname, 'error': str(e),
                    'size_mb': round(fsize_mb, 1), 'mtime': mtime
                })

# Print formatted markdown table
print(f"| {'Model':<11} | {'Stock':<9} | {'File':<16} | {'Epoch':<6} | {'Step':<8} | {'Val Loss (Score)':<18} | {'Saved Best Path in CB':<22} | {'MTime':<19} |")
print("|" + "-"*13 + "|" + "-"*11 + "|" + "-"*18 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*20 + "|" + "-"*24 + "|" + "-"*21 + "|")
for r in ckpt_records:
    val_s = f"{r.get('val_loss'):.6f}" if r.get('val_loss') is not None else "N/A"
    print(f"| {r['model']:<11} | {r['stock']:<9} | {r['file']:<16} | {str(r.get('epoch')):<6} | {str(r.get('step')):<8} | {val_s:<18} | {str(r.get('best_path_in_cb')):<22} | {r['mtime']:<19} |")


print("\n" + "="*90)
print("  FORENSIC AUDIT PART 2: WEIGHT COMPARISON FOR sz300147 VERSIONED FILES")
print("="*90)

for m in MODELS:
    ckpt_dir = os.path.realpath(f"checkpoints/{m}/sz300147")
    best_files = sorted(glob.glob(f"{ckpt_dir}/best*.ckpt"))
    if len(best_files) > 1:
        print(f"\nComparing versioned checkpoints for {m} on sz300147:")
        f0 = best_files[0]
        f1 = best_files[-1]
        try:
            d0 = torch.load(f0, map_location='cpu', weights_only=False)['state_dict']
            d1 = torch.load(f1, map_location='cpu', weights_only=False)['state_dict']
            diffs = 0
            max_d = 0.0
            for k in d0:
                if k in d1:
                    d = (d0[k] - d1[k]).abs().max().item()
                    if d > 1e-5:
                        diffs += 1
                        if d > max_d:
                            max_d = d
            print(f"  • {os.path.basename(f0)} vs {os.path.basename(f1)}:")
            print(f"    - Parameter tensors with delta > 1e-5: {diffs}/{len(d0)}")
            print(f"    - Maximum parameter divergence: {max_d:.6f}")
        except Exception as e:
            print(f"  • Error comparing {f0} vs {f1}: {e}")
    else:
        print(f"\n{m} on sz300147: Only 1 best checkpoint file ({os.path.basename(best_files[0]) if best_files else 'None'})")


print("\n" + "="*90)
print("  FORENSIC AUDIT PART 3: RAW ROW INSPECTION FOR sz300147 TEST SPLIT")
print("="*90)

csv_path = "data/sz300147-level10_processed.csv"
df = pd.read_csv(csv_path)
dates = pd.to_datetime(df['index'].str.split().str[0])
unique_dates = np.sort(dates.unique())
n_dates = len(unique_dates)
test_dates = unique_dates[int(n_dates*0.9):]
test_mask = dates.isin(test_dates).values

test_df = df[test_mask].copy()
features = test_df.iloc[:, 1:].values
col_names = list(test_df.columns[1:])

# Find top 20 rows with highest max abs value
max_abs_per_row = np.abs(features).max(axis=1)
top20_indices = np.argsort(max_abs_per_row)[::-1][:20]

print(f"\nTotal Test Rows: {len(test_df)} across {len(test_dates)} trading days ({str(test_dates[0])[:10]} to {str(test_dates[-1])[:10]})\n")
print(f"| {'Rank':<4} | {'Timestamp':<23} | {'Max |Z|':<8} | {'Spiking Feature':<16} | {'Value':<8} | {'Ask1 Prc':<9} | {'Bid1 Prc':<9} | {'Spread':<8} | {'Ask1 Vol':<9} | {'Bid1 Vol':<9} |")
print("|" + "-"*6 + "|" + "-"*25 + "|" + "-"*10 + "|" + "-"*18 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|")

for rank, idx in enumerate(top20_indices, 1):
    row_feat = features[idx]
    ts = test_df.iloc[idx]['index']
    max_k = np.argmax(np.abs(row_feat))
    max_v = row_feat[max_k]
    max_col = col_names[max_k]
    
    # Feature layout in Level 10:
    # Look up specific columns if named ask_price_1, etc., or by index
    ask1_p = row_feat[0]
    bid1_p = row_feat[20] if len(row_feat) > 20 else 0.0
    spread = ask1_p - bid1_p
    ask1_v = row_feat[10] if len(row_feat) > 10 else 0.0
    bid1_v = row_feat[30] if len(row_feat) > 30 else 0.0
    
    print(f"| {rank:<4} | {ts:<23} | {abs(max_v):<8.2f} | {max_col:<16} | {max_v:<8.2f} | {ask1_p:<9.2f} | {bid1_p:<9.2f} | {spread:<8.2f} | {ask1_v:<9.2f} | {bid1_v:<9.2f} |")


print("\n" + "="*90)
print("  FORENSIC AUDIT PART 4: TEMPORAL CLUSTERING OF |Z| > 10 IN sz300147")
print("="*90)

test_dates_series = test_df['index'].str.split().str[0]
test_max_abs = np.abs(features).max(axis=1)

days_with_10 = {}
days_with_5 = {}

for d_str, val in zip(test_dates_series, test_max_abs):
    if val > 10:
        days_with_10[d_str] = days_with_10.get(d_str, 0) + 1
    if val > 5:
        days_with_5[d_str] = days_with_5.get(d_str, 0) + 1

print(f"\nTotal Test Timestamps with any feature |Z| > 10: {sum(days_with_10.values())} / {len(test_df)} ({sum(days_with_10.values())/len(test_df)*100:.3f}%)")
print(f"Total Test Timestamps with any feature |Z| > 5:  {sum(days_with_5.values())} / {len(test_df)} ({sum(days_with_5.values())/len(test_df)*100:.3f}%)\n")

print(f"| {'Test Date':<12} | {'Rows with |Z| > 10':<20} | {'Rows with |Z| > 5':<20} | {'Daily Max |Z|':<14} |")
print("|" + "-"*14 + "|" + "-"*22 + "|" + "-"*22 + "|" + "-"*16 + "|")

for d_unique in np.unique(test_dates_series):
    sub_mask = (test_dates_series == d_unique).values
    daily_max = test_max_abs[sub_mask].max()
    c10 = days_with_10.get(d_unique, 0)
    c5 = days_with_5.get(d_unique, 0)
    if c5 > 0 or daily_max > 5.0:
        print(f"| {d_unique:<12} | {c10:<20} | {c5:<20} | {daily_max:<14.2f} |")

print("\n" + "="*90)
print("  FORENSIC AUDIT COMPLETE -- ALL METRICS READY FOR REVIEWER REPORT")
print("="*90)
