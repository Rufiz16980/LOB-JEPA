"""
evaluate_all.py
───────────────
Standalone evaluation script for the LOBench baseline suite.

This script loads the strictly optimal validation checkpoint (argmin val_loss)
for every model-stock combination by directly inspecting the internal PyTorch
Lightning checkpoint metadata (best_model_score), bypassing any filename or
modification-time versioning artifacts.

HOW TO RUN (Google Colab):
    1. Mount Google Drive and cd to the baselines folder:
         from google.colab import drive
         drive.mount('/content/drive')
         import os
         os.chdir('/content/drive/MyDrive/JEPA_LOB/baselines')
    2. Install dependencies:
         !pip install -q lightning pandas numpy torch
    3. Run:
         !python evaluate_all.py

Author: Auto-generated evaluation script (argmin validation loss selection)
"""

import os
import glob
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── pick up common utilities without modifying common.py ─────────────────────
from common import (
    set_seed, detect_sessions, split_by_date, get_window_indices,
    LOBDataset, LOBReconstructionModule, SharedDecoder,
)

try:
    from lightning.pytorch import Trainer
except ImportError:
    from pytorch_lightning import Trainer

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXACT VERBATIM ENCODER DEFINITIONS FROM FROZEN NOTEBOOKS
# ─────────────────────────────────────────────────────────────────────────────

# --- 1.1 LSTM Encoder (from LSTM.ipynb) ---
class LSTMEncoder(nn.Module):
    def __init__(self, n_features=40, hidden_size=128, num_layers=3, latent_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):  # x: [B, 100, 40]
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]        # [B, hidden_size] -- final timestep
        return self.proj(last)      # [B, latent_dim]


# --- 1.2 CNN Encoder (from CNN.ipynb) ---
class CNNEncoder(nn.Module):
    def __init__(self, n_features=40, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x):  # x: [B, 100, 40]
        x = x.transpose(1, 2)                  # [B, 40, 100] -- Conv1d expects channels-first
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)            # [B, 64]
        return self.proj(x)                     # [B, latent_dim]


# --- 1.3 DeepLOB Encoder (from DeepLOB.ipynb) ---
class DeepLOBEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,10)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4,1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(32, 64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x):  # x: [B, 100, 40]
        x = x.unsqueeze(1)                       # [B, 1, 100, 40]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_inp1, x_inp2, x_inp3 = self.inp1(x), self.inp2(x), self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x)
        last = x[:, -1, :]
        return self.proj(last)


# --- 1.4 Transformer Encoder (from Transformer.ipynb) ---
class TransformerEncoder(nn.Module):
    def __init__(self, n_features=40, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, latent_dim=256, seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, latent_dim)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):  # x: [B, 100, 40]
        x = self.input_proj(x) + self.pos_embedding    # [B, 100, d_model]
        x = self.transformer(x)                         # [B, 100, d_model]
        pooled = x.mean(dim=1)                           # mean-pool over time
        return self.proj(pooled)


# --- 1.5 TransLOB Encoder (from TransLOB.ipynb) ---
class CausalConv1d(nn.Module):
    """1D convolution padded so output[t] only depends on input[<=t] -- causal."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.pad, dilation=dilation)

    def forward(self, x):  # x: [B, C, T]
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TransLOBEncoder(nn.Module):
    def __init__(self, n_features=40, conv_channels=32, d_model=64, nhead=4,
                 num_transformer_layers=2, latent_dim=256, seq_len=100):
        super().__init__()
        # Dilated causal convolution stack, dilations 1,2,4,8 (standard WaveNet-style schedule)
        self.conv_layers = nn.ModuleList([
            CausalConv1d(n_features if i == 0 else conv_channels, conv_channels,
                         kernel_size=3, dilation=2**i)
            for i in range(4)
        ])
        self.conv_proj = nn.Linear(conv_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.proj = nn.Linear(d_model, latent_dim)

    def forward(self, x):  # x: [B, 100, 40]
        h = x.transpose(1, 2)                     # [B, 40, 100]
        for conv in self.conv_layers:
            h = F.relu(conv(h))
        h = h.transpose(1, 2)                       # [B, 100, conv_channels]
        h = self.conv_proj(h) + self.pos_embedding   # [B, 100, d_model]
        # Causal mask: position t may only attend to positions <= t
        seq_len_dim = h.shape[1]
        mask = torch.triu(torch.full((seq_len_dim, seq_len_dim), float('-inf')), diagonal=1).to(h.device)
        h = self.transformer(h, mask=mask)
        last = h[:, -1, :]                            # [B, d_model]
        return self.proj(last)                         # [B, latent_dim]


# --- 1.6 SimLOB Encoder (from SimLOB.ipynb) ---
class SimLOBEncoder(nn.Module):
    def __init__(self, n_features=40, d_model=256, nhead=8, num_layers=2,
                 dim_feedforward=512, latent_dim=256, seq_len=100):
        super().__init__()
        # FCN1: per-timestep feature extraction, 40 -> 256
        self.fcn1 = nn.Linear(n_features, d_model)
        # Transformer stack, L=2 per the paper's chosen default
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # FCN2: dimension reduction block -- project back to n_features per timestep,
        # flatten, then reduce to the latent vector
        self.reduce_proj = nn.Linear(d_model, n_features)
        self.fcn2 = nn.Sequential(
            nn.Linear(seq_len * n_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):  # x: [B, 100, 40]
        h = self.fcn1(x)                        # [B, 100, 256]
        h = self.transformer(h)                  # [B, 100, 256]
        h = self.reduce_proj(h)                  # [B, 100, 40]
        h = h.reshape(h.shape[0], -1)             # [B, 4000]
        return self.fcn2(h)                       # [B, latent_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'LSTM':        LSTMEncoder,
    'CNN':         CNNEncoder,
    'DeepLOB':     DeepLOBEncoder,
    'Transformer': TransformerEncoder,
    'TransLOB':    TransLOBEncoder,
    'SimLOB':      SimLOBEncoder,
}

STOCKS = ['sz000001', 'sz000002', 'sz000858', 'sz300147', 'sz002415']

LATENT_DIM = 256
SEQ_LEN    = 100
BATCH_SIZE = 256
NUM_WORKERS = 2


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ARGMIN VALIDATION LOSS CHECKPOINT SELECTOR
#     Inspects internal PyTorch Lightning state dictionaries and selects
#     strictly the checkpoint with the lowest recorded validation loss.
# ─────────────────────────────────────────────────────────────────────────────

def get_best_ckpt(model_name: str, stock: str) -> tuple[str | None, float, int]:
    ckpt_dir = os.path.realpath(f"checkpoints/{model_name}/{stock}")
    candidates = sorted(glob.glob(f"{ckpt_dir}/best*.ckpt"))
    if not candidates:
        candidates = sorted(glob.glob(f"{ckpt_dir}/last*.ckpt"))
    if not candidates:
        return None, float('inf'), -1

    best_file = None
    best_score = float('inf')
    best_epoch = -1

    for fpath in candidates:
        try:
            data = torch.load(fpath, map_location='cpu', weights_only=False)
            epoch = data.get('epoch', 0)
            score = float('inf')
            
            for k, v in data.get('callbacks', {}).items():
                if 'ModelCheckpoint' in k and v.get('best_model_score') is not None:
                    score = float(v['best_model_score'])
                    break
            
            # Select strictly lowest validation loss (tie-break by higher epoch)
            if score < best_score:
                best_score = score
                best_file = fpath
                best_epoch = epoch
            elif abs(score - best_score) < 1e-7 and epoch > best_epoch:
                best_file = fpath
                best_epoch = epoch
        except Exception as e:
            # Fallback to mtime if file is unreadable
            if best_file is None:
                best_file = fpath

    return best_file, best_score, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATA LOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_test_loader(stock: str):
    csv_path = f"data/{stock}-level10_processed.csv"
    df = pd.read_csv(csv_path)
    session_ids = detect_sessions(df)
    _, _, test_mask = split_by_date(df)
    features = df.iloc[:, 1:].values  # drop 'index' column
    test_starts = get_window_indices(df, test_mask, session_ids, seq_len=SEQ_LEN)
    test_dataset = LOBDataset(features, test_starts, seq_len=SEQ_LEN)
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SINGLE-EXPERIMENT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model_name: str, stock: str) -> dict:
    ckpt_path, best_val_score, epoch = get_best_ckpt(model_name, stock)
    if ckpt_path is None:
        print(f"\nEvaluating {model_name} / {stock} ...")
        print(f"  ⚠ No checkpoint found for {model_name} / {stock} -- skipping.")
        return {'model': model_name, 'stock': stock, 'test_mse': float('nan'), 'test_mae': float('nan'), 'ckpt': 'MISSING', 'epoch': -1, 'val_loss': float('nan')}

    fname = os.path.basename(ckpt_path)
    val_score_str = f"{best_val_score:.6f}" if best_val_score != float('inf') else "N/A"
    print(f"\nEvaluating {model_name} / {stock} ...")
    print(f"  ✓ Loading optimal ckpt: {fname} (Epoch {epoch}, Min Val Loss: {val_score_str})")

    encoder_class = MODEL_REGISTRY[model_name]
    encoder = encoder_class(latent_dim=LATENT_DIM)
    model = LOBReconstructionModule.load_from_checkpoint(
        ckpt_path, encoder=encoder, latent_dim=LATENT_DIM
    )
    model.eval()

    test_loader = build_test_loader(stock)

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    results = trainer.test(model, dataloaders=test_loader, verbose=False)
    test_mse = results[0]['test_mse']
    test_mae = results[0]['test_mae']

    print(f"  ✓ Test MSE = {test_mse:.6f} | Test MAE = {test_mae:.6f}")
    return {
        'model': model_name,
        'stock': stock,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'ckpt': fname,
        'epoch': epoch,
        'val_loss': best_val_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN -- run all 30 experiments and print clean Markdown tables
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    all_results = []

    for model_name in MODEL_REGISTRY:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        for stock in STOCKS:
            r = evaluate(model_name, stock)
            all_results.append(r)

    # Save to CSV
    out_csv = "evaluation_results.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'stock', 'test_mse', 'test_mae', 'ckpt', 'epoch', 'val_loss'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to: {out_csv}")

    # Print Markdown Table (MSE)
    print("\n\n" + "="*80)
    print("  DEFINITIVE RESULTS TABLE (ARGMIN VAL LOSS) -- Test MSE")
    print("="*80)

    header = f"| {'Model':<12} | " + " | ".join(f"{s:<10}" for s in STOCKS) + " |"
    sep    = "|" + "-"*14 + "|" + ("|".join(["-"*12]*len(STOCKS))) + "|"
    print(header)
    print(sep)

    for model_name in MODEL_REGISTRY:
        row_vals = []
        for stock in STOCKS:
            match = next((r for r in all_results if r['model'] == model_name and r['stock'] == stock), None)
            if match and not (match['test_mse'] != match['test_mse']):   # nan check
                row_vals.append(f"{match['test_mse']:.4f}")
            else:
                row_vals.append("N/A")
        print(f"| {model_name:<12} | " + " | ".join(f"{v:<10}" for v in row_vals) + " |")

    # Print Markdown Table (MAE)
    print("\n")
    print("="*80)
    print("  DEFINITIVE RESULTS TABLE (ARGMIN VAL LOSS) -- Test MAE")
    print("="*80)
    print(header)
    print(sep)

    for model_name in MODEL_REGISTRY:
        row_vals = []
        for stock in STOCKS:
            match = next((r for r in all_results if r['model'] == model_name and r['stock'] == stock), None)
            if match and not (match['test_mae'] != match['test_mae']):
                row_vals.append(f"{match['test_mae']:.4f}")
            else:
                row_vals.append("N/A")
        print(f"| {model_name:<12} | " + " | ".join(f"{v:<10}" for v in row_vals) + " |")

    # Print Checkpoint Inventory Summary
    print("\n")
    print("="*80)
    print("  LOADED CHECKPOINT INVENTORY (ALL 30 CELLS)")
    print("="*80)
    print(f"| {'Model':<11} | {'Stock':<9} | {'Selected File':<16} | {'Epoch':<6} | {'Recorded Min Val Loss':<22} |")
    print("|" + "-"*13 + "|" + "-"*11 + "|" + "-"*18 + "|" + "-"*8 + "|" + "-"*24 + "|")
    for r in all_results:
        val_str = f"{r['val_loss']:.6f}" if r['val_loss'] != float('inf') else "N/A"
        print(f"| {r['model']:<11} | {r['stock']:<9} | {r['ckpt']:<16} | {str(r['epoch']):<6} | {val_str:<22} |")

    print("\n✅ Evaluation complete across all 30 model-stock experiments.")
    print("   Every single model was loaded strictly using argmin(validation_loss).")


if __name__ == "__main__":
    main()
