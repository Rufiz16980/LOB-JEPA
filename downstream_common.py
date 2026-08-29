"""
downstream_common.py
────────────────────
Shared Library for Phase 2 Downstream Probe Probing & Evaluation.

Contains:
  1. Strict Metadata-Based Checkpoint Selector (argmin validation loss)
  2. Verbatim Model Architectures (Encoders & SharedDecoder)
  3. Trend Label Generator (FI-2010 Z-score protocol, train-split thresholding)
  4. TrendHead (Single Linear Layer Probe)
  5. Contiguous Masking & Masked Loss Function for Imputation
  6. Low-Resource Transfer Harness (20% Target Train Split Fine-Tuning)
  7. High-Performance Latent Cacher & Probe Trainers

Author: Phase 2 Downstream Probe Suite
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────────────────────
# 1.  STRICT METADATA-BASED CHECKPOINT SELECTOR (ARGMIN VAL LOSS)
# ─────────────────────────────────────────────────────────────────────────────

def select_optimal_checkpoint(ckpt_dir: str, metric_name: str = 'val_mse') -> str:
    """Scan every best*.ckpt in ckpt_dir and return the path with the true lowest
    recorded validation score. Never infer this from filename, version suffix,
    or modification time -- read the actual stored metric from each file."""
    candidates = sorted(glob.glob(f"{ckpt_dir}/best*.ckpt"))
    if not candidates:
        candidates = sorted(glob.glob(f"{ckpt_dir}/last*.ckpt"))
    assert candidates, f"No checkpoint files found in {ckpt_dir}"

    scored = []
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', 0)
            score = float('inf')
            callbacks = ckpt.get('callbacks', {})
            for cb_key, cb_val in callbacks.items():
                if 'ModelCheckpoint' in cb_key and cb_val.get('best_model_score') is not None:
                    score = float(cb_val['best_model_score'])
                    break
            scored.append((path, float(score), epoch))
        except Exception as e:
            scored.append((path, float('inf'), -1))

    scored.sort(key=lambda x: (x[1], -x[2] if x[2] is not None else 0))  # lowest score wins, higher epoch breaks ties
    best_path, best_score, best_epoch = scored[0]

    print(f"  ✓ Selected {os.path.basename(best_path)} (epoch {best_epoch}, {metric_name}={best_score:.6f}) from {len(candidates)} candidate(s) in {ckpt_dir}")
    for path, score, epoch in scored[1:]:
        print(f"    (rejected: {os.path.basename(path)}, epoch {epoch}, {metric_name}={score:.6f})")

    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXACT VERBATIM ENCODER & DECODER ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, n_features=40, hidden_size=128, num_layers=3, latent_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.proj(last)


class CNNEncoder(nn.Module):
    def __init__(self, n_features=40, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


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

    def forward(self, x):
        x = x.unsqueeze(1)
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

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.proj(pooled)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.pad, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TransLOBEncoder(nn.Module):
    def __init__(self, n_features=40, conv_channels=32, d_model=64, nhead=4,
                 num_transformer_layers=2, latent_dim=256, seq_len=100):
        super().__init__()
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

    def forward(self, x):
        h = x.transpose(1, 2)
        for conv in self.conv_layers:
            h = F.relu(conv(h))
        h = h.transpose(1, 2)
        h = self.conv_proj(h) + self.pos_embedding
        seq_len_dim = h.shape[1]
        mask = torch.triu(torch.full((seq_len_dim, seq_len_dim), float('-inf')), diagonal=1).to(h.device)
        h = self.transformer(h, mask=mask)
        last = h[:, -1, :]
        return self.proj(last)


class SimLOBEncoder(nn.Module):
    def __init__(self, n_features=40, d_model=256, nhead=8, num_layers=2,
                 dim_feedforward=512, latent_dim=256, seq_len=100):
        super().__init__()
        self.fcn1 = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reduce_proj = nn.Linear(d_model, n_features)
        self.fcn2 = nn.Sequential(
            nn.Linear(seq_len * n_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        h = self.fcn1(x)
        h = self.transformer(h)
        h = self.reduce_proj(h)
        h = h.reshape(h.shape[0], -1)
        return self.fcn2(h)


class SharedDecoder(nn.Module):
    def __init__(self, latent_dim=256, n_features=40, seq_len=100):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, seq_len * n_features),
        )

    def forward(self, z):
        h = self.net(z)
        return h.view(z.shape[0], self.seq_len, self.n_features)


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
SEQ_LEN = 100


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA UTILITIES & SESSION DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_sessions(df: pd.DataFrame, max_gap_seconds=300):
    timestamps = pd.to_datetime(df['index'])
    diffs = timestamps.diff().dt.total_seconds().fillna(0)
    session_ids = (diffs > max_gap_seconds).cumsum().values
    return session_ids


def split_by_date(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    dates = pd.to_datetime(df['index'].str.split().str[0])
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    train_cutoff = unique_dates[int(n_dates * train_ratio)]
    val_cutoff = unique_dates[int(n_dates * (train_ratio + val_ratio))]

    train_mask = dates < train_cutoff
    val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
    test_mask = dates >= val_cutoff
    return train_mask.values, val_mask.values, test_mask.values


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TREND LABEL GENERATION (FI-2010 PROTOCOL, TRAIN-ONLY THRESHOLD)
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_labels_and_windows(df: pd.DataFrame, k=5, seq_len=100):
    """
    Computes 3-class trend labels:
      mid(t) = (BidPrice1(t) + AskPrice1(t)) / 2
      m_past(t) = mean(mid(t-k+1) ... mid(t))
      m_future(t) = mean(mid(t+1) ... mid(t+k))
      l(t) = m_future(t) - m_past(t)  (raw difference)
      threshold θ: 33rd/67th percentile computed on TRAIN split only.
    """
    train_mask, val_mask, test_mask = split_by_date(df)
    session_ids = detect_sessions(df)
    
    # Compute mid price
    mid = ((df['BidPrice1'] + df['AskPrice1']) / 2.0).values
    N = len(df)
    
    # Rolling averages: past k and future k
    # m_past(t): average of mid[t-k+1 : t+1]
    # m_future(t): average of mid[t+1 : t+k+1]
    m_past = pd.Series(mid).rolling(window=k).mean().values
    # Future rolling mean by reversing, rolling, and reversing back
    m_future = pd.Series(mid[::-1]).rolling(window=k).mean().values[::-1]
    # Shift m_future so m_future[t] is mean of mid[t+1 : t+k+1]
    m_future_shifted = np.full_like(mid, np.nan)
    if N > k:
        m_future_shifted[:-k] = m_future[k:]
    
    raw_signal = m_future_shifted - m_past  # l(t) where t is window end
    
    # Identify valid window start positions i (where input window is [i : i+seq_len])
    # The end of the window is t = i + seq_len - 1
    # Validity requires:
    #   1. session_ids[i] == session_ids[i + seq_len - 1] (entire window in same session)
    #   2. session_ids[i + seq_len - 1] == session_ids[i + seq_len - 1 + k] (future k ticks in same session)
    #   3. i + seq_len - 1 + k < N
    
    split_indices = {'train': [], 'val': [], 'test': []}
    split_signals = {'train': [], 'val': [], 'test': []}
    
    for i in range(N - seq_len - k):
        t_end = i + seq_len - 1
        t_fut = t_end + k
        
        # Check session continuity across the full window + prediction horizon
        if session_ids[i] == session_ids[t_fut]:
            sig = raw_signal[t_end]
            if not np.isnan(sig):
                if train_mask[i]:
                    split_indices['train'].append(i)
                    split_signals['train'].append(sig)
                elif val_mask[i]:
                    split_indices['val'].append(i)
                    split_signals['val'].append(sig)
                elif test_mask[i]:
                    split_indices['test'].append(i)
                    split_signals['test'].append(sig)

    train_signals = np.array(split_signals['train'])
    assert len(train_signals) > 0, "No valid training windows found"

    # Compute threshold θ from TRAIN SPLIT ONLY (33rd / 67th percentile)
    # Class 0: Down (l < -θ), Class 1: Stable (-θ <= l <= θ), Class 2: Up (l > θ)
    # Using symmetric threshold from 67th percentile of absolute signal or percentile boundaries
    p33 = np.percentile(train_signals, 33.33)
    p67 = np.percentile(train_signals, 66.67)
    theta = (abs(p33) + abs(p67)) / 2.0
    
    # Assign labels
    split_labels = {}
    for split in ['train', 'val', 'test']:
        sigs = np.array(split_signals[split])
        labs = np.ones(len(sigs), dtype=np.int64)  # default 1 (stable)
        labs[sigs < -theta] = 0                    # 0: Down
        labs[sigs > theta] = 2                     # 2: Up
        split_labels[split] = labs

    return split_indices, split_labels, theta


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PROBE ARCHITECTURES & LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TrendHead(nn.Module):
    """Single linear layer probing head testing representation quality directly."""
    def __init__(self, latent_dim=256, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


class LatentDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


# --- Imputation Protocol (Contiguous 20-step masking) ---

def imputation_loss(output, target, mask):
    """Loss computed STRICTLY on masked positions only."""
    diff = (output - target) ** 2
    # mask: [B, 100] boolean, True = masked position
    masked_diff = diff[mask.unsqueeze(-1).expand_as(diff)]
    return masked_diff.mean()


def imputation_mae(output, target, mask):
    """MAE computed STRICTLY on masked positions only."""
    diff = (output - target).abs()
    masked_diff = diff[mask.unsqueeze(-1).expand_as(diff)]
    return masked_diff.mean()


class ContiguousMaskingDataset(Dataset):
    """Wraps raw LOB window starts; dynamically applies a single contiguous 20-step mask per sample."""
    def __init__(self, features, start_indices, seq_len=100, mask_len=20):
        self.features = features
        self.start_indices = start_indices
        self.seq_len = seq_len
        self.mask_len = mask_len

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        x_orig = torch.tensor(self.features[start : start + self.seq_len], dtype=torch.float32)
        
        # Dynamically generate random mask start in [0, seq_len - mask_len]
        mask_start = np.random.randint(0, self.seq_len - self.mask_len + 1)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[mask_start : mask_start + self.mask_len] = True
        
        # Zero out masked timesteps in input
        x_masked = x_orig.clone()
        x_masked[mask] = 0.0
        
        return x_masked, x_orig, mask


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HIGH-PERFORMANCE LATENT CACHER
# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_encoder(model_name: str, stock: str, device='cuda') -> nn.Module:
    ckpt_dir = f"checkpoints/{model_name}/{stock}"
    best_path = select_optimal_checkpoint(ckpt_dir)
    
    encoder_cls = MODEL_REGISTRY[model_name]
    encoder = encoder_cls(latent_dim=LATENT_DIM)
    
    # Load state dict
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    
    # Strip encoder. prefix if present from Lightning module
    encoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            encoder_dict[k[len('encoder.'):]] = v
        elif not k.startswith('decoder.'):
            encoder_dict[k] = v
            
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def precompute_and_cache_latents(model_name: str, stock: str, out_dir: str = "latents", device='cuda'):
    """Precomputes and caches [N, 256] latents for train, val, and test splits."""
    os.makedirs(f"{out_dir}/{model_name}/{stock}", exist_ok=True)
    
    csv_path = f"data/{stock}-level10_processed.csv"
    df = pd.read_csv(csv_path)
    split_indices, split_labels, theta = compute_trend_labels_and_windows(df, k=5, seq_len=SEQ_LEN)
    features = df.iloc[:, 1:].values
    
    encoder = load_frozen_encoder(model_name, stock, device=device)
    
    cached_data = {}
    for split in ['train', 'val', 'test']:
        starts = split_indices[split]
        labels = split_labels[split]
        
        # Batch forward pass over windows
        latents_list = []
        batch_size = 512
        for b_idx in range(0, len(starts), batch_size):
            b_starts = starts[b_idx : b_idx + batch_size]
            b_windows = np.stack([features[s : s + SEQ_LEN] for s in b_starts])
            b_tensor = torch.tensor(b_windows, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                z = encoder(b_tensor)
                latents_list.append(z.cpu().numpy())
                
        all_z = np.concatenate(latents_list, axis=0) if latents_list else np.empty((0, LATENT_DIM))
        
        np.save(f"{out_dir}/{model_name}/{stock}/{split}_latents.npy", all_z)
        np.save(f"{out_dir}/{model_name}/{stock}/{split}_labels.npy", labels)
        cached_data[split] = (all_z, labels)
        
    np.save(f"{out_dir}/{model_name}/{stock}/theta.npy", np.array([theta]))
    print(f"  ✓ Cached {model_name}/{stock}: Train {cached_data['train'][0].shape}, Val {cached_data['val'][0].shape}, Test {cached_data['test'][0].shape} (θ={theta:.6f})")
    return cached_data


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PROBE TRAINER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_trend_head_probe(train_z, train_y, val_z, val_y, test_z, test_y, epochs=50, lr=1e-3, batch_size=256, device='cuda'):
    """Trains a TrendHead (Linear probe) and evaluates test Accuracy, Macro-F1, Precision, Recall."""
    train_loader = DataLoader(LatentDataset(train_z, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LatentDataset(val_z, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(LatentDataset(test_z, test_y), batch_size=batch_size, shuffle=False)
    
    head = TrendHead(latent_dim=LATENT_DIM, n_classes=3).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = -1.0
    best_weights = None
    
    for epoch in range(epochs):
        head.train()
        for z, y in train_loader:
            z, y = z.to(device), y.to(device)
            optimizer.zero_grad()
            out = head(z)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        # Validate
        head.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for z, y in val_loader:
                z, y = z.to(device), y.to(device)
                preds = head(z).argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            
    # Load best validation head and evaluate on test
    head.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    head.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for z, y in test_loader:
            z, y = z.to(device), y.to(device)
            preds = head(z).argmax(dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
            
    acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(test_targets, test_preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'precision_down': prec[0], 'recall_down': rec[0],
        'precision_stable': prec[1], 'recall_stable': rec[1],
        'precision_up': prec[2], 'recall_up': rec[2],
    }


def train_imputation_probe(model_name: str, stock: str, epochs=50, lr=1e-3, batch_size=256, device='cuda'):
    """Trains fresh SharedDecoder on 20-step contiguous masked inputs with frozen encoder."""
    csv_path = f"data/{stock}-level10_processed.csv"
    df = pd.read_csv(csv_path)
    train_mask, val_mask, test_mask = split_by_date(df)
    session_ids = detect_sessions(df)
    features = df.iloc[:, 1:].values
    
    from common import get_window_indices, LOBDataset
    train_starts = get_window_indices(df, train_mask, session_ids, seq_len=SEQ_LEN)
    val_starts = get_window_indices(df, val_mask, session_ids, seq_len=SEQ_LEN)
    test_starts = get_window_indices(df, test_mask, session_ids, seq_len=SEQ_LEN)
    
    train_ds = ContiguousMaskingDataset(features, train_starts, seq_len=SEQ_LEN, mask_len=20)
    val_ds = ContiguousMaskingDataset(features, val_starts, seq_len=SEQ_LEN, mask_len=20)
    test_ds = ContiguousMaskingDataset(features, test_starts, seq_len=SEQ_LEN, mask_len=20)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    encoder = load_frozen_encoder(model_name, stock, device=device)
    decoder = SharedDecoder(latent_dim=LATENT_DIM, n_features=40, seq_len=SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        decoder.train()
        for x_masked, x_orig, mask in train_loader:
            x_masked, x_orig, mask = x_masked.to(device), x_orig.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z = encoder(x_masked)
            x_hat = decoder(z)
            loss = imputation_loss(x_hat, x_orig, mask)
            loss.backward()
            optimizer.step()
            
        # Validate
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for x_masked, x_orig, mask in val_loader:
                x_masked, x_orig, mask = x_masked.to(device), x_orig.to(device), mask.to(device)
                z = encoder(x_masked)
                x_hat = decoder(z)
                v_loss = imputation_loss(x_hat, x_orig, mask)
                val_losses.append(v_loss.item())
        mean_val = np.mean(val_losses)
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_weights = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            
    # Test Evaluation
    decoder.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    decoder.eval()
    test_mses, test_maes = [], []
    with torch.no_grad():
        for x_masked, x_orig, mask in test_loader:
            x_masked, x_orig, mask = x_masked.to(device), x_orig.to(device), mask.to(device)
            z = encoder(x_masked)
            x_hat = decoder(z)
            test_mses.append(imputation_loss(x_hat, x_orig, mask).item())
            test_maes.append(imputation_mae(x_hat, x_orig, mask).item())
            
    return {
        'masked_test_mse': float(np.mean(test_mses)),
        'masked_test_mae': float(np.mean(test_maes)),
    }
