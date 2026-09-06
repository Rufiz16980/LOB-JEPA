"""
jepa_common.py
──────────────
Core module for Phase 3: JEPA Model Design, Training & Ablation.

Strict Compliance with Phase 3 Specification (phase3-jepa-training.md):
  1. JEPABackbone: exact extraction of SimLOB's FCN1 + 2-layer TransformerEncoder (no FCN2/reduce_proj).
  2. Fixed-length sequence masking (seq_len=100) using learnable mask_token (trunc_normal std=0.02).
  3. Multi-block temporal masking (Phase 3a: 4 blocks, length 10-20, overlap rejection).
  4. Spatio-temporal masking (Phase 3b: temporal masking + 30% of visible timesteps zero one field group).
  5. Predictor: 4-layer Transformer with learnable positional embedding and output projection.
  6. Masked-only prediction loss + VICReg variance & covariance collapse-prevention loss (lambda=1.0).
  7. Momentum-based target encoder EMA update (0.996 -> 1.0 linear annealing).
  8. JEPAEncoderForEval adapter: temporal mean-pooling to [B, 256] for Phase 2 harness compatibility.
  9. Joint 5-stock pooled dataset preparation and PyTorch Lightning training module.

Phase 1 & Phase 2 files are strictly read-only and never modified.
"""

import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# Lightning imports with environment fallback
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch import Trainer
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning import Trainer

# Read-only imports from Phase 1 & Phase 2
from common import split_by_date, detect_sessions, get_window_indices, set_seed
from downstream_common import (
    select_optimal_checkpoint,
    compute_trend_labels_and_windows,
    train_trend_head_probe,
    ContiguousMaskingDataset,
    SharedDecoder,
    imputation_loss,
    imputation_mae,
    LatentDataset,
    TrendHead,
    seed_worker,
    LATENT_DIM,
    SEQ_LEN,
)

ALL_STOCKS = ['sz000001', 'sz000002', 'sz000858', 'sz300147', 'sz002415']
LAMBDA_COLLAPSE = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BACKBONE ARCHITECTURE (SECTION 1)
# ─────────────────────────────────────────────────────────────────────────────

class JEPABackbone(nn.Module):
    """
    Reused verbatim from the Phase 1 SimLOBEncoder (Section 3.5 of the baseline master doc),
    with reduce_proj / fcn2 removed. Output is per-timestep [B, 100, 256], not pooled.
    Total parameter count: exactly 1,064,704.
    """
    def __init__(self, n_features=40, d_model=256, nhead=8, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.fcn1 = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, 100, 40] (already masked, if this is the context encoder)
        h = self.fcn1(x)          # [B, 100, 256]
        h = self.transformer(h)   # [B, 100, 256]
        return h


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MASKING MECHANISM & TOKENS (SECTION 2.2 & SECTION 6)
# ─────────────────────────────────────────────────────────────────────────────

def init_mask_token(n_features=40):
    mask_token = nn.Parameter(torch.zeros(1, 1, n_features))
    nn.init.trunc_normal_(mask_token, std=0.02)
    return mask_token


def apply_mask(x, mask, mask_token):
    """
    x: [B, 100, 40]
    mask: [B, 100] bool, True = masked
    mask_token: [1, 1, 40] Parameter
    Replaces all masked timesteps with the learnable mask token.
    """
    x_masked = x.clone()
    x_masked[mask] = mask_token.squeeze(0).squeeze(0)  # broadcast [40] to all [M, 40] masked positions
    return x_masked


def temporal_mask(batch_size, seq_len=100, n_blocks=4, block_size_range=(10, 20), device=None):
    """
    Returns [B, 100] bool tensor, True = masked. Each sample gets n_blocks
    non-overlapping temporal blocks masked, block lengths drawn uniformly from block_size_range.
    """
    masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for b in range(batch_size):
        occupied = torch.zeros(seq_len, dtype=torch.bool, device=device)
        blocks_placed = 0
        attempts = 0
        while blocks_placed < n_blocks and attempts < 50:
            attempts += 1
            block_len = torch.randint(block_size_range[0], block_size_range[1] + 1, (1,)).item()
            start = torch.randint(0, seq_len - block_len + 1, (1,)).item()
            span = slice(start, start + block_len)
            if occupied[span].any():
                continue   # overlaps an existing block, retry
            occupied[span] = True
            blocks_placed += 1
        masks[b] = occupied
    return masks


FIELD_GROUPS = {
    'BidPrice':  slice(0, 10),
    'AskPrice':  slice(10, 20),
    'BidVolume': slice(20, 30),
    'AskVolume': slice(30, 40),
}


def spatiotemporal_mask(x, batch_size, seq_len=100, n_blocks=4, block_size_range=(10, 20),
                        field_mask_prob=0.3):
    """
    Returns (temporal_masked, x_out).
    temporal_masked is identical to Section 6.1's output [B, 100] bool.
    x_out additionally zeroes out one randomly chosen field group, for a
    field_mask_prob fraction of the temporally-VISIBLE positions.
    """
    device = x.device
    temporal_masked = temporal_mask(batch_size, seq_len, n_blocks, block_size_range, device=device)
    x_out = x.clone()
    field_names = list(FIELD_GROUPS.keys())
    for b in range(batch_size):
        visible_positions = (~temporal_masked[b]).nonzero(as_tuple=True)[0]
        n_to_corrupt = int(len(visible_positions) * field_mask_prob)
        if n_to_corrupt == 0:
            continue
        chosen_positions = visible_positions[torch.randperm(len(visible_positions))[:n_to_corrupt]]
        for pos in chosen_positions:
            field = field_names[torch.randint(0, len(field_names), (1,)).item()]
            x_out[b, pos, FIELD_GROUPS[field]] = 0.0   # zero out that field group at that timestep
    return temporal_masked, x_out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREDICTOR ARCHITECTURE (SECTION 2.3)
# ─────────────────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Predictor Transformer: maps context representations + positional info
    to predictions of target encoder representations.
    4 layers, d_model=256, nhead=8, dim_feedforward=512, seq_len=100.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, seq_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, context_output):  # [B, 100, 256]
        h = context_output + self.pos_embedding
        h = self.transformer(h)
        return self.output_proj(h)      # [B, 100, 256]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOSS FUNCTIONS & REGULARIZATION (SECTION 3)
# ─────────────────────────────────────────────────────────────────────────────

def prediction_loss(pred, target, mask):
    """
    Loss computed STRICTLY at masked positions only.
    pred: [B, 100, 256]
    target: [B, 100, 256]
    mask: [B, 100] bool, True = masked
    """
    diff = (pred - target) ** 2
    masked_diff = diff[mask.unsqueeze(-1).expand_as(diff)]
    return masked_diff.mean()


def collapse_prevention_loss(z, gamma=1.0, eps=1e-4):
    """
    VICReg variance + covariance penalty (Bardes, Ponce & LeCun, 2022).
    z: [B, 100, 256] -- context encoder output, flattened to [B*100, 256].
    Variance term: penalize any embedding dimension whose std falls below gamma.
    Covariance term: decorrelate different embedding dimensions from each other.
    """
    z = z.reshape(-1, z.shape[-1])              # [B*100, 256]
    z = z - z.mean(dim=0)

    std_z = torch.sqrt(z.var(dim=0) + eps)
    loss_var = torch.mean(F.relu(gamma - std_z))

    N, D = z.shape
    cov_z = (z.T @ z) / (N - 1)                  # [256, 256]
    off_diagonal = cov_z.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()  # off-diag elements
    loss_cov = off_diagonal.pow(2).sum() / D

    return loss_var + loss_cov  # equal weighting between the two sub-terms


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TARGET ENCODER EMA UPDATE (SECTION 4)
# ─────────────────────────────────────────────────────────────────────────────

def update_target_encoder(context_encoder, target_encoder, momentum):
    """
    In-place momentum update of target encoder weights:
    p_target = momentum * p_target + (1 - momentum) * p_context
    """
    with torch.no_grad():
        for p_context, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)


def get_momentum(step, total_steps, start=0.996, end=1.0):
    """Linear annealing of EMA momentum from 0.996 to 1.0."""
    if total_steps <= 0:
        return end
    progress = min(max(step / total_steps, 0.0), 1.0)
    return start + (end - start) * progress


# ─────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATION ADAPTER (SECTION 7.2)
# ─────────────────────────────────────────────────────────────────────────────

class JEPAEncoderForEval(nn.Module):
    """
    Wraps a trained JEPA context encoder so it presents the same [B, 256] output
    interface Phase 2's harness expects from every Phase 1 baseline encoder.
    Does not modify downstream_common.py -- this wrapper lives in the Phase 3 module.
    """
    def __init__(self, jepa_backbone):
        super().__init__()
        self.backbone = jepa_backbone

    def forward(self, x):  # x: [B, 100, 40], UNMASKED -- evaluation always uses the full window
        h = self.backbone(x)          # [B, 100, 256]
        return h.mean(dim=1)           # [B, 256] -- matches every Phase 1/2 encoder's output shape


# ─────────────────────────────────────────────────────────────────────────────
# 7.  POOLED 5-STOCK DATA PIPELINE (SECTION 5)
# ─────────────────────────────────────────────────────────────────────────────

class PooledLOBDataset(Dataset):
    """
    Memory-efficient dataset that pools sequences across all 5 stocks.
    Holds raw feature arrays in memory (~935MB total across 5 stocks) and references
    windows via (stock_idx, start_idx) pairs to prevent OOM.
    """
    def __init__(self, stock_features_list, sample_index_pairs, seq_len=100):
        self.stock_features = [torch.tensor(f, dtype=torch.float32) for f in stock_features_list]
        self.sample_index_pairs = sample_index_pairs  # np.ndarray of shape [N, 2]: (stock_idx, start_idx)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sample_index_pairs)

    def __getitem__(self, idx):
        stock_idx, start_idx = self.sample_index_pairs[idx]
        feat = self.stock_features[stock_idx]
        x = feat[start_idx : start_idx + self.seq_len]
        return x


def prepare_pooled_datasets(stocks=ALL_STOCKS, data_dir='data', seq_len=100):
    """
    Loads all stocks, performs Phase 1 split_by_date and detect_sessions,
    and returns PooledLOBDataset instances for train and validation splits.
    """
    stock_features = []
    train_pairs = []
    val_pairs = []
    
    for s_idx, stock in enumerate(stocks):
        csv_path = os.path.join(data_dir, f"{stock}-level10_processed.csv")
        assert os.path.exists(csv_path), f"Missing data file: {csv_path}"
        print(f"Loading {stock} from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        train_mask, val_mask, _ = split_by_date(df)
        session_ids = detect_sessions(df)
        features = df.iloc[:, 1:].values
        stock_features.append(features)
        
        t_starts = get_window_indices(df, train_mask, session_ids, seq_len=seq_len)
        v_starts = get_window_indices(df, val_mask, session_ids, seq_len=seq_len)
        
        for st in t_starts:
            train_pairs.append((s_idx, st))
        for sv in v_starts:
            val_pairs.append((s_idx, sv))
            
        print(f"  ✓ {stock}: {len(t_starts):,} train windows, {len(v_starts):,} val windows")

    train_pairs = np.array(train_pairs, dtype=np.int32)
    val_pairs = np.array(val_pairs, dtype=np.int32)
    
    train_ds = PooledLOBDataset(stock_features, train_pairs, seq_len=seq_len)
    val_ds = PooledLOBDataset(stock_features, val_pairs, seq_len=seq_len)
    
    print(f"\nPooled Dataset Summary:")
    print(f"  Total Train Windows: {len(train_ds):,}")
    print(f"  Total Val Windows:   {len(val_ds):,}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PYTORCH LIGHTNING JEPA MODULE (SECTION 3 & SECTION 8)
# ─────────────────────────────────────────────────────────────────────────────

class JEPALightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module encapsulating:
      - Context Encoder (trained via backprop)
      - Target Encoder (updated strictly via EMA)
      - Predictor (trained via backprop)
      - Mask Token (learnable parameter)
      - Adam optimizer (lr=1e-4)
      - Collapse-prevention loss & representation std diagnostic
    """
    def __init__(self, variant: str = '3a', lr: float = 1e-4, total_steps: int = 100000,
                 n_features: int = 40, d_model: int = 256, seq_len: int = 100,
                 lambda_collapse: float = LAMBDA_COLLAPSE):
        super().__init__()
        assert variant in ('3a', '3b'), f"Variant must be '3a' or '3b', got {variant}"
        self.save_hyperparameters()
        self.variant = variant
        self.lr = lr
        self.total_steps = total_steps
        self.lambda_collapse = lambda_collapse
        self.seq_len = seq_len

        # 1. Context and Target Encoders
        self.context_encoder = JEPABackbone(n_features=n_features, d_model=d_model)
        self.target_encoder = JEPABackbone(n_features=n_features, d_model=d_model)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False  # target encoder never trained by backprop

        # 2. Mask Token
        self.mask_token = init_mask_token(n_features=n_features)

        # 3. Predictor
        self.predictor = Predictor(d_model=d_model, seq_len=seq_len)

    def forward(self, x):
        """Standard forward pass for context encoder only (unmasked)."""
        return self.context_encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        B = x.shape[0]

        # 1. Generate mask according to variant
        if self.variant == '3a':
            mask = temporal_mask(B, seq_len=self.seq_len, device=x.device)
            x_masked = apply_mask(x, mask, self.mask_token)
        elif self.variant == '3b':
            mask, x_corrupted = spatiotemporal_mask(x, B, seq_len=self.seq_len)
            mask = mask.to(x.device)
            x_masked = apply_mask(x_corrupted, mask, self.mask_token)

        # 2. Context encoder forward on masked input
        context_output = self.context_encoder(x_masked)      # [B, 100, 256]

        # 3. Target encoder forward on FULL, unmasked input (no gradient)
        with torch.no_grad():
            target_output = self.target_encoder(x).detach()   # [B, 100, 256]

        # 4. Predictor maps context output to target output prediction
        predicted_output = self.predictor(context_output)     # [B, 100, 256]

        # 5. Losses
        loss_pred = prediction_loss(predicted_output, target_output, mask)
        loss_collapse = collapse_prevention_loss(context_output)
        total_loss = loss_pred + self.lambda_collapse * loss_collapse

        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_pred', loss_pred, on_step=False, on_epoch=True)
        self.log('loss_collapse', loss_collapse, on_step=False, on_epoch=True)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update target encoder via EMA immediately after optimizer step."""
        step = self.global_step
        momentum = get_momentum(step, self.total_steps, start=0.996, end=1.0)
        update_target_encoder(self.context_encoder, self.target_encoder, momentum)

    def validation_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        B = x.shape[0]

        if self.variant == '3a':
            mask = temporal_mask(B, seq_len=self.seq_len, device=x.device)
            x_masked = apply_mask(x, mask, self.mask_token)
        elif self.variant == '3b':
            mask, x_corrupted = spatiotemporal_mask(x, B, seq_len=self.seq_len)
            mask = mask.to(x.device)
            x_masked = apply_mask(x_corrupted, mask, self.mask_token)

        context_output = self.context_encoder(x_masked)
        with torch.no_grad():
            target_output = self.target_encoder(x).detach()
        predicted_output = self.predictor(context_output)

        loss_pred = prediction_loss(predicted_output, target_output, mask)
        loss_collapse = collapse_prevention_loss(context_output)
        total_loss = loss_pred + self.lambda_collapse * loss_collapse

        # Cheap, direct collapse check: standard deviation across batch & time
        z_flat = context_output.reshape(-1, context_output.shape[-1])
        rep_std = torch.sqrt(z_flat.var(dim=0) + 1e-4).mean()

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_pred', loss_pred, on_step=False, on_epoch=True)
        self.log('val_loss_collapse', loss_collapse, on_step=False, on_epoch=True)
        self.log('val_rep_std', rep_std, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        # Target encoder is NEVER trained by backpropagation
        trainable_params = (
            list(self.context_encoder.parameters()) +
            list(self.predictor.parameters()) +
            [self.mask_token]
        )
        return torch.optim.Adam(trainable_params, lr=self.lr)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CHECKPOINT LOADING & EVALUATION HARNESS ADAPTERS
# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_jepa_encoder(variant: str, ckpt_dir: str = None, device: str = 'cuda') -> nn.Module:
    """
    Selects the optimal checkpoint from jepa_checkpoints/{variant} via argmin val_loss,
    loads context_encoder weights into JEPABackbone, wraps it in JEPAEncoderForEval,
    freezes all parameters, and moves to target device.
    """
    if ckpt_dir is None:
        ckpt_dir = f"jepa_checkpoints/{variant}"
    best_path = select_optimal_checkpoint(ckpt_dir, metric_name='val_loss')

    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    prefix = 'context_encoder.'
    backbone_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            backbone_dict[k[len(prefix):]] = v

    backbone = JEPABackbone()
    load_res = backbone.load_state_dict(backbone_dict, strict=True)
    assert len(load_res.missing_keys) == 0, f"Missing keys in JEPA backbone: {load_res.missing_keys}"
    print(f"  ✓ Successfully loaded JEPA context encoder from {os.path.basename(best_path)}")

    eval_encoder = JEPAEncoderForEval(backbone)
    eval_encoder.to(device)
    eval_encoder.eval()
    for param in eval_encoder.parameters():
        param.requires_grad = False
    return eval_encoder


def precompute_and_cache_jepa_latents(variant: str, stock: str, encoder: nn.Module = None,
                                      out_dir: str = "latents", device: str = 'cuda'):
    """
    Precomputes and caches [N, 256] latents for a JEPA variant on a given stock.
    Saves to latents/JEPA_{variant}/{stock}/
    """
    model_name = f"JEPA_{variant}"
    train_file = f"{out_dir}/{model_name}/{stock}/train_latents.npy"
    thetas_file = f"{out_dir}/{model_name}/{stock}/thetas.npy"
    if os.path.exists(train_file) and os.path.exists(thetas_file):
        print(f"  ✓ Found cached latents: {model_name}/{stock} (skipping)")
        return {}

    os.makedirs(f"{out_dir}/{model_name}/{stock}", exist_ok=True)

    csv_path = f"data/{stock}-level10_processed.csv"
    df = pd.read_csv(csv_path)
    split_indices, split_labels, thetas = compute_trend_labels_and_windows(df, k=5, seq_len=SEQ_LEN)
    features = df.iloc[:, 1:].values

    if encoder is None:
        encoder = load_frozen_jepa_encoder(variant=variant, device=device)

    cached_data = {}
    for split in ['train', 'val', 'test']:
        starts = split_indices[split]
        labels = split_labels[split]

        latents_list = []
        batch_size = 512
        for b_idx in range(0, len(starts), batch_size):
            b_starts = starts[b_idx : b_idx + batch_size]
            b_windows = np.stack([features[s : s + SEQ_LEN] for s in b_starts])
            b_tensor = torch.tensor(b_windows, dtype=torch.float32, device=device)

            with torch.no_grad():
                z = encoder(b_tensor)  # JEPAEncoderForEval produces [B, 256]
                latents_list.append(z.cpu().numpy())

        all_z = np.concatenate(latents_list, axis=0) if latents_list else np.empty((0, LATENT_DIM))
        np.save(f"{out_dir}/{model_name}/{stock}/{split}_latents.npy", all_z)
        np.save(f"{out_dir}/{model_name}/{stock}/{split}_labels.npy", labels)
        cached_data[split] = (all_z, labels)

    np.save(f"{out_dir}/{model_name}/{stock}/thetas.npy", np.array(thetas))
    print(f"  ✓ Cached {model_name}/{stock}: Train {cached_data['train'][0].shape}, Val {cached_data['val'][0].shape}, Test {cached_data['test'][0].shape} (θ_down={thetas[0]:.6f}, θ_up={thetas[1]:.6f})")
    return cached_data


def run_jepa_imputation_probe(encoder: nn.Module, stock: str, epochs: int = 50,
                              lr: float = 1e-3, batch_size: int = 256, device: str = 'cuda'):
    """
    Runs Task 2: Contiguous Imputation Probe on a frozen JEPA encoder.
    Reuses ContiguousMaskingDataset, SharedDecoder, imputation_loss, and imputation_mae
    directly from downstream_common.py.
    """
    csv_path = f"data/{stock}-level10_processed.csv"
    df = pd.read_csv(csv_path)
    train_mask, val_mask, test_mask = split_by_date(df)
    session_ids = detect_sessions(df)
    features = df.iloc[:, 1:].values

    train_starts = get_window_indices(df, train_mask, session_ids, seq_len=SEQ_LEN)
    val_starts = get_window_indices(df, val_mask, session_ids, seq_len=SEQ_LEN)
    test_starts = get_window_indices(df, test_mask, session_ids, seq_len=SEQ_LEN)

    train_ds = ContiguousMaskingDataset(features, train_starts, seq_len=SEQ_LEN, mask_len=20)
    val_ds = ContiguousMaskingDataset(features, val_starts, seq_len=SEQ_LEN, mask_len=20)
    test_ds = ContiguousMaskingDataset(features, test_starts, seq_len=SEQ_LEN, mask_len=20)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

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
                z = encoder(x_masked)  # JEPAEncoderForEval outputs [B, 256]
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
