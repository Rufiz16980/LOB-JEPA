import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Handle pytorch-lightning import flexibility for different environments
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def detect_sessions(df):
    timestamps = pd.to_datetime(df['index'])
    deltas = timestamps.diff()
    # First row has no diff, we set it to 3s to keep it in session 0
    is_new_session = deltas != pd.Timedelta(seconds=3)
    is_new_session.iloc[0] = False
    
    # Cumulative sum yields unique session IDs
    session_ids = is_new_session.cumsum()
    return session_ids.values

def split_by_date(df):
    timestamps = pd.to_datetime(df['index'])
    dates = timestamps.dt.date
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    
    n_train = int(n_dates * 0.8)
    n_val = int(n_dates * 0.1)
    
    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train:n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val:])
    
    train_mask = dates.isin(train_dates).values
    val_mask = dates.isin(val_dates).values
    test_mask = dates.isin(test_dates).values
    
    return train_mask, val_mask, test_mask

def get_window_indices(df, mask, session_ids, seq_len=100):
    row_indices = np.where(mask)[0]
    if len(row_indices) == 0:
        return np.array([], dtype=np.int32)
    
    split_session_ids = session_ids[row_indices]
    window_starts = []
    
    for sess_id in np.unique(split_session_ids):
        sess_rows = row_indices[split_session_ids == sess_id]
        if len(sess_rows) >= seq_len:
            start_row = sess_rows[0]
            end_row = sess_rows[-1]
            
            # Verify rows are strictly contiguous
            assert end_row - start_row + 1 == len(sess_rows), "Session row indices are not contiguous!"
            
            # All valid starting row indices in this session
            window_starts.extend(range(start_row, end_row - seq_len + 2))
            
    return np.array(window_starts, dtype=np.int32)

class LOBDataset(Dataset):
    def __init__(self, features, window_indices, seq_len=100):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.window_indices = window_indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        start_idx = self.window_indices[idx]
        x = self.features[start_idx : start_idx + self.seq_len]
        return x, x  # input, target (autoencoding reconstruction)

class SharedDecoder(nn.Module):
    def __init__(self, latent_dim=256, seq_len=100, n_features=40):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, seq_len * n_features)

    def forward(self, z):  # z: [B, latent_dim]
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.seq_len, self.n_features)

class LOBReconstructionModule(pl.LightningModule):
    def __init__(self, encoder, latent_dim=256, seq_len=100, n_features=40):
        super().__init__()
        self.encoder = encoder
        self.decoder = SharedDecoder(latent_dim=latent_dim, seq_len=seq_len, n_features=n_features)
        self.loss_fn = nn.MSELoss()
        self.mae_fn = nn.L1Loss()
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self(x)
        mse = self.loss_fn(x_recon, x)
        mae = self.mae_fn(x_recon, x)
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        return {'test_mse': mse, 'test_mae': mae}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def run_experiment(encoder_class, model_name, stock, latent_dim=256, max_epochs=100):
    # Establish determinism
    set_seed(42)
    
    # Load dataset
    csv_path = f"data/{stock}-level10_processed.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded shape: {df.shape}")
    
    # Verify exact columns matching Level 10 configuration
    expected_cols = ['index'] + [f'BidPrice{i}' for i in range(10, 0, -1)] + \
                    [f'AskPrice{i}' for i in range(1, 11)] + \
                    [f'BidVolume{i}' for i in range(10, 0, -1)] + \
                    [f'AskVolume{i}' for i in range(1, 11)]
    
    actual_cols = df.columns.tolist()
    assert len(actual_cols) == 41, f"Expected 41 columns, got {len(actual_cols)}"
    for i, col in enumerate(expected_cols):
        assert actual_cols[i] == col, f"Column mismatch at index {i}: expected {col}, got {actual_cols[i]}"
    print("Columns and Level 10 ordering verified successfully.")
    
    # Session boundary detection
    print("Detecting sessions (3-second continuous chunks)...")
    session_ids = detect_sessions(df)
    print(f"Detected {len(np.unique(session_ids))} sessions.")
    
    # Split chronologically
    print("Splitting train/val/test splits (80/10/10 by calendar dates)...")
    train_mask, val_mask, test_mask = split_by_date(df)
    
    # Separate index from numerical features
    features = df.iloc[:, 1:].values
    
    # Calculate sequence window starts
    print("Constructing sequence window indices (seq_len=100, stride=1)...")
    train_starts = get_window_indices(df, train_mask, session_ids)
    val_starts = get_window_indices(df, val_mask, session_ids)
    test_starts = get_window_indices(df, test_mask, session_ids)
    
    # Print the set sizes table
    print("\n--- Train/Val/Test Splits & Windows ---")
    print(f"Split      | Rows       | Window Count")
    print(f"Train      | {np.sum(train_mask):<10} | {len(train_starts)}")
    print(f"Validation | {np.sum(val_mask):<10} | {len(val_starts)}")
    print(f"Test       | {np.sum(test_mask):<10} | {len(test_starts)}")
    print("---------------------------------------\n")
    
    # Setup PyTorch Datasets
    train_dataset = LOBDataset(features, train_starts)
    val_dataset = LOBDataset(features, val_starts)
    test_dataset = LOBDataset(features, test_starts)
    
    # Setup PyTorch DataLoaders (Uniform batch size 256)
    batch_size = 256
    num_workers = 2
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    
    # Instantiate models
    encoder = encoder_class(latent_dim=latent_dim)
    model = LOBReconstructionModule(encoder, latent_dim=latent_dim)
    
    # Print parameter counts
    encoder_param_count = sum(p.numel() for p in encoder.parameters())
    decoder_param_count = sum(p.numel() for p in model.decoder.parameters())
    total_param_count = encoder_param_count + decoder_param_count
    
    print(f"Encoder parameters: {encoder_param_count:,}")
    print(f"Shared Decoder parameters: {decoder_param_count:,}")
    print(f"Total model parameters: {total_param_count:,}")
    
    # Checkpointing paths
    # CRITICAL: Resolve to canonical absolute path via os.path.realpath() so that
    # Lightning's ModelCheckpoint always sees the same dirpath string, even when
    # Google Drive alternates between '/content/drive/MyDrive/...' and
    # '/content/drive/.shortcut-targets-by-id/...' across Colab session restarts.
    # Without this, Lightning creates versioned checkpoint files (last-v1, last-v2, ...)
    # and silently fails to overwrite them on subsequent saves, causing lost progress.
    ckpt_dir = os.path.realpath(f"checkpoints/{model_name}/{stock}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',
        save_last=True,
        save_top_k=1,
        monitor='val_mse',
        mode='min',
    )
    
    import glob
    last_ckpts = sorted(glob.glob(f"{ckpt_dir}/last*.ckpt"), key=os.path.getmtime)
    resume_path = last_ckpts[-1] if last_ckpts else None
    if resume_path:
        print(f"Found existing checkpoint at {resume_path}. Resuming training...")
    else:
        print("No prior checkpoint found. Training from scratch...")
        
    # Setup CSV logger
    logger = CSVLogger(save_dir="training_logs", name=model_name, version=stock)
    
    # Setup PyTorch Lightning Trainer
    import time as timer_lib
    start_time = timer_lib.time()
    
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
    )
    
    # Run fit (training + validation loops)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_path)
    
    training_seconds = timer_lib.time() - start_time
    print(f"Training loop finished in {training_seconds:.0f} seconds.")
    
    # Load and evaluate test metrics on the best saved checkpoint
    best_ckpts = sorted(glob.glob(f"{ckpt_dir}/best*.ckpt"), key=os.path.getmtime)
    if best_ckpts:
        best_ckpt_path = best_ckpts[-1]
        print(f"Loading best checkpoint for evaluation: {best_ckpt_path}")
        model = LOBReconstructionModule.load_from_checkpoint(best_ckpt_path, encoder=encoder, latent_dim=latent_dim)
    else:
        print("Best checkpoint not found. Evaluating with final model weights.")
        
    test_results = trainer.test(model, dataloaders=test_loader)
    test_mse = test_results[0]['test_mse']
    test_mae = test_results[0]['test_mae']
    
    # Format and print the final evaluation report as specified in the master doc
    print("\n=== FINAL REPORT ===")
    print(f"Model: {model_name} | Stock: {stock}")
    print(f"Encoder params: {encoder_param_count:,}")
    print(f"Total params (encoder + shared decoder): {total_param_count:,}")
    print(f"Training time: {training_seconds:.0f}s")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print("====================\n")
