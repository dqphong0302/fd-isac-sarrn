#!/usr/bin/env python3
"""
training_v36_optimized.py
Strategy 3: Robust Training with Auxiliary Loss

Optimizations:
1. Residual Learning (h_true = h_ls + delta)
2. Auxiliary Loss: MSE(|h|, |h_hat|) to learn magnitude strictly
3. Input Augmentation: Dropout (0.2)
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class ResidualMLP(nn.Module):
    def __init__(self, in_dim=210, width=512, depth=4, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 8))  # delta_h
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, h_rls):
        delta = self.net(x)
        return h_rls + delta


class OptimizedDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            X_raw = f['/X'][:]
            Y_raw = f['/Y'][:]
            # Handle MATLAB column-major
            if X_raw.shape[0] < X_raw.shape[1]: X_raw = X_raw.T
            if Y_raw.shape[0] < Y_raw.shape[1]: Y_raw = Y_raw.T
            
            self.X = X_raw.astype(np.float32)
            self.Y = Y_raw.astype(np.float32)
            self.Hrls = self.X[:, -8:]
            
            # Check optimization flag
            if '/meta/optimized' in f:
                print("Dataset: Optimized (Robust LS)")
            else:
                print("Dataset: Standard")
                
            self.n_samples, self.x_dim = self.X.shape
            print(f"Loaded {self.n_samples} samples, {self.x_dim} features")

    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]), torch.from_numpy(self.Hrls[i])


def nmse_loss(yhat, y, eps=1e-12):
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def auxiliary_magnitude_loss(yhat, y):
    # yhat: [B, 8] -> [B, 4] complex
    # Treat first 4 real, last 4 imag
    h_real = yhat[:, :4]
    h_imag = yhat[:, 4:]
    mag_hat = torch.sqrt(h_real**2 + h_imag**2 + 1e-8)
    
    y_real = y[:, :4]
    y_imag = y[:, 4:]
    mag_true = torch.sqrt(y_real**2 + y_imag**2 + 1e-8)
    
    return F.mse_loss(mag_hat, mag_true)


def train_epoch(model, loader, optimizer, device, scaler, alpha=0.5):
    model.train()
    total_loss = 0.0
    total_nmse = 0.0
    n_batches = 0
    
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            yhat = model(x, h_rls)
            nmse = nmse_loss(yhat, y)
            mag_loss = auxiliary_magnitude_loss(yhat, y)
            
            # Total Loss = NMSE + alpha * Magnitude_Loss
            loss = nmse + alpha * mag_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_nmse += nmse.item()
        n_batches += 1
        
    return total_loss / n_batches, total_nmse / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_nmse = 0.0
    n_batches = 0
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        yhat = model(x, h_rls)
        total_nmse += nmse_loss(yhat, y).item()
        n_batches += 1
    return total_nmse / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="dataset_v36_optimized.h5")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5, help="Aux loss weight")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.h5):
        print("Dataset not found. Please run generate_dataset_v36_optimized.m")
        sys.exit(1)
        
    ds = OptimizedDataset(args.h5)
    n_samples = len(ds)
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    
    model = ResidualMLP(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V3.6 Optimized (Robust LS + Aux Loss)")
    print("="*60 + "\n")
    
    for epoch in range(1, args.epochs+1):
        loss, train_nmse = train_epoch(model, train_loader, optimizer, device, scaler, args.alpha)
        val_nmse = evaluate(model, val_loader, device)
        scheduler.step()
        
        train_db = 10*np.log10(train_nmse + 1e-12)
        val_db = 10*np.log10(val_nmse + 1e-12)
        
        saved = ""
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'in_dim': 210, 'width': 512, 'depth': 4,
                'val_nmse': val_nmse
            }, "best_v36_opt.pt")
            saved = " *"
            
        print(f"[ep {epoch:03d}] Loss: {loss:.4f} | Train: {train_db:.2f} dB | Val: {val_db:.2f} dB{saved}")
        
    print(f"\nBest Val NMSE: {10*np.log10(best_nmse):.2f} dB")
    print("Checkpoint: best_v36_opt.pt")


if __name__ == "__main__":
    main()
