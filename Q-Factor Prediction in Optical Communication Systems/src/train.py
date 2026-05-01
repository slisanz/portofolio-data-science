from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True
    show_batch_bar: bool = False


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def train_torch_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig | None = None,
    loss_fn: Callable | None = None,
) -> dict:
    cfg = cfg or TrainConfig()
    model = model.to(cfg.device)
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    loss_fn = loss_fn or torch.nn.functional.mse_loss

    best_val = float("inf")
    best_state = None
    wait = 0
    history: dict = {"train": [], "val": []}

    epoch_iter = tqdm(range(cfg.epochs), desc="epoch", disable=not cfg.verbose, leave=True,
                      mininterval=1.0, ncols=80, dynamic_ncols=False, file=sys.stdout)
    for epoch in epoch_iter:
        t0 = time.time()
        model.train()
        train_losses = []
        batch_iter = tqdm(
            train_loader,
            desc=f"  ep {epoch+1:02d}/{cfg.epochs}",
            leave=False,
            disable=not (cfg.verbose and cfg.show_batch_bar),
            mininterval=1.0, ncols=80, dynamic_ncols=False, file=sys.stdout,
        )
        for xb, yb in batch_iter:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            optim.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device); yb = yb.to(cfg.device)
                preds = model(xb)
                val_losses.append(loss_fn(preds, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            improved = "*"
        else:
            wait += 1
            improved = " "

        dt = time.time() - t0
        epoch_iter.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", best=f"{best_val:.4f}", patience=f"{wait}/{cfg.patience}")
        if cfg.verbose:
            tqdm.write(f"  ep {epoch+1:02d}/{cfg.epochs} {improved}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}  ({dt:.1f}s)")

        if wait >= cfg.patience:
            if cfg.verbose:
                tqdm.write(f"  early-stop at epoch {epoch+1} (no val improvement for {cfg.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_val, "model": model}


@torch.no_grad()
def predict_torch(model: torch.nn.Module, X: np.ndarray, batch_size: int = 8192, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)
