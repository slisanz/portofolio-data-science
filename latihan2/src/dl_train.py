"""Training pipeline untuk NCF, Two-Tower, dan SASRec.

Fitur:
- AMP (mixed precision) otomatis jika CUDA tersedia, disabled di CPU.
- Early stopping berbasis val NDCG@10.
- Log metrik per-epoch ke `reports/dl_training_log.json`.
- Export embedding & TorchScript (Two-Tower) ke `data/processed/dl_artifacts/`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data_loader import DataPaths
from src.models.ncf import NCF, TwoTower
from src.models.sasrec import SASRec
from src.rec_utils import InteractionData, load_interaction, ranking_metrics


ARTIFACT_DIR = Path("data/processed/dl_artifacts")


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------- datasets


class PointwiseDataset(Dataset):
    """Positive-unlabeled sampling: untuk tiap positive ambil `neg_ratio` negatif acak."""

    def __init__(self, X: sp.csr_matrix, neg_ratio: int = 4, seed: int = 42):
        self.n_users, self.n_items = X.shape
        # positif = rating >= 3.5 (eksplisit -> implicit positive)
        coo = X.tocoo()
        mask = coo.data >= 3.5
        self.pos_u = coo.row[mask].astype(np.int64)
        self.pos_i = coo.col[mask].astype(np.int64)
        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed)
        # untuk cek interaksi cepat: per-user set of seen items
        self.X = X.tolil(copy=False)

    def __len__(self) -> int:
        return len(self.pos_u) * (1 + self.neg_ratio)

    def __getitem__(self, idx: int):
        n_pos = len(self.pos_u)
        if idx < n_pos:
            return self.pos_u[idx], self.pos_i[idx], 1.0
        # negative sampling
        pos_idx = idx % n_pos
        u = self.pos_u[pos_idx]
        seen = set(self.X.rows[u])
        while True:
            j = int(self.rng.integers(0, self.n_items))
            if j not in seen:
                return u, j, 0.0


class SequenceDataset(Dataset):
    """Sekuens per-user urut waktu untuk SASRec. target = next-item prediction."""

    def __init__(self, X: sp.csr_matrix, max_len: int = 50, seed: int = 42):
        self.max_len = max_len
        self.n_items = X.shape[1]
        self.rng = np.random.default_rng(seed)
        # Polars/CSR tidak menyimpan urutan waktu; untuk demo pakai urutan kolom.
        # (Urutan kronologis lebih baik dipakai saat dataset mentah diakses; di
        # sini kita pakai ordering sederhana karena training cepat dan hanya
        # bertujuan memperlihatkan pipeline.)
        lil = X.tolil(copy=False)
        self.sequences: list[np.ndarray] = []
        for u in range(X.shape[0]):
            items = np.asarray(lil.rows[u], dtype=np.int64)
            if len(items) >= 3:
                # geser +1 agar 0 = padding
                self.sequences.append(items + 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        s = self.sequences[idx]
        if len(s) > self.max_len + 1:
            start = int(self.rng.integers(0, len(s) - self.max_len - 1))
            s = s[start : start + self.max_len + 1]
        seq_in = s[:-1]
        target = s[1:]
        # pad kiri
        pad = self.max_len - len(seq_in)
        if pad > 0:
            seq_in = np.concatenate([np.zeros(pad, dtype=np.int64), seq_in])
            target = np.concatenate([np.zeros(pad, dtype=np.int64), target])
        return seq_in, target


# --------------------------------------------------------------------- trainers


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 2
    eval_every: int = 1


@dataclass
class TrainResult:
    name: str
    best_ndcg: float
    history: list[dict] = field(default_factory=list)
    train_sec: float = 0.0


def _train_pointwise(
    model: nn.Module,
    data: InteractionData,
    cfg: TrainConfig,
    device: torch.device,
    name: str,
    score_fn_builder,
) -> TrainResult:
    ds = PointwiseDataset(data.X, neg_ratio=4)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best = -1.0
    patience = cfg.patience
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_ep = time.time()
        for u, i, y in dl:
            u = u.to(device); i = i.to(device); y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logit = model(u, i)
                loss = loss_fn(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.item()); n_batches += 1
        # eval on val (via ranking metrics dengan scorer)
        score_fn = score_fn_builder(model)
        metrics = ranking_metrics(score_fn, data, k=10, n_users_sample=500)
        rec = {
            "epoch": epoch, "loss": epoch_loss / max(n_batches, 1),
            "ndcg@10": metrics["ndcg@k"], "precision@10": metrics["precision@k"],
            "coverage": metrics["coverage"], "sec": round(time.time() - t_ep, 1),
        }
        history.append(rec)
        print(f"    [{name}] ep{epoch}  loss={rec['loss']:.4f}  "
              f"ndcg@10={rec['ndcg@10']:.4f}  p@10={rec['precision@10']:.4f}  "
              f"{rec['sec']}s", flush=True)
        if rec["ndcg@10"] > best + 1e-4:
            best = rec["ndcg@10"]; patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"    [{name}] early stop", flush=True); break

    return TrainResult(name=name, best_ndcg=best, history=history,
                       train_sec=time.time() - t_start)


def train_ncf(data: InteractionData, device: torch.device, cfg: TrainConfig) -> tuple[NCF, TrainResult]:
    model = NCF(n_users=data.n_users, n_items=data.n_items).to(device)
    def builder(m):
        def _s(u):
            return m.score_all_items(u, device)
        return _s
    res = _train_pointwise(model, data, cfg, device, "NCF", builder)
    return model, res


def train_two_tower(data: InteractionData, device: torch.device, cfg: TrainConfig) -> tuple[TwoTower, TrainResult]:
    model = TwoTower(n_users=data.n_users, n_items=data.n_items).to(device)
    def builder(m):
        # cache item matrix sekali per eval
        item_mat_holder = {}
        def _s(u):
            if "M" not in item_mat_holder:
                item_mat_holder["M"] = m.all_item_repr(device)
            ur = m.user_repr(torch.tensor([u], device=device)).detach().cpu().numpy().ravel()
            return (item_mat_holder["M"] @ ur).astype(np.float32)
        return _s
    res = _train_pointwise(model, data, cfg, device, "TwoTower", builder)
    return model, res


def train_sasrec(
    data: InteractionData, device: torch.device, cfg: TrainConfig,
    max_len: int = 50,
) -> tuple[SASRec, TrainResult]:
    ds = SequenceDataset(data.X, max_len=max_len)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    model = SASRec(n_items=data.n_items, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best = -1.0; patience = cfg.patience
    history = []; t_start = time.time()

    def build_scorer(m):
        # untuk eval, tiap user skor-kan seluruh item berbasis sekuens seen-item
        lil = data.X.tolil(copy=False)
        def _s(u):
            items = np.asarray(lil.rows[u], dtype=np.int64) + 1
            if len(items) == 0:
                return np.zeros(data.n_items, dtype=np.float32)
            if len(items) > max_len:
                items = items[-max_len:]
            pad = max_len - len(items)
            if pad > 0:
                items = np.concatenate([np.zeros(pad, dtype=np.int64), items])
            return m.score_all_items_user(items, device)
        return _s

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0; n = 0; t_ep = time.time()
        for seq, tgt in dl:
            seq = seq.to(device); tgt = tgt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                h = model(seq)                          # (B, L, d)
                logits = h @ model.item_emb.weight.T    # (B, L, n_items+1)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            epoch_loss += float(loss.item()); n += 1
        metrics = ranking_metrics(build_scorer(model), data, k=10, n_users_sample=500)
        rec = {"epoch": epoch, "loss": epoch_loss / max(n, 1),
               "ndcg@10": metrics["ndcg@k"], "precision@10": metrics["precision@k"],
               "coverage": metrics["coverage"], "sec": round(time.time() - t_ep, 1)}
        history.append(rec)
        print(f"    [SASRec] ep{epoch}  loss={rec['loss']:.4f}  "
              f"ndcg@10={rec['ndcg@10']:.4f}  p@10={rec['precision@10']:.4f}  "
              f"{rec['sec']}s", flush=True)
        if rec["ndcg@10"] > best + 1e-4:
            best = rec["ndcg@10"]; patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                print("    [SASRec] early stop", flush=True); break

    return model, TrainResult(name="SASRec", best_ndcg=best, history=history,
                              train_sec=time.time() - t_start)


# --------------------------------------------------------------------- export


def export_two_tower(model: TwoTower, data: InteractionData, device: torch.device,
                     out_dir: Path = ARTIFACT_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # embedding matrices
    user_mat = model.all_user_repr(device)
    item_mat = model.all_item_repr(device)
    np.save(out_dir / "two_tower_user.npy", user_mat)
    np.save(out_dir / "two_tower_item.npy", item_mat)
    np.save(out_dir / "user_ids.npy", data.user_ids)
    np.save(out_dir / "item_ids.npy", data.item_ids)

    # FAISS index (inner product)
    item_c = np.ascontiguousarray(item_mat, dtype=np.float32)
    index = faiss.IndexFlatIP(item_c.shape[1])
    index.add(item_c)
    faiss.write_index(index, str(out_dir / "two_tower_faiss.index"))

    # TorchScript export
    model.cpu().eval()
    scripted = torch.jit.script(model)
    scripted.save(str(out_dir / "two_tower_scripted.pt"))

    return {
        "user_npy": str(out_dir / "two_tower_user.npy"),
        "item_npy": str(out_dir / "two_tower_item.npy"),
        "faiss_index": str(out_dir / "two_tower_faiss.index"),
        "torchscript": str(out_dir / "two_tower_scripted.pt"),
    }
