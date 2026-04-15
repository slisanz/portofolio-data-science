"""Latih NCF, Two-Tower, SASRec pada sampel user kecil, lalu ekspor artefak."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.data_loader import DataPaths
from src.dl_train import (
    ARTIFACT_DIR, TrainConfig, device_auto,
    export_two_tower, train_ncf, train_sasrec, train_two_tower,
)
from src.rec_utils import load_interaction, ranking_metrics

sns.set_theme(style="whitegrid", context="notebook")


def step(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    paths = DataPaths(
        raw_dir=ROOT / "ml-latest",
        processed_dir=ROOT / "data" / "processed",
        samples_dir=ROOT / "data" / "samples",
    )
    device = device_auto()
    step(f"Device: {device}")

    step("Load interaction (sample 10K user untuk DL - CPU-friendly)")
    data = load_interaction(paths, train_user_cap=10_000, seed=42)
    step(f"  X={data.X.shape}  nnz={data.X.nnz:,}  "
         f"val={len(data.val_pairs):,} test={len(data.test_pairs):,}")

    cfg = TrainConfig(epochs=3, batch_size=4096, lr=1e-3, patience=2)

    results = []

    step("Train NCF")
    ncf_model, ncf_res = train_ncf(data, device, cfg)
    step(f"  NCF best NDCG@10={ncf_res.best_ndcg:.4f} in {ncf_res.train_sec:.1f}s")

    step("Train Two-Tower")
    tt_model, tt_res = train_two_tower(data, device, cfg)
    step(f"  TwoTower best NDCG@10={tt_res.best_ndcg:.4f} in {tt_res.train_sec:.1f}s")

    step("Train SASRec")
    sas_model, sas_res = train_sasrec(data, device, cfg, max_len=30)
    step(f"  SASRec best NDCG@10={sas_res.best_ndcg:.4f} in {sas_res.train_sec:.1f}s")

    for r in [ncf_res, tt_res, sas_res]:
        results.append({
            "model": r.name, "best_ndcg@10": r.best_ndcg,
            "train_sec": round(r.train_sec, 1), "n_epochs": len(r.history),
            "history": r.history,
        })

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "dl_bench.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Flatten history for plotting
    flat = []
    for r in results:
        for h in r["history"]:
            flat.append({"model": r["model"], **h})
    flat_df = pd.DataFrame(flat)
    flat_df.to_csv(fig_dir / "dl_bench_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in flat_df["model"].unique():
        sub = flat_df[flat_df["model"] == m]
        axes[0].plot(sub["epoch"], sub["loss"], marker="o", label=m)
        axes[1].plot(sub["epoch"], sub["ndcg@10"], marker="o", label=m)
    axes[0].set_title("Training loss per epoch"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Val NDCG@10 per epoch"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "23_dl_training_curves.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Gabungkan ke benchmark keseluruhan
    summary = pd.DataFrame([
        {"model": r["model"], "best_ndcg@10": r["best_ndcg@10"],
         "train_sec": r["train_sec"], "n_epochs": r["n_epochs"]}
        for r in results
    ])
    summary.to_csv(fig_dir / "dl_bench.csv", index=False)

    step("Ekspor Two-Tower (npy + FAISS + TorchScript)")
    arts = export_two_tower(tt_model, data, device, out_dir=ROOT / ARTIFACT_DIR)
    (ROOT / ARTIFACT_DIR / "export_manifest.json").write_text(
        json.dumps(arts, indent=2), encoding="utf-8"
    )
    step(f"  artifacts: {arts}")

    # FAISS smoke test
    import faiss
    index = faiss.read_index(arts["faiss_index"])
    user_mat = np.load(arts["user_npy"])
    D, I = index.search(user_mat[:5].astype(np.float32), 10)
    step(f"  FAISS smoke: top-10 item idx for 5 user:\n{I}")

    step("DONE")


if __name__ == "__main__":
    main()
