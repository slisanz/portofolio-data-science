"""Benchmark semua model klasik pada split yang sama (sample 50K user).

Output: `reports/figures/classical_bench.json` + `classical_bench.csv` +
`22_classical_bench_bar.png`.
"""

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

from src.data_loader import DataPaths
from src.hybrid import WeightedHybrid
from src.models.baseline import GlobalMean, ItemMean, Popularity, UserMean
from src.models.cf_knn import ItemKNN, UserKNN
from src.models.content_based import GenomeContentBased
from src.models.mf import ALSModel, BPRModel, SVDModel
from src.rec_utils import load_interaction, ranking_metrics, rating_metrics


def step(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    paths = DataPaths(
        raw_dir=ROOT / "ml-latest",
        processed_dir=ROOT / "data" / "processed",
        samples_dir=ROOT / "data" / "samples",
    )

    step("Loading interaction data (sample 50K users)")
    data = load_interaction(paths, train_user_cap=50_000, seed=42)
    step(f"  X shape: {data.X.shape}  nnz: {data.X.nnz:,}  "
         f"val={len(data.val_pairs)} test={len(data.test_pairs)}")

    results = []

    def run(model, supports_rmse=True):
        t0 = time.time()
        try:
            if isinstance(model, GenomeContentBased):
                model.fit(data.X, data.item_ids)
            else:
                model.fit(data.X)
        except Exception as e:
            step(f"  [FAIL fit] {model.name}: {e}")
            return
        fit_sec = time.time() - t0

        t0 = time.time()
        rank = ranking_metrics(model.score_all, data, k=10, n_users_sample=2000)
        rank_sec = time.time() - t0

        rm = {"rmse": float("nan"), "mae": float("nan")}
        if supports_rmse:
            try:
                rm = rating_metrics(model.predict, data.test_pairs[:20000])
            except Exception as e:
                step(f"  [WARN rmse] {model.name}: {e}")

        row = {
            "model": model.name,
            **rank,
            **rm,
            "fit_sec": round(fit_sec, 2),
            "rank_sec": round(rank_sec, 2),
        }
        results.append(row)
        step(f"  {model.name:30s}  P@10={row['precision@k']:.4f}  "
             f"NDCG@10={row['ndcg@k']:.4f}  Cov={row['coverage']:.3f}  "
             f"RMSE={row['rmse']:.4f}  fit={fit_sec:.1f}s")

    step("Train & eval models")
    run(GlobalMean())
    run(UserMean())
    run(ItemMean())
    run(Popularity())
    run(ItemKNN(k=50))
    run(UserKNN(k=50))
    run(SVDModel(k=64))
    als = ALSModel(factors=64, iterations=15, regularization=0.05)
    run(als)
    run(BPRModel(factors=64, iterations=40), supports_rmse=False)

    content = GenomeContentBased(paths)
    run(content, supports_rmse=False)

    hybrid = WeightedHybrid(als, content, alpha=0.7)
    run(hybrid, supports_rmse=False)

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(fig_dir / "classical_bench.csv", index=False)
    (fig_dir / "classical_bench.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    step("Plot benchmark bar")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df_sorted = df.sort_values("ndcg@k", ascending=True)
    sns.barplot(data=df_sorted, y="model", x="ndcg@k", ax=axes[0], color="#4C72B0")
    axes[0].set_title("NDCG@10 per model")
    sns.barplot(data=df_sorted, y="model", x="precision@k", ax=axes[1], color="#55A868")
    axes[1].set_title("Precision@10 per model")
    sns.barplot(data=df_sorted, y="model", x="coverage", ax=axes[2], color="#C44E52")
    axes[2].set_title("Coverage per model")
    plt.tight_layout()
    plt.savefig(fig_dir / "22_classical_bench_bar.png", bbox_inches="tight", dpi=150)
    plt.close()
    step(f"DONE. Results -> {fig_dir / 'classical_bench.csv'}")


if __name__ == "__main__":
    main()
