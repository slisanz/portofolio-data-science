"""Evaluasi komprehensif Fase 6.

- Metrik akurasi (RMSE/MAE) + ranking lengkap (P/R/NDCG/MAP/MRR@10)
- Beyond-accuracy: Coverage, Diversity (genome emb), Novelty, Serendipity
- Skenario cold-start: segment user by train activity quartile,
  segment item by popularity quartile
- Ablation ALS factors, BPR regularization, hybrid with/without genome
- Radar chart, cold-start bar, ablation line
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
import polars as pl
import seaborn as sns

from src.data_loader import DataPaths
from src.hybrid import WeightedHybrid
from src.models.cf_knn import ItemKNN
from src.models.content_based import GenomeContentBased
from src.models.mf import ALSModel, BPRModel, SVDModel
from src.rec_utils import load_interaction, ranking_metrics, rating_metrics

FIG = ROOT / "reports" / "figures"
OUT = ROOT / "data" / "processed" / "eval"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_item_aux(data, paths: DataPaths):
    # popularity from train
    pop = np.asarray(data.X.getnnz(axis=0)).flatten().astype(np.float32)
    # genome embedding aligned to item index
    g = pl.read_parquet(paths.processed_dir / "genome_embedding.parquet")
    gmap = {int(m): np.asarray(v, dtype=np.float32) for m, v in zip(g["movieId"].to_list(), g["embedding"].to_list())}
    dim = len(next(iter(gmap.values())))
    emb = np.zeros((data.n_items, dim), dtype=np.float32)
    for mid, idx in data.item_idx.items():
        v = gmap.get(int(mid))
        if v is not None:
            emb[idx] = v
    return pop, emb


def eval_model(model, data, pop, emb, n_users=1500, supports_rmse=True):
    out = {"model": model.name}
    t0 = time.time()
    rank = ranking_metrics(
        model.score_all, data, k=10, n_users_sample=n_users,
        item_pop=pop, item_emb=emb,
    )
    out.update(rank)
    if supports_rmse:
        try:
            rm = rating_metrics(model.predict, data.test_pairs[:10000])
            out.update(rm)
        except Exception:
            out.update({"rmse": float("nan"), "mae": float("nan")})
    else:
        out.update({"rmse": float("nan"), "mae": float("nan")})
    out["eval_sec"] = round(time.time() - t0, 2)
    return out


def segment_eval(model, data, pop, emb, user_quartiles):
    """Evaluate per user activity quartile -> cold-start approximation."""
    rows = []
    X = data.X
    user_nnz = np.asarray(X.getnnz(axis=1)).flatten()
    for q_name, (lo, hi) in user_quartiles.items():
        mask_users = np.where((user_nnz >= lo) & (user_nnz < hi))[0]
        users_set = set(mask_users.tolist())
        # filter test to these users only
        orig_test = data.test_pairs
        data.test_pairs = [(u, i, r) for (u, i, r) in orig_test if u in users_set]
        if len(data.test_pairs) == 0:
            data.test_pairs = orig_test
            continue
        m = ranking_metrics(
            model.score_all, data, k=10, n_users_sample=1000,
            item_pop=pop, item_emb=emb,
        )
        m["segment"] = q_name
        m["model"] = model.name
        m["n_users_in_segment"] = int(len(mask_users))
        rows.append(m)
        data.test_pairs = orig_test
    return rows


def main():
    paths = DataPaths(
        raw_dir=ROOT / "ml-latest",
        processed_dir=ROOT / "data" / "processed",
        samples_dir=ROOT / "data" / "samples",
    )
    log("load interaction (30K users)")
    data = load_interaction(paths, train_user_cap=30_000, seed=42)
    log(f"X={data.X.shape} nnz={data.X.nnz:,} test={len(data.test_pairs)}")

    pop, emb = build_item_aux(data, paths)
    log(f"pop shape={pop.shape} emb shape={emb.shape}")

    # ------------------------------------------------------------- main bench
    log("fit models")
    als = ALSModel(factors=64, iterations=15, regularization=0.05); als.fit(data.X)
    bpr = BPRModel(factors=64, iterations=40); bpr.fit(data.X)
    svd = SVDModel(k=64); svd.fit(data.X)
    iknn = ItemKNN(k=50); iknn.fit(data.X)
    content = GenomeContentBased(paths); content.fit(data.X, data.item_ids)
    hybrid = WeightedHybrid(als, content, alpha=0.7)
    hybrid.fit(data.X)

    models_full = [
        (als, True), (bpr, False), (svd, True),
        (iknn, True), (content, False), (hybrid, False),
    ]
    rows = []
    for m, supports in models_full:
        log(f"eval {m.name}")
        rows.append(eval_model(m, data, pop, emb, supports_rmse=supports))
    df_main = pd.DataFrame(rows)
    df_main.to_csv(FIG / "final_benchmark.csv", index=False)
    log("saved final_benchmark.csv")

    # ------------------------------------------------------------- cold-start
    log("cold-start segmentation")
    X = data.X
    user_nnz = np.asarray(X.getnnz(axis=1)).flatten()
    q1, q2, q3 = np.quantile(user_nnz[user_nnz > 0], [0.25, 0.5, 0.75])
    quartiles = {
        f"Q1 (<= {int(q1)})": (0, q1 + 1),
        f"Q2 ({int(q1)+1}-{int(q2)})": (q1 + 1, q2 + 1),
        f"Q3 ({int(q2)+1}-{int(q3)})": (q2 + 1, q3 + 1),
        f"Q4 (> {int(q3)})": (q3 + 1, 1e9),
    }
    cold_rows = []
    for m, _ in [(als, True), (content, False), (hybrid, False)]:
        cold_rows.extend(segment_eval(m, data, pop, emb, quartiles))
    df_cold = pd.DataFrame(cold_rows)
    df_cold.to_csv(FIG / "coldstart_segments.csv", index=False)
    log(f"saved coldstart ({len(df_cold)} rows)")

    # ------------------------------------------------------------- ablation
    log("ablation ALS factors")
    abl = []
    for f in [16, 32, 64, 128]:
        m = ALSModel(factors=f, iterations=15, regularization=0.05); m.fit(data.X)
        r = ranking_metrics(m.score_all, data, k=10, n_users_sample=1000, item_pop=pop, item_emb=emb)
        r.update({"variant": f"ALS factors={f}", "group": "ALS_factors", "x": f})
        abl.append(r)
    log("ablation BPR learning_rate")
    for lr in [1e-3, 5e-3, 1e-2, 5e-2]:
        m = BPRModel(factors=64, iterations=40, learning_rate=lr); m.fit(data.X)
        r = ranking_metrics(m.score_all, data, k=10, n_users_sample=1000, item_pop=pop, item_emb=emb)
        r.update({"variant": f"BPR lr={lr}", "group": "BPR_lr", "x": lr})
        abl.append(r)
    log("ablation hybrid alpha (with vs without genome)")
    for a in [1.0, 0.9, 0.7, 0.5, 0.3, 0.0]:
        h = WeightedHybrid(als, content, alpha=a); h.fit(data.X)
        r = ranking_metrics(h.score_all, data, k=10, n_users_sample=1000, item_pop=pop, item_emb=emb)
        r.update({"variant": f"hybrid alpha={a}", "group": "hybrid_alpha", "x": a})
        abl.append(r)
    df_abl = pd.DataFrame(abl)
    df_abl.to_csv(FIG / "ablation.csv", index=False)
    log("saved ablation.csv")

    # ------------------------------------------------------------- viz
    log("plot radar")
    radar_metrics = ["precision@k", "recall@k", "ndcg@k", "map@k", "mrr", "coverage", "diversity", "novelty"]
    dfr = df_main.set_index("model")[radar_metrics].copy()
    dfr_norm = (dfr - dfr.min()) / (dfr.max() - dfr.min() + 1e-12)
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for model_name in dfr_norm.index:
        vals = dfr_norm.loc[model_name].tolist(); vals += vals[:1]
        ax.plot(angles, vals, label=model_name, linewidth=1.5)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Radar chart - metrik dinormalisasi (higher is better)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "29_radar_benchmark.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    log("plot coldstart")
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = df_cold.pivot(index="segment", columns="model", values="ndcg@k")
    pivot = pivot.reindex(sorted(pivot.index))
    pivot.plot(kind="bar", ax=ax, rot=0, width=0.75)
    ax.set_title("NDCG@10 per kuartil aktivitas user (cold-start segmentation)")
    ax.set_ylabel("NDCG@10")
    fig.tight_layout()
    fig.savefig(FIG / "30_coldstart.png", dpi=130)
    plt.close(fig)

    log("plot ablation")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, g in zip(axes, ["ALS_factors", "BPR_lr", "hybrid_alpha"]):
        sub = df_abl[df_abl["group"] == g].sort_values("x")
        ax.plot(sub["x"], sub["ndcg@k"], "-o", label="NDCG@10", color="#2a7fbf")
        ax.plot(sub["x"], sub["coverage"], "-s", label="Coverage", color="#c44e52")
        ax.set_title(g); ax.set_xlabel("x"); ax.legend()
        if g == "BPR_lr": ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(FIG / "31_ablation.png", dpi=130)
    plt.close(fig)

    summary = {
        "main_bench_models": int(len(df_main)),
        "coldstart_rows": int(len(df_cold)),
        "ablation_rows": int(len(df_abl)),
        "best_ndcg_model": df_main.sort_values("ndcg@k", ascending=False).iloc[0]["model"],
        "best_coverage_model": df_main.sort_values("coverage", ascending=False).iloc[0]["model"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE {summary}")


if __name__ == "__main__":
    main()
