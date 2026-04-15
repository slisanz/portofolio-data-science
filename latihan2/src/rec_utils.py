"""Utilitas bersama untuk RecSys klasik."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.data_loader import DataPaths


@dataclass
class InteractionData:
    """Matriks interaksi sparse + pemetaan ID asli ke indeks kontigu."""

    X: sp.csr_matrix           # shape (n_users, n_items), nilai = rating
    user_ids: np.ndarray       # index -> userId asli
    item_ids: np.ndarray       # index -> movieId asli
    user_idx: dict[int, int]
    item_idx: dict[int, int]
    test_pairs: list[tuple[int, int, float]]  # (u_idx, i_idx, rating)
    val_pairs: list[tuple[int, int, float]]

    @property
    def n_users(self) -> int:
        return self.X.shape[0]

    @property
    def n_items(self) -> int:
        return self.X.shape[1]


def load_interaction(
    paths: DataPaths = DataPaths(),
    train_user_cap: int | None = None,
    seed: int = 42,
) -> InteractionData:
    """Load split train/val/test dan bentuk sparse matrix indeks kontigu.

    `train_user_cap`: batasi jumlah user di train untuk eksperimen cepat
    (stratified sample dari user yang punya setidaknya 1 rating di val).
    """
    proc = paths.processed_dir
    train = pl.read_parquet(proc / "splits" / "train.parquet")
    val = pl.read_parquet(proc / "splits" / "val.parquet")
    test = pl.read_parquet(proc / "splits" / "test.parquet")

    if train_user_cap is not None:
        eval_users = pl.concat([test.select("userId"), val.select("userId")]).unique()
        rng = np.random.default_rng(seed)
        sampled = eval_users.sample(
            n=min(train_user_cap, eval_users.height), seed=seed
        )
        train = train.join(sampled, on="userId", how="inner")
        val = val.join(sampled, on="userId", how="inner")
        test = test.join(sampled, on="userId", how="inner")

    # ID mapping berdasarkan gabungan ketiga split
    all_users = np.sort(
        pl.concat([train.select("userId"), val.select("userId"), test.select("userId")])
        .unique()["userId"].to_numpy()
    )
    all_items = np.sort(
        pl.concat([train.select("movieId"), val.select("movieId"), test.select("movieId")])
        .unique()["movieId"].to_numpy()
    )
    user_idx = {int(u): i for i, u in enumerate(all_users)}
    item_idx = {int(m): i for i, m in enumerate(all_items)}

    def to_idx(df: pl.DataFrame):
        u = df["userId"].to_numpy()
        m = df["movieId"].to_numpy()
        u_i = np.fromiter((user_idx[int(x)] for x in u), dtype=np.int32, count=len(u))
        m_i = np.fromiter((item_idx[int(x)] for x in m), dtype=np.int32, count=len(m))
        r = df["rating"].to_numpy().astype(np.float32)
        return u_i, m_i, r

    tu, ti, tr = to_idx(train)
    X = sp.csr_matrix(
        (tr, (tu, ti)), shape=(len(all_users), len(all_items)), dtype=np.float32
    )

    vu, vi, vr = to_idx(val)
    teu, tei, ter = to_idx(test)
    val_pairs = list(zip(vu.tolist(), vi.tolist(), vr.tolist()))
    test_pairs = list(zip(teu.tolist(), tei.tolist(), ter.tolist()))

    return InteractionData(
        X=X,
        user_ids=all_users,
        item_ids=all_items,
        user_idx=user_idx,
        item_idx=item_idx,
        test_pairs=test_pairs,
        val_pairs=val_pairs,
    )


# ---------------------------------------------------------------- metrics


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def ranking_metrics(
    score_fn,
    data: InteractionData,
    k: int = 10,
    n_users_sample: int | None = 3000,
    seed: int = 42,
    item_pop: np.ndarray | None = None,
    item_emb: np.ndarray | None = None,
    return_topk: bool = False,
) -> dict[str, float]:
    """Evaluasi ranking top-K pada test set (leave-last-2-out).

    Untuk setiap user di test, skor semua item, exclude item yang muncul di
    train (seen), ambil top-K, dan hitung metrik vs ground truth (film test
    dengan rating >= 3.5 dianggap relevan).
    """
    X = data.X  # sparse csr: seen items di train
    by_user: dict[int, list[tuple[int, float]]] = {}
    for u, i, r in data.test_pairs:
        by_user.setdefault(u, []).append((i, r))

    users = list(by_user.keys())
    rng = np.random.default_rng(seed)
    if n_users_sample is not None and len(users) > n_users_sample:
        users = rng.choice(users, size=n_users_sample, replace=False).tolist()

    precisions, recalls, ndcgs, maps, mrrs = [], [], [], [], []
    hits_total, rel_items = [], []
    novelties, diversities, serendipities = [], [], []
    topk_per_user: dict[int, list[int]] = {}

    log2 = np.log2(np.arange(2, k + 2))
    n_total = max(int(X.sum()), 1)
    pop_prob = None
    if item_pop is not None:
        pop_prob = np.clip(item_pop / max(item_pop.sum(), 1), 1e-12, 1.0)
    popular_items_set = None
    if item_pop is not None:
        top_pop_idx = np.argsort(-item_pop)[: max(int(0.1 * data.n_items), 1)]
        popular_items_set = set(top_pop_idx.tolist())

    for u in users:
        rel_set = {i for i, r in by_user[u] if r >= 3.5}
        if not rel_set:
            continue
        scores = score_fn(u)  # array shape (n_items,)
        seen = X[u].indices
        scores = scores.copy()
        scores[seen] = -np.inf
        top_k = np.argpartition(-scores, k)[:k]
        top_k = top_k[np.argsort(-scores[top_k])]

        hits = np.fromiter((1.0 if i in rel_set else 0.0 for i in top_k), dtype=np.float32)
        n_hits = hits.sum()
        precisions.append(n_hits / k)
        recalls.append(n_hits / len(rel_set))
        dcg = float((hits / log2).sum())
        ideal_hits = np.ones(min(k, len(rel_set)))
        idcg = float((ideal_hits / log2[: len(ideal_hits)]).sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        if n_hits > 0:
            ranks = np.where(hits > 0)[0] + 1
            mrrs.append(1.0 / ranks[0])
            cum = np.cumsum(hits) / np.arange(1, k + 1)
            maps.append(float((cum * hits).sum() / min(k, len(rel_set))))
        else:
            mrrs.append(0.0)
            maps.append(0.0)
        hits_total.append(n_hits)
        rel_items.extend(top_k.tolist())
        if return_topk:
            topk_per_user[u] = top_k.tolist()

        if pop_prob is not None:
            novelties.append(float(-np.log2(pop_prob[top_k]).mean()))
        if item_emb is not None and len(top_k) > 1:
            v = item_emb[top_k]
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            sim = v @ v.T
            iu = np.triu_indices(len(top_k), k=1)
            diversities.append(float(1.0 - sim[iu].mean()))
        if popular_items_set is not None:
            unexp = [i for i in top_k if i not in popular_items_set]
            if unexp:
                rel_unexp = sum(1 for i in unexp if i in rel_set)
                serendipities.append(rel_unexp / k)
            else:
                serendipities.append(0.0)

    coverage = len(set(rel_items)) / data.n_items
    out = {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "ndcg@k": float(np.mean(ndcgs)),
        "map@k": float(np.mean(maps)),
        "mrr": float(np.mean(mrrs)),
        "coverage": coverage,
        "n_eval_users": len(precisions),
    }
    if novelties:
        out["novelty"] = float(np.mean(novelties))
    if diversities:
        out["diversity"] = float(np.mean(diversities))
    if serendipities:
        out["serendipity"] = float(np.mean(serendipities))
    if return_topk:
        out["_topk"] = topk_per_user
    return out


def rating_metrics(predict_fn, pairs: list[tuple[int, int, float]]) -> dict[str, float]:
    y_true = np.array([r for _, _, r in pairs], dtype=np.float32)
    y_pred = np.array([predict_fn(u, i) for u, i, _ in pairs], dtype=np.float32)
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
