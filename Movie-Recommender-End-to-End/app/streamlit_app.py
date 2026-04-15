"""Streamlit dashboard portfolio MovieLens RecSys.

Empat tab: EDA Explorer, Recommender, Semantic Search, Model Arena.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
ART = PROC / "dl_artifacts"
NLP = PROC / "nlp"

st.set_page_config(page_title="MovieLens RecSys Portfolio", layout="wide")


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    return pl.read_parquet(PROC / "movies.parquet").to_pandas()


@st.cache_data(show_spinner=False)
def load_bench() -> pd.DataFrame:
    return pd.read_csv(FIG / "final_benchmark.csv")


@st.cache_data(show_spinner=False)
def load_classical_bench() -> pd.DataFrame:
    return pd.read_csv(FIG / "classical_bench.csv")


@st.cache_data(show_spinner=False)
def load_item_features() -> pd.DataFrame:
    p = PROC / "item_features.parquet"
    cols = ["movieId", "rating_count", "rating_mean", "year"]
    df = pl.read_parquet(p)
    keep = [c for c in cols if c in df.columns]
    return df.select(keep).to_pandas()


@st.cache_resource(show_spinner=True)
def load_twotower():
    import faiss

    user_emb = np.load(ART / "two_tower_user.npy").astype("float32")
    item_emb = np.load(ART / "two_tower_item.npy").astype("float32")
    user_ids = np.load(ART / "user_ids.npy")
    item_ids = np.load(ART / "item_ids.npy")
    index = faiss.read_index(str(ART / "two_tower_faiss.index"))
    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "uid2idx": {int(u): i for i, u in enumerate(user_ids)},
        "mid2idx": {int(m): i for i, m in enumerate(item_ids)},
        "index": index,
    }


@st.cache_resource(show_spinner=True)
def load_text_index():
    import faiss

    p = NLP / "movie_text.faiss"
    if not p.exists():
        return None
    return {
        "index": faiss.read_index(str(p)),
        "ids": np.load(NLP / "movie_text_ids.npy"),
    }


@st.cache_resource(show_spinner=True)
def load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _attach_meta(movie_ids: list[int], scores: list[float], movies: pd.DataFrame) -> pd.DataFrame:
    df = movies[movies["movieId"].isin(movie_ids)][["movieId", "title", "genres"]]
    order = {m: i for i, m in enumerate(movie_ids)}
    df = df.assign(_ord=df["movieId"].map(order)).sort_values("_ord").drop(columns="_ord")
    df["score"] = [scores[order[m]] for m in df["movieId"]]
    return df.reset_index(drop=True)


st.title("MovieLens RecSys — Portfolio Dashboard")
st.caption("Dataset ml-latest (Juli 2023) • 33.8M rating • 331K user • 83K film")

tab_eda, tab_rec, tab_sem, tab_arena = st.tabs(
    ["EDA Explorer", "Recommender", "Semantic Search", "Model Arena"]
)


# ---------- TAB 1: EDA ----------
with tab_eda:
    st.subheader("EDA Explorer")
    movies = load_movies()
    feats = load_item_features()
    df = movies.merge(feats, on="movieId", how="left")

    all_genres = sorted({g for gs in movies["genres"].dropna() for g in gs.split("|") if g != "(no genres listed)"})
    c1, c2, c3 = st.columns(3)
    with c1:
        yr = st.slider(
            "Rentang tahun rilis",
            min_value=int(df["year"].dropna().min() or 1902),
            max_value=int(df["year"].dropna().max() or 2023),
            value=(1990, 2020),
        )
    with c2:
        sel_genres = st.multiselect("Genre (intersection)", all_genres, default=[])
    with c3:
        min_ratings = st.number_input("Min rating_count", 0, 100000, 50, step=50)

    mask = (df["year"].between(yr[0], yr[1])) & (df["rating_count"].fillna(0) >= min_ratings)
    if sel_genres:
        for g in sel_genres:
            mask &= df["genres"].fillna("").str.contains(g, regex=False)
    sub = df[mask].copy()

    m1, m2, m3 = st.columns(3)
    m1.metric("Film (filter)", f"{len(sub):,}")
    m2.metric("Rating rata-rata", f"{sub['rating_mean'].mean():.3f}" if len(sub) else "-")
    m3.metric("Total interaksi", f"{int(sub['rating_count'].sum()):,}" if len(sub) else "-")

    if len(sub):
        fig = px.scatter(
            sub.sample(min(5000, len(sub)), random_state=0),
            x="rating_count",
            y="rating_mean",
            hover_data=["title", "year"],
            log_x=True,
            opacity=0.5,
            title="Popularitas vs rating rata-rata",
        )
        st.plotly_chart(fig, use_container_width=True)

        top = sub.sort_values("rating_count", ascending=False).head(20)[
            ["title", "year", "genres", "rating_count", "rating_mean"]
        ]
        st.markdown("**Top-20 film paling banyak dirating (sesuai filter):**")
        st.dataframe(top, use_container_width=True, hide_index=True)

    with st.expander("Galeri figur EDA statis (Fase 1–5)"):
        gallery = sorted(FIG.glob("*.png"))
        cols = st.columns(3)
        for i, p in enumerate(gallery[:18]):
            with cols[i % 3]:
                st.image(str(p), caption=p.name, use_container_width=True)


# ---------- TAB 2: Recommender ----------
with tab_rec:
    st.subheader("Recommender (Two-Tower)")
    tt = load_twotower()
    movies = load_movies()

    mode = st.radio("Mode", ["User ID", "Cold-start (pilih film yang disukai)"], horizontal=True)
    k = st.slider("Top-K", 5, 50, 10)

    if mode == "User ID":
        default_uid = int(tt["user_ids"][0])
        uid = st.number_input("User ID", min_value=1, value=default_uid, step=1)
        if st.button("Rekomendasikan", type="primary"):
            idx = tt["uid2idx"].get(int(uid))
            if idx is None:
                st.error(f"User {uid} tidak ada di index Two-Tower (sampel training). Coba mode cold-start.")
            else:
                vec = tt["user_emb"][idx : idx + 1]
                scores, ids = tt["index"].search(vec, k)
                mids = [int(tt["item_ids"][i]) for i in ids[0]]
                st.dataframe(_attach_meta(mids, scores[0].tolist(), movies), use_container_width=True, hide_index=True)
    else:
        options = movies.sort_values("movieId").head(5000)
        picks = st.multiselect(
            "Pilih minimal 1 film yang disukai",
            options=options["movieId"].tolist(),
            format_func=lambda m: f"{m} - {movies.loc[movies['movieId']==m,'title'].iloc[0]}",
            max_selections=15,
        )
        if st.button("Rekomendasikan", type="primary") and picks:
            import faiss

            idxs = [tt["mid2idx"][m] for m in picks if m in tt["mid2idx"]]
            if not idxs:
                st.warning("Film tidak ada di katalog Two-Tower.")
            else:
                pseudo = tt["item_emb"][idxs].mean(axis=0, keepdims=True).astype("float32")
                faiss.normalize_L2(pseudo)
                scores, ids = tt["index"].search(pseudo, k + len(idxs))
                liked = set(picks)
                mids, sc = [], []
                for i, s in zip(ids[0], scores[0]):
                    mid = int(tt["item_ids"][i])
                    if mid in liked:
                        continue
                    mids.append(mid)
                    sc.append(float(s))
                    if len(mids) == k:
                        break
                st.dataframe(_attach_meta(mids, sc, movies), use_container_width=True, hide_index=True)


# ---------- TAB 3: Semantic ----------
with tab_sem:
    st.subheader("Semantic Search (tag-based)")
    tx = load_text_index()
    movies = load_movies()
    if tx is None:
        st.warning("Index semantic belum ter-build (Fase 5).")
    else:
        q = st.text_input("Query (mis. 'mind-bending sci-fi thriller')", value="mind-bending sci-fi thriller")
        k = st.slider("Top-K", 5, 30, 10, key="sem_k")
        if st.button("Cari"):
            model = load_sbert()
            vec = model.encode([q], normalize_embeddings=True).astype("float32")
            scores, ids = tx["index"].search(vec, k)
            mids = [int(tx["ids"][i]) for i in ids[0]]
            st.dataframe(_attach_meta(mids, scores[0].tolist(), movies), use_container_width=True, hide_index=True)


# ---------- TAB 4: Arena ----------
with tab_arena:
    st.subheader("Model Arena — Benchmark Akhir")
    bench = load_bench()
    st.dataframe(bench, use_container_width=True, hide_index=True)

    metric_cols = [c for c in bench.columns if c not in ("model", "n_eval_users", "eval_sec")]
    picked = st.multiselect(
        "Metrik",
        metric_cols,
        default=["ndcg@k", "recall@k", "coverage", "diversity", "novelty"],
    )
    if picked:
        norm = bench[["model"] + picked].copy()
        for c in picked:
            v = norm[c].astype(float)
            rng = v.max() - v.min()
            norm[c] = 0.0 if rng == 0 else (v - v.min()) / rng
        long = norm.melt(id_vars="model", var_name="metric", value_name="score")
        fig = px.line_polar(
            long, r="score", theta="metric", color="model", line_close=True,
            title="Radar perbandingan (skala 0–1 per metrik)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Figur statis Fase 6"):
        for name in ("29_radar_benchmark.png", "30_coldstart.png", "31_ablation.png"):
            p = FIG / name
            if p.exists():
                st.image(str(p), caption=name, use_container_width=True)

st.caption("API backend: set env `API_URL` untuk menghubungkan ke FastAPI (Fase 7). Dashboard ini sudah mandiri.")
