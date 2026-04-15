from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "data" / "processed" / "dl_artifacts"
PROC = ROOT / "data" / "processed"
NLP = PROC / "nlp"


class Store:
    user_emb: np.ndarray
    item_emb: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_id_to_idx: dict
    item_id_to_idx: dict
    faiss_index: faiss.Index
    movies: pl.DataFrame
    text_emb: Optional[np.ndarray] = None
    text_index: Optional[faiss.Index] = None
    text_ids: Optional[np.ndarray] = None
    semantic_model = None


store = Store()


def _load():
    store.user_emb = np.load(ART / "two_tower_user.npy").astype("float32")
    store.item_emb = np.load(ART / "two_tower_item.npy").astype("float32")
    store.user_ids = np.load(ART / "user_ids.npy")
    store.item_ids = np.load(ART / "item_ids.npy")
    store.user_id_to_idx = {int(u): i for i, u in enumerate(store.user_ids)}
    store.item_id_to_idx = {int(m): i for i, m in enumerate(store.item_ids)}
    store.faiss_index = faiss.read_index(str(ART / "two_tower_faiss.index"))
    store.movies = pl.read_parquet(PROC / "movies.parquet")

    text_faiss = NLP / "movie_text.faiss"
    if text_faiss.exists():
        store.text_index = faiss.read_index(str(text_faiss))
        store.text_ids = np.load(NLP / "movie_text_ids.npy")


def _movie_rows(movie_ids: List[int], scores: List[float]) -> List[dict]:
    if not movie_ids:
        return []
    df = store.movies.filter(pl.col("movieId").is_in(movie_ids))
    meta = {int(r["movieId"]): r for r in df.to_dicts()}
    out = []
    for mid, sc in zip(movie_ids, scores):
        m = meta.get(int(mid), {})
        out.append(
            {
                "movieId": int(mid),
                "title": m.get("title"),
                "genres": m.get("genres"),
                "score": float(sc),
            }
        )
    return out


app = FastAPI(
    title="MovieLens RecSys API",
    version="1.0.0",
    description="Two-Tower retrieval + content similarity + semantic search.",
)


@app.on_event("startup")
def _startup():
    _load()


class RecommendItem(BaseModel):
    movieId: int
    title: Optional[str] = None
    genres: Optional[str] = None
    score: float


class RecommendResponse(BaseModel):
    userId: int
    k: int
    items: List[RecommendItem]


class SimilarResponse(BaseModel):
    movieId: int
    k: int
    items: List[RecommendItem]


class ColdStartRequest(BaseModel):
    liked_movie_ids: List[int] = Field(..., min_length=1, description="Film yang disukai user baru")
    k: int = 10


class SemanticResponse(BaseModel):
    query: str
    k: int
    items: List[RecommendItem]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "users": int(store.user_emb.shape[0]),
        "items": int(store.item_emb.shape[0]),
        "dim": int(store.item_emb.shape[1]),
        "semantic": store.text_index is not None,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    idx = store.user_id_to_idx.get(int(user_id))
    if idx is None:
        raise HTTPException(status_code=404, detail=f"user {user_id} not in index; use /cold_start")
    vec = store.user_emb[idx : idx + 1]
    scores, ids = store.faiss_index.search(vec, k)
    movie_ids = [int(store.item_ids[i]) for i in ids[0]]
    return RecommendResponse(
        userId=user_id, k=k, items=_movie_rows(movie_ids, scores[0].tolist())
    )


@app.get("/similar/{movie_id}", response_model=SimilarResponse)
def similar(movie_id: int, k: int = Query(10, ge=1, le=100)):
    idx = store.item_id_to_idx.get(int(movie_id))
    if idx is None:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not in item index")
    vec = store.item_emb[idx : idx + 1]
    scores, ids = store.faiss_index.search(vec, k + 1)
    out_ids, out_scores = [], []
    for i, s in zip(ids[0], scores[0]):
        mid = int(store.item_ids[i])
        if mid == int(movie_id):
            continue
        out_ids.append(mid)
        out_scores.append(float(s))
        if len(out_ids) == k:
            break
    return SimilarResponse(movieId=movie_id, k=k, items=_movie_rows(out_ids, out_scores))


@app.post("/cold_start", response_model=RecommendResponse)
def cold_start(req: ColdStartRequest):
    idxs = [store.item_id_to_idx[m] for m in req.liked_movie_ids if m in store.item_id_to_idx]
    if not idxs:
        raise HTTPException(status_code=400, detail="no liked movies found in catalog")
    pseudo_user = store.item_emb[idxs].mean(axis=0, keepdims=True).astype("float32")
    faiss.normalize_L2(pseudo_user)
    scores, ids = store.faiss_index.search(pseudo_user, req.k + len(idxs))
    liked = set(req.liked_movie_ids)
    out_ids, out_scores = [], []
    for i, s in zip(ids[0], scores[0]):
        mid = int(store.item_ids[i])
        if mid in liked:
            continue
        out_ids.append(mid)
        out_scores.append(float(s))
        if len(out_ids) == req.k:
            break
    return RecommendResponse(userId=-1, k=req.k, items=_movie_rows(out_ids, out_scores))


@lru_cache(maxsize=1)
def _semantic_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/semantic", response_model=SemanticResponse)
def semantic(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50)):
    if store.text_index is None:
        raise HTTPException(status_code=503, detail="semantic index unavailable")
    vec = _semantic_model().encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = store.text_index.search(vec, k)
    movie_ids = [int(store.text_ids[i]) for i in ids[0]]
    return SemanticResponse(query=q, k=k, items=_movie_rows(movie_ids, scores[0].tolist()))
