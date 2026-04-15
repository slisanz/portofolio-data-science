"""NLP pipeline untuk free-text tags & tag-genome (Fase 5).

Fungsi publik:
- clean_tags(): normalisasi tags.csv (lowercase, strip, dedup, lemmatization opsional).
- build_movie_tag_docs(): agregasi tag per movie menjadi satu dokumen teks.
- encode_movies(): sentence-transformer embedding per movie.
- semantic_search(): query natural-language -> top-k movie.
- cluster_genome(): UMAP -> HDBSCAN pada genome embedding 1128-dim.
- tag_cooccurrence(): matrix + graph co-occurrence tag.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

PROCESSED = Path("data/processed")
NLP_DIR = PROCESSED / "nlp"
NLP_DIR.mkdir(parents=True, exist_ok=True)

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9\s\-']")


def _get_lemmatizer():
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        lemm = WordNetLemmatizer()
        return lambda w: lemm.lemmatize(w)
    except Exception:
        return lambda w: w


def normalize_tag(text: str, lemmatizer=None) -> str:
    if text is None:
        return ""
    t = text.lower().strip()
    t = _NONWORD.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if lemmatizer is not None and t:
        t = " ".join(lemmatizer(w) for w in t.split())
    return t


def clean_tags(
    tags_parquet: Path = PROCESSED / "tags.parquet",
    out_path: Path = NLP_DIR / "tags_clean.parquet",
    lemmatize: bool = True,
) -> pl.DataFrame:
    tags = pl.read_parquet(tags_parquet).select(["userId", "movieId", "tag", "timestamp"])
    lemm = _get_lemmatizer() if lemmatize else None
    tags = tags.with_columns(
        pl.col("tag")
        .map_elements(lambda s: normalize_tag(s, lemm), return_dtype=pl.Utf8)
        .alias("tag_clean")
    ).filter(pl.col("tag_clean").str.len_chars() > 0)
    tags = tags.unique(subset=["userId", "movieId", "tag_clean"])
    tags.write_parquet(out_path, compression="zstd")
    return tags


def build_movie_tag_docs(
    tags_clean: pl.DataFrame | None = None,
    min_count: int = 3,
    out_path: Path = NLP_DIR / "movie_tag_docs.parquet",
) -> pl.DataFrame:
    if tags_clean is None:
        tags_clean = pl.read_parquet(NLP_DIR / "tags_clean.parquet")
    agg = (
        tags_clean.group_by(["movieId", "tag_clean"]).agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 1)
    )
    docs = (
        agg.sort(["movieId", "n"], descending=[False, True])
        .group_by("movieId", maintain_order=True)
        .agg(
            pl.col("tag_clean").alias("tags"),
            pl.col("n").alias("counts"),
            pl.col("tag_clean").str.concat(" ").alias("doc"),
            pl.col("n").sum().alias("total_tags"),
        )
        .filter(pl.col("total_tags") >= min_count)
    )
    docs.write_parquet(out_path, compression="zstd")
    return docs


def encode_movies(
    docs: pl.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    out_emb: Path = NLP_DIR / "movie_text_embeddings.npy",
    out_ids: Path = NLP_DIR / "movie_text_ids.npy",
) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = docs["doc"].to_list()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    ids = docs["movieId"].to_numpy().astype(np.int64)
    np.save(out_emb, emb)
    np.save(out_ids, ids)
    return emb, ids


def build_faiss(emb: np.ndarray, out_path: Path = NLP_DIR / "movie_text.faiss"):
    import faiss

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_path))
    return index


def semantic_search(query: str, model, index, ids: np.ndarray, k: int = 10):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, k)
    return [(int(ids[i]), float(s)) for s, i in zip(D[0], I[0])]


def cluster_genome(
    genome_parquet: Path = PROCESSED / "genome_embedding.parquet",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 40,
    out_path: Path = NLP_DIR / "genome_clusters.parquet",
):
    import hdbscan
    import umap

    g = pl.read_parquet(genome_parquet)
    X = np.vstack([np.asarray(v, dtype=np.float32) for v in g["embedding"].to_list()])
    movie_ids = g["movieId"].to_numpy()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="cosine", random_state=42
    )
    emb2d = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(emb2d)
    out = pl.DataFrame(
        {
            "movieId": movie_ids.astype(np.uint32),
            "umap_x": emb2d[:, 0].astype(np.float32),
            "umap_y": emb2d[:, 1].astype(np.float32),
            "cluster": labels.astype(np.int32),
        }
    )
    out.write_parquet(out_path, compression="zstd")
    return out


def tag_cooccurrence(
    tags_clean: pl.DataFrame,
    top_n: int = 80,
    min_edge: int = 30,
):
    top_tags = (
        tags_clean.group_by("tag_clean").agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)["tag_clean"].to_list()
    )
    top_set = set(top_tags)
    sub = tags_clean.filter(pl.col("tag_clean").is_in(top_tags))
    grouped = sub.group_by("movieId").agg(pl.col("tag_clean").unique().alias("tags"))
    from collections import Counter
    from itertools import combinations

    co = Counter()
    for tags in grouped["tags"].to_list():
        tags = sorted(set(tags) & top_set)
        for a, b in combinations(tags, 2):
            co[(a, b)] += 1
    edges = [(a, b, c) for (a, b), c in co.items() if c >= min_edge]
    return top_tags, edges
