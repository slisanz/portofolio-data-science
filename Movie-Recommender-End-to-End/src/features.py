"""Feature engineering untuk dataset MovieLens ml-latest.

Modul ini membangun feature store terpisah (user, item, genome) dan split
train/val/test berbasis timestamp (leave-last-N-out per user). Semua output
disimpan sebagai Parquet dengan kompresi zstd di `data/processed/`.

Desain:
- Komputasi selalu dimulai dari level paling rendah (per movieId atau per
  userId) agar tidak menggandakan 33M baris rating saat join metadata.
- Tag-genome disimpan sebagai kolom `embedding: list[float32]` panjang 1128,
  bukan 1128 kolom flat, supaya satu file Parquet cukup (~380 MB mentah).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import DataPaths, load_ratings_parquet


GENRES_ALL = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _movies_with_year_and_genres(paths: DataPaths) -> pl.DataFrame:
    movies = pl.read_parquet(paths.processed_dir / "movies.parquet")
    year_pat = re.compile(r"\((\d{4})\)")
    titles = movies["title"].to_list()
    release_year = np.array([
        int(m.group(1)) if (m := year_pat.search(t or "")) else -1 for t in titles
    ], dtype=np.int32)
    movies = movies.with_columns(pl.Series("release_year", release_year))
    # one-hot genre
    genre_cols = [
        pl.col("genres").str.contains(rf"(?:^|\|){re.escape(g)}(?:\||$)").cast(pl.UInt8).alias(f"g_{g}")
        for g in GENRES_ALL
    ]
    movies = movies.with_columns(genre_cols)
    return movies


def build_item_features(paths: DataPaths = DataPaths()) -> Path:
    """Fitur level film: popularitas, avg rating, std, genre one-hot, dekade."""
    ratings = load_ratings_parquet(paths).select("userId", "movieId", "rating", "rated_at")

    item_stats = (
        ratings.group_by("movieId")
        .agg(
            pl.len().alias("n_ratings"),
            pl.col("rating").mean().alias("mean_rating"),
            pl.col("rating").std().alias("std_rating"),
            pl.col("userId").n_unique().alias("n_unique_users"),
            pl.col("rated_at").min().alias("first_rated_at"),
            pl.col("rated_at").max().alias("last_rated_at"),
        )
        .collect(engine="streaming")
    )

    # popularity bucket via quantile
    n = item_stats["n_ratings"].to_numpy()
    q = np.quantile(n, [0.5, 0.8, 0.95, 0.99])
    bucket = np.searchsorted(q, n)  # 0..4
    item_stats = item_stats.with_columns(pl.Series("popularity_bucket", bucket.astype(np.uint8)))

    movies = _movies_with_year_and_genres(paths)
    item_features = item_stats.join(
        movies.with_columns(((pl.col("release_year") // 10) * 10).alias("decade")),
        on="movieId",
        how="left",
    )

    out = paths.processed_dir / "item_features.parquet"
    item_features.write_parquet(out, compression="zstd")
    return out


def build_user_features(paths: DataPaths = DataPaths()) -> Path:
    """Fitur level user: total ratings, mean, std, tenure, recency, genre-affinity."""
    ratings = load_ratings_parquet(paths).select("userId", "movieId", "rating", "rated_at")

    base = (
        ratings.group_by("userId")
        .agg(
            pl.len().alias("n_ratings"),
            pl.col("rating").mean().alias("mean_rating"),
            pl.col("rating").std().alias("std_rating"),
            pl.col("rated_at").min().alias("first_rated_at"),
            pl.col("rated_at").max().alias("last_rated_at"),
            pl.col("movieId").n_unique().alias("n_unique_movies"),
        )
        .with_columns(
            (pl.col("last_rated_at") - pl.col("first_rated_at"))
            .dt.total_days().cast(pl.Int32).alias("tenure_days"),
        )
        .collect(engine="streaming")
    )

    # Genre affinity: agregasi per genre dijalankan terpisah (loop) agar ukuran
    # intermediate tetap kecil. Untuk tiap genre kita hanya butuh dua agregat
    # per user: sum(rating di film genre tsb) dan jumlah rating di genre tsb.
    movies_g = _movies_with_year_and_genres(paths).select(
        ["movieId"] + [f"g_{g}" for g in GENRES_ALL]
    )

    user_features = base
    for g in GENRES_ALL:
        gcol = f"g_{g}"
        g_movies = movies_g.filter(pl.col(gcol) == 1).select("movieId").lazy()
        per_user = (
            ratings.join(g_movies, on="movieId", how="inner")
            .group_by("userId")
            .agg(
                pl.col("rating").sum().alias(f"_rsum_{g}"),
                pl.len().alias(f"n_{g}"),
            )
            .with_columns(
                (pl.col(f"_rsum_{g}") / pl.col(f"n_{g}").cast(pl.Float32)).alias(f"affinity_{g}")
            )
            .select("userId", f"affinity_{g}", f"n_{g}")
            .collect(engine="streaming")
        )
        user_features = user_features.join(per_user, on="userId", how="left")
    out = paths.processed_dir / "user_features.parquet"
    user_features.write_parquet(out, compression="zstd")
    return out


def build_genome_embedding(paths: DataPaths = DataPaths()) -> Path:
    """Simpan embedding genome 1128-dim per film sebagai list[float32]."""
    genome_tags = pl.read_parquet(paths.processed_dir / "genome_tags.parquet").sort("tagId")
    tag_ids = genome_tags["tagId"].to_list()
    tag_id_to_idx = {t: i for i, t in enumerate(tag_ids)}
    n_tags = len(tag_ids)

    scores = pl.read_parquet(paths.processed_dir / "genome_scores.parquet")
    # Pivot manual via numpy untuk kontrol memori. Ukuran: n_movies_in_genome x 1128.
    movie_ids = scores["movieId"].unique().sort().to_list()
    movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}
    n_movies = len(movie_ids)

    mat = np.zeros((n_movies, n_tags), dtype=np.float32)
    m_idx = np.array([movie_id_to_idx[m] for m in scores["movieId"].to_list()], dtype=np.int32)
    t_idx = np.array([tag_id_to_idx[t] for t in scores["tagId"].to_list()], dtype=np.int32)
    rel = scores["relevance"].to_numpy().astype(np.float32)
    mat[m_idx, t_idx] = rel

    df = pl.DataFrame(
        {
            "movieId": pl.Series(movie_ids, dtype=pl.UInt32),
            "embedding": [row.tolist() for row in mat],
        }
    )
    out = paths.processed_dir / "genome_embedding.parquet"
    df.write_parquet(out, compression="zstd")
    return out


def build_time_split(
    paths: DataPaths = DataPaths(),
    n_test: int = 2,
    n_val: int = 2,
    min_user_ratings: int = 10,
) -> dict[str, Path]:
    """Leave-last-N-out per user: interaksi paling baru -> test, sebelumnya -> val.

    User dengan <min_user_ratings interaksi dimasukkan semuanya ke train
    (tidak dievaluasi - cold-start akan dievaluasi terpisah).
    Menambahkan fitur temporal inline: hour, dow, movie_age.
    """
    ratings = load_ratings_parquet(paths).select("userId", "movieId", "rating", "rated_at")
    movies = _movies_with_year_and_genres(paths).select("movieId", "release_year")

    enriched = (
        ratings.join(movies.lazy(), on="movieId", how="left")
        .with_columns(
            pl.col("rated_at").dt.hour().cast(pl.UInt8).alias("hour"),
            pl.col("rated_at").dt.weekday().cast(pl.UInt8).alias("dow"),
            (pl.col("rated_at").dt.year() - pl.col("release_year")).cast(pl.Int16).alias("movie_age"),
        )
        .with_columns(
            pl.col("rated_at").rank("ordinal", descending=True).over("userId").alias("rn_desc"),
            pl.len().over("userId").alias("user_n"),
        )
    )

    # split_tag: 0=train, 1=val, 2=test
    enriched = enriched.with_columns(
        pl.when(pl.col("user_n") < min_user_ratings).then(pl.lit(0, dtype=pl.UInt8))
        .when(pl.col("rn_desc") <= n_test).then(pl.lit(2, dtype=pl.UInt8))
        .when(pl.col("rn_desc") <= (n_test + n_val)).then(pl.lit(1, dtype=pl.UInt8))
        .otherwise(pl.lit(0, dtype=pl.UInt8))
        .alias("split")
    ).drop("rn_desc", "user_n", "release_year")

    out_dir = paths.processed_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: dict[str, Path] = {}
    for name, code in [("train", 0), ("val", 1), ("test", 2)]:
        dst = out_dir / f"{name}.parquet"
        (
            enriched.filter(pl.col("split") == code)
            .drop("split")
            .sink_parquet(dst, compression="zstd")
        )
        outs[name] = dst
    return outs


def build_all(paths: DataPaths = DataPaths()) -> dict[str, Path]:
    out: dict[str, Path] = {}
    print("[features] item_features ...", flush=True)
    out["item_features"] = build_item_features(paths)
    print("[features] user_features ...", flush=True)
    out["user_features"] = build_user_features(paths)
    print("[features] genome_embedding ...", flush=True)
    out["genome_embedding"] = build_genome_embedding(paths)
    print("[features] time split ...", flush=True)
    out.update(build_time_split(paths))
    return out


if __name__ == "__main__":
    paths = DataPaths()
    result = build_all(paths)
    for k, v in result.items():
        print(f"{k}: {v}")
