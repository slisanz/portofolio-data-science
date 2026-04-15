"""Data loader untuk dataset MovieLens ml-latest.

Tugas utama:
- Membaca CSV besar (ratings 891 MB, tags 82 MB, genome-scores 498 MB) secara
  efisien memakai Polars (lazy / streaming) dengan type downcasting.
- Mengkonversi CSV ke Parquet ber-partisi berdasarkan tahun rating agar bisa
  dibaca cepat untuk EDA maupun training.
- Menyediakan helper untuk stratified sample berbasis user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


RAW_DIR_DEFAULT = Path("ml-latest")
PROCESSED_DIR_DEFAULT = Path("data/processed")
SAMPLES_DIR_DEFAULT = Path("data/samples")


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path = RAW_DIR_DEFAULT
    processed_dir: Path = PROCESSED_DIR_DEFAULT
    samples_dir: Path = SAMPLES_DIR_DEFAULT

    @property
    def ratings_csv(self) -> Path:
        return self.raw_dir / "ratings.csv"

    @property
    def tags_csv(self) -> Path:
        return self.raw_dir / "tags.csv"

    @property
    def movies_csv(self) -> Path:
        return self.raw_dir / "movies.csv"

    @property
    def links_csv(self) -> Path:
        return self.raw_dir / "links.csv"

    @property
    def genome_scores_csv(self) -> Path:
        return self.raw_dir / "genome-scores.csv"

    @property
    def genome_tags_csv(self) -> Path:
        return self.raw_dir / "genome-tags.csv"


RATINGS_SCHEMA = {
    "userId": pl.UInt32,
    "movieId": pl.UInt32,
    "rating": pl.Float32,
    "timestamp": pl.Int64,
}

TAGS_SCHEMA = {
    "userId": pl.UInt32,
    "movieId": pl.UInt32,
    "tag": pl.Utf8,
    "timestamp": pl.Int64,
}

MOVIES_SCHEMA = {
    "movieId": pl.UInt32,
    "title": pl.Utf8,
    "genres": pl.Utf8,
}

GENOME_SCORES_SCHEMA = {
    "movieId": pl.UInt32,
    "tagId": pl.UInt16,
    "relevance": pl.Float32,
}

GENOME_TAGS_SCHEMA = {
    "tagId": pl.UInt16,
    "tag": pl.Utf8,
}

LINKS_SCHEMA = {
    "movieId": pl.UInt32,
    "imdbId": pl.Utf8,
    "tmdbId": pl.Utf8,
}


def scan_ratings(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return (
        pl.scan_csv(paths.ratings_csv, schema=RATINGS_SCHEMA)
        .with_columns(
            pl.from_epoch("timestamp", time_unit="s").alias("rated_at"),
        )
        .with_columns(pl.col("rated_at").dt.year().cast(pl.Int16).alias("year"))
    )


def scan_tags(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return (
        pl.scan_csv(paths.tags_csv, schema=TAGS_SCHEMA)
        .with_columns(pl.from_epoch("timestamp", time_unit="s").alias("tagged_at"))
    )


def scan_movies(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return pl.scan_csv(paths.movies_csv, schema=MOVIES_SCHEMA)


def scan_genome_scores(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return pl.scan_csv(paths.genome_scores_csv, schema=GENOME_SCORES_SCHEMA)


def scan_genome_tags(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return pl.scan_csv(paths.genome_tags_csv, schema=GENOME_TAGS_SCHEMA)


def scan_links(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return pl.scan_csv(paths.links_csv, schema=LINKS_SCHEMA)


def convert_ratings_to_parquet_by_year(
    paths: DataPaths = DataPaths(),
    overwrite: bool = False,
) -> Path:
    out_dir = paths.processed_dir / "ratings_by_year"
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "_SUCCESS"
    if marker.exists() and not overwrite:
        return out_dir

    df = scan_ratings(paths).collect(streaming=True)
    for year, part in df.group_by("year"):
        year_val = year[0] if isinstance(year, tuple) else year
        part.drop("year").write_parquet(
            out_dir / f"year={int(year_val)}.parquet",
            compression="zstd",
        )
    marker.write_text("ok")
    return out_dir


def convert_all_to_parquet(
    paths: DataPaths = DataPaths(),
    overwrite: bool = False,
) -> dict[str, Path]:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "movies": (scan_movies, "movies.parquet"),
        "tags": (scan_tags, "tags.parquet"),
        "genome_tags": (scan_genome_tags, "genome_tags.parquet"),
        "genome_scores": (scan_genome_scores, "genome_scores.parquet"),
        "links": (scan_links, "links.parquet"),
    }

    out: dict[str, Path] = {}
    for name, (scan_fn, filename) in targets.items():
        dst = paths.processed_dir / filename
        if dst.exists() and not overwrite:
            out[name] = dst
            continue
        scan_fn(paths).sink_parquet(dst, compression="zstd")
        out[name] = dst

    out["ratings_by_year"] = convert_ratings_to_parquet_by_year(paths, overwrite)
    return out


def load_ratings_parquet(paths: DataPaths = DataPaths()) -> pl.LazyFrame:
    return pl.scan_parquet(paths.processed_dir / "ratings_by_year" / "year=*.parquet")


def stratified_user_sample(
    ratings: pl.LazyFrame,
    frac_users: float = 0.1,
    min_ratings: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """Stratified sample berbasis user: ambil `frac_users` dari user aktif."""
    active_users = (
        ratings.group_by("userId")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_ratings)
        .collect(streaming=True)
    )
    sampled = active_users.sample(fraction=frac_users, seed=seed).select("userId")
    return ratings.join(sampled.lazy(), on="userId", how="inner").collect(streaming=True)


if __name__ == "__main__":
    paths = DataPaths()
    result = convert_all_to_parquet(paths)
    for name, path in result.items():
        print(f"{name}: {path}")
