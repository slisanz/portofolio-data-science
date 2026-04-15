"""Jalankan seluruh kode EDA (cell notebook 01) sebagai skrip.

Alasan: pada environment Windows jupyter kernel sempat tidak stabil (zmq assertion
dan OOM saat nbconvert). Menjalankan kode yang sama sebagai skrip murni lebih
deterministik untuk menghasilkan figure. Setelah skrip sukses, notebook bisa
di-run ulang kemudian memakai hasil Parquet yang sudah cached.
"""

from __future__ import annotations

import importlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interaktif agar tidak buka window
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.data_loader import DataPaths, load_ratings_parquet  # noqa: E402

PATHS = DataPaths(
    raw_dir=ROOT / "ml-latest",
    processed_dir=ROOT / "data" / "processed",
    samples_dir=ROOT / "data" / "samples",
)


def step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def savefig(name: str) -> None:
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close("all")
    print(f"  saved -> {out.relative_to(ROOT)}", flush=True)


def main() -> None:
    step("Load ratings parquet (lazy)")
    ratings_lf = load_ratings_parquet(PATHS)
    movies = pl.read_parquet(PATHS.processed_dir / "movies.parquet")
    tags = pl.read_parquet(PATHS.processed_dir / "tags.parquet")
    genome_scores_lf = pl.scan_parquet(PATHS.processed_dir / "genome_scores.parquet")
    genome_tags = pl.read_parquet(PATHS.processed_dir / "genome_tags.parquet")

    step("Materialize ratings to memory (UInt32/Float32)")
    ratings = ratings_lf.select("userId", "movieId", "rating", "rated_at").collect()
    print(f"  ratings in-memory: {ratings.estimated_size('mb'):.0f} MB, shape={ratings.shape}", flush=True)

    n_ratings = ratings.height
    n_users = ratings.select(pl.col("userId").n_unique()).item()
    n_movies_rated = ratings.select(pl.col("movieId").n_unique()).item()
    year_min = ratings.select(pl.col("rated_at").dt.year().min()).item()
    year_max = ratings.select(pl.col("rated_at").dt.year().max()).item()
    density = n_ratings / (n_users * n_movies_rated)

    step("Ringkasan dataset")
    print(f"  Ratings={n_ratings:,}  Users={n_users:,}  MoviesRated={n_movies_rated:,}  "
          f"Catalog={movies.height:,}  Tags={tags.height:,}  GenomeTags={genome_tags.height:,}", flush=True)
    print(f"  Year {year_min}..{year_max}  density={density:.3e}  sparsity={1-density:.8f}", flush=True)

    # ------------------------------------------------------------------ 1
    step("Fig 01 - distribusi rating")
    rating_dist = (
        ratings.group_by("rating").agg(pl.len().alias("count")).sort("rating").to_pandas()
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=rating_dist, x="rating", y="count", ax=ax, color="#4C72B0")
    ax.set_title("Distribusi nilai rating"); ax.set_ylabel("Jumlah rating"); ax.set_xlabel("Rating")
    savefig("01_rating_distribution.png")

    # ------------------------------------------------------------------ 2
    step("Fig 02 - long-tail user & film")
    user_counts = ratings.group_by("userId").agg(pl.len().alias("n"))["n"].to_numpy()
    item_counts = ratings.group_by("movieId").agg(pl.len().alias("n"))["n"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, counts, lbl, color in [
        (axes[0], user_counts, "user", "#4C72B0"),
        (axes[1], item_counts, "movie", "#C44E52"),
    ]:
        cs = np.sort(counts)[::-1]
        ax.loglog(np.arange(1, len(cs) + 1), cs, color=color, lw=1.2)
        ax.set_xlabel(f"Rank {lbl} (log)"); ax.set_ylabel("Jumlah rating (log)")
        ax.set_title(f"Long-tail {lbl}")
    savefig("02_longtail_user_item.png")

    # ------------------------------------------------------------------ 3
    step("Fig 03 - temporal")
    yearly = (
        ratings.group_by(pl.col("rated_at").dt.year().alias("year"))
        .agg(pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating"))
        .sort("year").to_pandas()
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].bar(yearly["year"], yearly["n"], color="#4C72B0")
    axes[0].set_ylabel("Jumlah rating"); axes[0].set_yscale("log")
    axes[0].set_title("Aktivitas rating per tahun (1995-2023)")
    axes[1].plot(yearly["year"], yearly["mean_rating"], marker="o", color="#C44E52")
    axes[1].set_ylabel("Mean rating"); axes[1].set_xlabel("Tahun")
    axes[1].set_title("Evolusi rata-rata rating per tahun")
    savefig("03_temporal_activity.png")

    # ------------------------------------------------------------------ 4
    step("Fig 04 - bias heatmap user & item")
    user_stats = ratings.group_by("userId").agg(
        pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating")
    )
    item_stats = ratings.group_by("movieId").agg(
        pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    u_logn = np.log10(user_stats["n"].to_numpy())
    u_mean = user_stats["mean_rating"].to_numpy()
    h, xe, ye = np.histogram2d(u_logn, u_mean, bins=[40, 40], range=[[0, u_logn.max()], [0.5, 5]])
    axes[0].imshow(h.T, origin="lower", aspect="auto",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis", norm="log")
    axes[0].set_xlabel("log10(jumlah rating user)"); axes[0].set_ylabel("Mean rating user")
    axes[0].set_title("Bias user: aktivitas vs rata-rata rating")

    i_logn = np.log10(item_stats["n"].to_numpy())
    i_mean = item_stats["mean_rating"].to_numpy()
    h2, xe2, ye2 = np.histogram2d(i_logn, i_mean, bins=[40, 40], range=[[0, i_logn.max()], [0.5, 5]])
    axes[1].imshow(h2.T, origin="lower", aspect="auto",
                   extent=[xe2[0], xe2[-1], ye2[0], ye2[-1]], cmap="magma", norm="log")
    axes[1].set_xlabel("log10(jumlah rating film)"); axes[1].set_ylabel("Mean rating film")
    axes[1].set_title("Bias item: popularitas vs rata-rata rating")
    savefig("04_bias_heatmaps.png")

    # ------------------------------------------------------------------ 5
    step("Fig 05 - genre popularity & mean")
    movies_genre = (
        movies.with_columns(pl.col("genres").str.split("|").alias("genre_list"))
        .explode("genre_list")
        .rename({"genre_list": "genre"})
        .filter(pl.col("genre") != "(no genres listed)")
    )
    ratings_with_genre = ratings.join(movies_genre.select("movieId", "genre"), on="movieId")
    genre_agg = (
        ratings_with_genre.group_by("genre")
        .agg(pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating"))
        .sort("n", descending=True).to_pandas()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=genre_agg, y="genre", x="n", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Popularitas genre (jumlah rating)"); axes[0].set_xscale("log")
    sns.barplot(data=genre_agg.sort_values("mean_rating", ascending=False),
                y="genre", x="mean_rating", ax=axes[1], color="#55A868")
    axes[1].set_title("Rata-rata rating per genre"); axes[1].set_xlim(3, 4.2)
    savefig("05_genre_popularity_mean.png")

    # ------------------------------------------------------------------ 6
    step("Fig 06 - genre x dekade heatmap")
    year_pat = re.compile(r"\((\d{4})\)")
    release_years = movies["title"].to_list()
    years_arr = np.array([
        int(m.group(1)) if (m := year_pat.search(t or "")) else -1
        for t in release_years
    ], dtype=np.int32)
    movies_year = movies.with_columns(
        pl.Series("release_year", years_arr)
    ).filter(pl.col("release_year") > 0)
    movies_year = movies_year.with_columns(
        ((pl.col("release_year") // 10) * 10).alias("decade")
    )

    # Hindari double-join atas 33M rows: reduksi dulu ke level movieId, baru merge
    # dengan metadata dekade + genre.
    movie_agg = (
        ratings.group_by("movieId")
        .agg(
            pl.len().alias("n"),
            pl.col("rating").sum().alias("rsum"),
        )
    )
    movie_agg = movie_agg.join(movies_year.select("movieId", "decade"), on="movieId")
    movie_agg = movie_agg.join(movies_genre.select("movieId", "genre"), on="movieId")
    decade_genre = (
        movie_agg.group_by(["decade", "genre"])
        .agg(pl.col("rsum").sum().alias("rsum"), pl.col("n").sum().alias("n"))
        .with_columns((pl.col("rsum") / pl.col("n")).alias("mean_rating"))
        .to_pandas()
    )
    top_genres = genre_agg["genre"].head(10).tolist()
    pivot = (
        decade_genre[decade_genre["genre"].isin(top_genres)]
        .pivot_table(index="decade", columns="genre", values="mean_rating")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot.T, annot=True, fmt=".2f", cmap="RdYlGn", vmin=3.0, vmax=4.2, ax=ax)
    ax.set_title("Mean rating per genre x dekade rilis film")
    savefig("06_genre_decade_heatmap.png")

    # ------------------------------------------------------------------ 7
    step("Fig 07 - cold-start histograms")
    pct_user_lt5 = (user_stats["n"] < 5).sum() / user_stats.height * 100
    pct_user_lt10 = (user_stats["n"] < 10).sum() / user_stats.height * 100
    pct_item_lt5 = (item_stats["n"] < 5).sum() / item_stats.height * 100
    pct_item_lt10 = (item_stats["n"] < 10).sum() / item_stats.height * 100
    pct_item_0 = (movies.height - item_stats.height) / movies.height * 100
    print(f"  user<5={pct_user_lt5:.2f}% user<10={pct_user_lt10:.2f}% "
          f"film<5={pct_item_lt5:.2f}% film<10={pct_item_lt10:.2f}% film_norating={pct_item_0:.2f}%", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.clip(user_stats["n"].to_numpy(), 1, 1000), bins=80, color="#4C72B0")
    axes[0].set_title("Histogram jumlah rating per user (clip 1000)")
    axes[0].set_xlabel("Jumlah rating"); axes[0].set_ylabel("User")
    axes[1].hist(np.clip(item_stats["n"].to_numpy(), 1, 5000), bins=80, color="#C44E52")
    axes[1].set_title("Histogram jumlah rating per film (clip 5000)")
    axes[1].set_xlabel("Jumlah rating"); axes[1].set_ylabel("Film")
    savefig("07_coldstart_histograms.png")

    # ------------------------------------------------------------------ 8
    step("Fig 08 - density vs sparsity")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Density", "Sparsity"]
    values = [density, 1 - density]
    ax.bar(labels, values, color=["#55A868", "#C44E52"]); ax.set_yscale("log")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3e}", ha="center", va="bottom")
    ax.set_title("Density vs sparsity interaction matrix")
    savefig("08_density_sparsity.png")

    # ------------------------------------------------------------------ 9 heavy vs casual
    step("Fig 09 - heavy vs casual users")
    heavy_cut = np.percentile(user_stats["n"].to_numpy(), 95)
    casual_cut = 20
    heavy_mean = user_stats.filter(pl.col("n") >= heavy_cut)["mean_rating"].to_numpy()
    casual_mean = user_stats.filter(pl.col("n") <= casual_cut)["mean_rating"].to_numpy()
    print(f"  heavy (>=p95={int(heavy_cut)}): {len(heavy_mean):,} users, mean={heavy_mean.mean():.3f}; "
          f"casual (<={casual_cut}): {len(casual_mean):,} users, mean={casual_mean.mean():.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(heavy_mean, ax=ax, label=f"Heavy (>=p95={int(heavy_cut)})",
                color="#4C72B0", fill=True, alpha=0.3)
    sns.kdeplot(casual_mean, ax=ax, label=f"Casual (<={casual_cut})",
                color="#C44E52", fill=True, alpha=0.3)
    ax.set_title("Distribusi mean rating: heavy vs casual user")
    ax.set_xlabel("Mean rating user"); ax.legend()
    savefig("09_heavy_vs_casual.png")

    # ------------------------------------------------------------------ 10 seasonality
    step("Fig 10 - seasonality (bulan & hari)")
    monthly = (
        ratings.group_by(pl.col("rated_at").dt.month().alias("month"))
        .agg(pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating"))
        .sort("month").to_pandas()
    )
    dow = (
        ratings.group_by(pl.col("rated_at").dt.weekday().alias("dow"))
        .agg(pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating"))
        .sort("dow").to_pandas()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(monthly["month"], monthly["n"], color="#4C72B0")
    axes[0].set_title("Jumlah rating per bulan"); axes[0].set_xlabel("Bulan")
    axes[1].bar(dow["dow"], dow["n"], color="#55A868")
    axes[1].set_title("Jumlah rating per hari dalam minggu")
    axes[1].set_xlabel("Hari (1=Senin .. 7=Minggu)")
    savefig("10_seasonality.png")

    # ------------------------------------------------------------------ 11 top free tags
    step("Fig 11 - top free tags")
    top_tags_free = (
        tags.group_by(pl.col("tag").str.to_lowercase())
        .agg(pl.len().alias("n"))
        .sort("n", descending=True).head(25).to_pandas()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top_tags_free, y="tag", x="n", ax=ax, color="#8172B2")
    ax.set_title("Top 25 free-text tag (lowercase)")
    savefig("11_top_free_tags.png")

    # ------------------------------------------------------------------ 12 genome relevance
    step("Fig 12 - genome relevance sampled")
    rel_sample = genome_scores_lf.select("relevance").collect().sample(
        n=min(500_000, 10_000_000), seed=42
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rel_sample["relevance"].to_numpy(), bins=60, color="#937860")
    ax.set_title("Distribusi relevance tag-genome (sampled 500k)")
    ax.set_xlabel("relevance"); ax.set_ylabel("freq")
    savefig("12_genome_relevance_dist.png")

    # ------------------------------------------------------------------ 13 movie age
    step("Fig 13 - age at rating")
    # Alih-alih join dulu (membengkakkan 33M rows), hitung age inline lalu agregasi.
    age_df = (
        ratings.lazy()
        .join(movies_year.lazy().select("movieId", "release_year"), on="movieId")
        .with_columns((pl.col("rated_at").dt.year() - pl.col("release_year")).alias("age"))
        .filter((pl.col("age") >= 0) & (pl.col("age") <= 100))
        .group_by("age")
        .agg(pl.len().alias("n"), pl.col("rating").mean().alias("mean_rating"))
        .sort("age")
        .collect(engine="streaming")
        .to_pandas()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(age_df["age"], age_df["n"], color="#4C72B0")
    axes[0].set_title("Jumlah rating vs usia film saat di-rating")
    axes[0].set_xlabel("Usia film (th)"); axes[0].set_yscale("log")
    axes[1].plot(age_df["age"], age_df["mean_rating"], color="#C44E52")
    axes[1].set_title("Mean rating vs usia film saat di-rating")
    axes[1].set_xlabel("Usia film (th)")
    savefig("13_movie_age_at_rating.png")

    # ------------------------------------------------------------------ 14 n genre / film
    step("Fig 14 - jumlah genre per film")
    n_genre = movies.with_columns(
        pl.col("genres").str.split("|").list.len().alias("n_genre")
    ).to_pandas()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=n_genre, x="n_genre", ax=ax, color="#4C72B0")
    ax.set_title("Jumlah genre per film")
    savefig("14_n_genre_per_movie.png")

    # ------------------------------------------------------------------ 15 summary
    step("Fig 15 - ringkasan dataset")
    genome_rows = int(genome_scores_lf.select(pl.len()).collect().item())
    summary = {
        "Ratings": n_ratings,
        "Users": n_users,
        "Movies rated": n_movies_rated,
        "Movies catalog": movies.height,
        "Free tags": tags.height,
        "Genome rows": genome_rows,
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(list(summary.keys()), list(summary.values()),
                   color=sns.color_palette("viridis", len(summary)))
    ax.set_xscale("log")
    for bar, v in zip(bars, summary.values()):
        ax.text(v, bar.get_y() + bar.get_height()/2, f"  {v:,}", va="center")
    ax.set_title("Ringkasan ukuran dataset MovieLens ml-latest")
    savefig("15_dataset_summary.png")

    step("Underrated high-volume films (mean>=4.3, n>=p90)")
    p90_n = np.percentile(item_stats["n"].to_numpy(), 90)
    underrated = (
        item_stats.filter((pl.col("n") >= p90_n) & (pl.col("mean_rating") >= 4.3))
        .sort("mean_rating", descending=True).head(15)
        .join(movies, on="movieId", how="left")
        .select("title", "genres", "n", "mean_rating")
    )
    underrated.write_csv(ROOT / "reports" / "figures" / "_underrated_highvolume.csv")
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(underrated, flush=True)

    step("DONE")


if __name__ == "__main__":
    main()
