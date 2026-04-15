"""Render portfolio_report.pdf (reportlab) — kompak 12-15 halaman."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "portfolio_report.pdf"

styles = getSampleStyleSheet()
H1 = styles["Heading1"]
H2 = styles["Heading2"]
BODY = ParagraphStyle("body", parent=styles["BodyText"], leading=14, spaceAfter=6)

STORY = []


def h1(t): STORY.append(Paragraph(t, H1)); STORY.append(Spacer(1, 0.2 * cm))
def h2(t): STORY.append(Paragraph(t, H2)); STORY.append(Spacer(1, 0.15 * cm))
def p(t): STORY.append(Paragraph(t, BODY))
def br(h=0.3): STORY.append(Spacer(1, h * cm))


def img(name: str, width_cm: float = 15.0):
    path = FIG / name
    if not path.exists():
        return
    im = Image(str(path))
    w = width_cm * cm
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth = w
    im.drawHeight = w * ratio
    STORY.append(im)
    br(0.2)


def table_from_csv(path: Path, cols: list[str], float_fmt: str = "{:.4f}", head: int | None = None):
    df = pd.read_csv(path)
    df = df[cols].copy()
    if head:
        df = df.head(head)
    for c in cols[1:]:
        df[c] = df[c].apply(lambda v: float_fmt.format(v) if pd.notnull(v) else "-")
    data = [cols] + df.values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f3f3")]),
            ]
        )
    )
    STORY.append(t)
    br()


# === Cover ===
h1("MovieLens RecSys — Portfolio Data Science")
p("<b>Dataset:</b> ml-latest (Juli 2023) — 33.832.162 rating, 331K user, 83K film, 14M tag, genome 1128-dim.")
p("<b>Scope:</b> end-to-end pipeline: data engineering, EDA, feature store, 11 model klasik, 3 arsitektur deep learning, NLP semantic search, evaluasi komprehensif, serving FastAPI + Docker, dashboard Streamlit.")
p("<b>Ringkasan hasil:</b> SVD menang RMSE=0.929 • ALS menang NDCG@10=0.0495 • ItemKNN menang Coverage=0.100 • API latensi p50=5.3ms.")
img("15_dataset_summary.png", 15)
STORY.append(PageBreak())

# === 1 EDA ===
h1("1. Exploratory Data Analysis")
p("Distribusi rating miring ke kanan (modus=4.0). Long-tail sangat tajam di kedua sisi: 47% film memiliki <5 rating, 7% user kurang dari 5 rating. Sparsity matriks user-item = 99.877% (density 1.23e-03).")
img("01_rating_distribution.png")
img("02_longtail_user_item.png")
STORY.append(PageBreak())

p("Evolusi temporal (1995-2023) menunjukkan puncak aktivitas 2015-2020. Mean rating cenderung naik perlahan — kemungkinan efek self-selection (user aktif memilih film yang diduga disukai).")
img("03_temporal_activity.png")
img("10_seasonality.png")
STORY.append(PageBreak())

h2("Cold-start & sparsity")
p("Cold-item dominan: 47% film <5 rating, 61.5% film <10 rating. Ini menjustifikasi jalur content-based (tag-genome) dan semantic search sebagai fallback.")
img("07_coldstart_histograms.png")
img("08_density_sparsity.png")
STORY.append(PageBreak())

# === 2 Feature Engineering ===
h1("2. Feature Engineering")
p("Feature store dipecah: user (331K x 46), item (83K x 30), genome (1128-dim per film), plus split leave-last-2-out per user dengan fitur temporal inline (hour, dow, movie_age).")
img("16_user_feature_overview.png")
img("18_item_popularity_bucket.png")
STORY.append(PageBreak())

# === 3 Classical ===
h1("3. RecSys Klasik — Benchmark")
p("11 model di-benchmark pada split yang sama. ALS memenangkan ranking accuracy; SVD (scipy svds pada residu mu+bu+bi) memenangkan RMSE; ItemKNN memenangkan coverage katalog.")
table_from_csv(
    FIG / "classical_bench.csv",
    ["model", "rmse", "ndcg@k", "coverage"],
    head=12,
)
img("22_classical_bench_bar.png")
STORY.append(PageBreak())

# === 4 DL ===
h1("4. Deep Learning RecSys")
p("Tiga arsitektur: NCF (GMF+MLP NeuMF), Two-Tower (dengan FAISS IndexFlatIP retrieval), SASRec (transformer kausal). Two-Tower menang NDCG@10=0.029 pada sampel 10K user CPU dan diekspor lengkap: TorchScript + npy embedding + FAISS index (siap produksi).")
img("23_dl_training_curves.png")
STORY.append(PageBreak())

# === 5 NLP ===
h1("5. NLP — Tag Genome & Free Tags")
p("2.32M tag dibersihkan (lowercase + lemmatize). BERTopic menemukan 94 topik semantik. Sentence-transformer MiniLM 384-dim + FAISS untuk semantic search natural-language (30K film). UMAP + HDBSCAN pada genome 1128-dim memperlihatkan klaster genre yang koheren.")
img("25_bertopic_sizes.png")
img("26_umap_genome.png")
STORY.append(PageBreak())

img("27_wordcloud_genre.png")
img("28_tag_cooccurrence.png")
STORY.append(PageBreak())

# === 6 Eval ===
h1("6. Evaluasi Komprehensif")
p("Metrik lengkap: accuracy (RMSE, MAE), ranking (Precision/Recall/NDCG/MAP@10, MRR), beyond-accuracy (Coverage, Diversity, Novelty, Serendipity).")
table_from_csv(
    FIG / "final_benchmark.csv",
    ["model", "ndcg@k", "recall@k", "coverage", "diversity", "novelty"],
)
img("29_radar_benchmark.png")
STORY.append(PageBreak())

h2("Cold-start segmentation & ablation")
p("Segmentasi user by train-nnz kuartil (Q1..Q4): content-based relatif tahan di Q1 dibanding ALS. Ablation: ALS factors in {16,32,64,128}, BPR learning rate 1e-3..5e-2, Hybrid alpha 0..1.")
img("30_coldstart.png")
img("31_ablation.png")
STORY.append(PageBreak())

# === 7 Serving ===
h1("7. Serving & MLOps")
p("<b>FastAPI</b> mengekspos 5 endpoint: <code>/health</code>, <code>/recommend/{user_id}</code>, <code>/similar/{movie_id}</code>, <code>/cold_start</code>, <code>/semantic</code>. Backend retrieval: Two-Tower FAISS IndexFlatIP; cold-start pakai mean embedding film yang disukai sebagai pseudo-user; semantic search pakai FAISS teks MiniLM.")
p("<b>Latensi</b> lokal (TestClient, 30 requests): p50 = 5.3 ms, p95 = 6.4 ms — jauh di bawah target 200 ms.")
p("<b>MLOps:</b> multi-stage Dockerfile, docker-compose (API + Streamlit), GitHub Actions CI (ruff + pytest + docker build), 9 unit test hijau (loader, evaluator, endpoint).")

# === 8 Dashboard ===
h1("8. Streamlit Dashboard")
p("Empat tab interaktif: <b>EDA Explorer</b> (filter tahun/genre + scatter Plotly), <b>Recommender</b> (mode User ID + mode cold-start mean-embedding), <b>Semantic Search</b> (FAISS teks), <b>Model Arena</b> (tabel benchmark + radar plotly normalisasi 0-1).")

# === 9 Insight & Rekomendasi ===
h1("9. Insight Bisnis & Rekomendasi Produksi")
p("1. <b>Long-tail item</b> adalah bottleneck utama (47% film <5 rating). Sistem produksi wajib punya jalur content-based/semantic sebagai fallback.")
p("2. <b>Heavy vs casual</b> user punya bias rating berbeda (3.40 vs 3.68). Normalisasi per-user (bu) terbukti menurunkan RMSE (SVD menang).")
p("3. <b>Ensemble</b> ALS (accuracy) + BPR atau ItemKNN (coverage) direkomendasikan — ranking kuat tanpa kehilangan diversity.")
p("4. <b>Two-Tower + FAISS</b> memberi candidate generation sub-10ms, siap dipakai untuk re-ranker Transformer di lapisan kedua.")
p("5. <b>Semantic search</b> pada tag + SBERT MiniLM memberi UX natural-language yang kuat, bagus sebagai onboarding cold-start (user baru mengetik preferensi).")

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=1.8 * cm,
    rightMargin=1.8 * cm,
    topMargin=1.5 * cm,
    bottomMargin=1.5 * cm,
    title="MovieLens RecSys Portfolio",
)
doc.build(STORY)
print(f"OK -> {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
