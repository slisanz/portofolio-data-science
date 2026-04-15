"""Builder notebook 03_RecSys_Classical.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03_RecSys_Classical.ipynb"


MD_INTRO = """\
# Notebook 03 - RecSys Klasik

Benchmark 11 model rekomendasi klasik di split yang identik (`splits/train.parquet`
/ `val.parquet` / `test.parquet`, leave-last-2-out per user) dengan sampel 50K
user agar iterasi cepat namun tetap representatif.

Model:
1. GlobalMean - predict mu saja.
2. UserMean, ItemMean - rata-rata baris/kolom.
3. Popularity - rank by jumlah interaksi.
4. ItemKNN(k=50) - cosine item-item.
5. UserKNN(k=50) - cosine user-user (fallback / gagal jika memori kurang).
6. SVD(k=64) - truncated SVD (scipy) pada matriks centered.
7. ALS(factors=64, it=15) - implicit ALS.
8. BPR(factors=64, it=40) - Bayesian Personalized Ranking (implicit binarisasi).
9. ContentBased(genome) - cosine di embedding tag-genome 1128-dim.
10. Hybrid(ALS*0.7 + Content*0.3) - z-score weighted blend.

Catatan: `scikit-surprise` tidak bisa dibuild di Windows; SVD diganti ke
`scipy.sparse.linalg.svds` pada residu (mu + bu + bi). UserKNN pada 50K user
butuh ~6 GB, di-skip jika OOM.
"""

CELL_SETUP = """\
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

FIG = ROOT / "reports" / "figures"
bench = pd.read_csv(FIG / "classical_bench.csv")
bench_sorted = bench.sort_values("ndcg@k", ascending=False).reset_index(drop=True)
bench_sorted
"""

CELL_BAR = """\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
order = bench_sorted.sort_values("ndcg@k").model
sns.barplot(data=bench_sorted, y="model", x="ndcg@k", order=order, ax=axes[0], color="#4C72B0")
axes[0].set_title("NDCG@10")
sns.barplot(data=bench_sorted, y="model", x="precision@k", order=order, ax=axes[1], color="#55A868")
axes[1].set_title("Precision@10")
sns.barplot(data=bench_sorted, y="model", x="coverage", order=order, ax=axes[2], color="#C44E52")
axes[2].set_title("Coverage")
plt.tight_layout()
plt.show()
"""

CELL_RMSE = """\
# RMSE/MAE hanya bermakna untuk model rating-prediction (baseline + SVD).
# ALS/BPR/Content tidak dioptimalkan untuk RMSE sehingga nilainya perlu
# dibaca berbeda (lihat analisis di bawah).
rmse_df = bench[bench["rmse"].notna()].sort_values("rmse")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=rmse_df, y="model", x="rmse", ax=ax, color="#DD8452")
ax.set_title("RMSE (lower = better) - hanya model rating-prediction relevan")
plt.tight_layout()
plt.show()
rmse_df
"""

CELL_TRADEOFF = """\
# Tradeoff NDCG vs Coverage: kita mau model yang relevan tapi tidak melulu
# mendorong item populer saja.
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=bench, x="coverage", y="ndcg@k", s=160, ax=ax)
for _, r in bench.iterrows():
    ax.annotate(r["model"], (r["coverage"], r["ndcg@k"]),
                xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.set_title("NDCG@10 vs Coverage - tradeoff accuracy vs diversity")
ax.set_xlabel("Coverage"); ax.set_ylabel("NDCG@10")
plt.tight_layout()
plt.show()
"""

CELL_FIT_TIME = """\
fig, ax = plt.subplots(figsize=(9, 4.5))
order = bench.sort_values("fit_sec").model
sns.barplot(data=bench, y="model", x="fit_sec", order=order, ax=ax, color="#937860")
ax.set_title("Waktu training per model (detik) - sample 50K user")
ax.set_xscale("log")
plt.tight_layout()
plt.show()
"""

MD_ANALYSIS = """\
## Analisis Singkat

- **ALS unggul di NDCG@10 dan Precision@10** padahal di-train hanya dengan
  sample 50K user, menandakan faktorisasi laten memang paling efisien untuk
  CF klasik pada sparsitas 99.88%.
- **Popularity** sangat kompetitif di NDCG meski tidak personal, menegaskan
  efek long-tail: banyak film populer memang relevan untuk mayoritas user.
  Ini juga yang membuat coverage Popularity sangat rendah - 2 item terpopuler
  memonopoli top-K.
- **ItemKNN** memberikan coverage tertinggi (~0.09) di antara model CF -
  cocok sebagai komponen diversity dalam ensemble.
- **SVD (scipy)** menang di RMSE (rating prediction) karena eksplisit
  memodelkan residu terhadap mu + bu + bi; namun ranking-nya lemah karena
  faktor laten tidak di-regularisasi untuk tugas ranking.
- **BPR** memberikan coverage jauh lebih tinggi dari ALS (0.053 vs 0.023)
  dengan NDCG yang setengahnya. Bagus sebagai diversifier.
- **ContentBased murni** underperform di ranking (cold-start OK, tapi di
  non-cold-start faktorisasi jauh lebih akurat). Dia memberi kontribusi saat
  item baru (cold item) - akan dievaluasi di Fase 6.
- **Hybrid (ALS 0.7 + Content 0.3)** praktis = ALS di non-cold-start; nilai
  hybrid sebenarnya muncul di skenario cold-item (akan diverifikasi kemudian).

## Keputusan untuk Fase 4

- Jadikan ALS sebagai baseline kuat yang harus dikalahkan oleh Deep Learning.
- Target: NDCG@10 > 0.08 dengan NCF / Two-Tower, coverage > ALS.
- Pertahankan ItemKNN & BPR sebagai anggota ensemble diversifier di evaluator
  komprehensif Fase 6.
"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb.cells = [
        md(MD_INTRO),
        md("## 1. Hasil benchmark"),
        code(CELL_SETUP),
        md("## 2. Ranking metrik per model"),
        code(CELL_BAR),
        md("## 3. RMSE (rating prediction)"),
        code(CELL_RMSE),
        md("## 4. Tradeoff NDCG vs Coverage"),
        code(CELL_TRADEOFF),
        md("## 5. Waktu training"),
        code(CELL_FIT_TIME),
        md(MD_ANALYSIS),
    ]
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    build()
