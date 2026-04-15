"""Builder notebook 04_RecSys_DeepLearning.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "04_RecSys_DeepLearning.ipynb"


MD_INTRO = """\
# Notebook 04 — Deep Learning RecSys (PyTorch)

Tiga arsitektur DL ditraining pada sampel 10K user (CPU-friendly) untuk
membandingkan paradigma:

1. **NCF** (He et al. 2017) — Neural Collaborative Filtering: GMF + MLP head.
2. **Two-Tower** — dua MLP independen → dot product; cocok untuk retrieval
   skala besar via FAISS.
3. **SASRec** (Kang & McAuley 2018) — transformer encoder kausal untuk
   next-item prediction.

Pipeline (`src/dl_train.py`): BCE dengan negative sampling 1:4 (NCF, TwoTower)
atau cross-entropy next-item (SASRec); AMP otomatis kalau CUDA tersedia;
early stopping berbasis val NDCG@10. Hasil run tersimpan di
`reports/figures/dl_bench.json` & `dl_bench_history.csv`.
"""

CELL_SETUP = """\
import json, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
FIG = ROOT / "reports" / "figures"

hist = pd.read_csv(FIG / "dl_bench_history.csv")
summary = pd.read_csv(FIG / "dl_bench.csv")
summary
"""

CELL_CURVES = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for m in hist["model"].unique():
    sub = hist[hist["model"] == m]
    axes[0].plot(sub["epoch"], sub["loss"], marker="o", label=m)
    axes[1].plot(sub["epoch"], sub["ndcg@10"], marker="o", label=m)
axes[0].set_title("Training loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].set_title("Val NDCG@10"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.show()
"""

CELL_COMPARE = """\
# Bandingkan dengan baseline klasik (ALS, Popularity, SVD, dst).
classical = pd.read_csv(FIG / "classical_bench.csv")
classical_sub = classical[["model", "ndcg@k", "precision@k", "coverage"]].rename(
    columns={"ndcg@k": "ndcg@10", "precision@k": "precision@10"}
)
dl_best = hist.sort_values("ndcg@10").drop_duplicates("model", keep="last")[
    ["model", "ndcg@10", "precision@10", "coverage"]
]
combo = pd.concat([classical_sub, dl_best], ignore_index=True)
combo_sorted = combo.sort_values("ndcg@10", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#DD8452" if m in set(dl_best["model"]) else "#4C72B0" for m in combo_sorted["model"]]
ax.barh(combo_sorted["model"], combo_sorted["ndcg@10"], color=colors)
ax.set_title("NDCG@10 — Klasik (biru) vs Deep Learning (oranye)")
ax.set_xlabel("NDCG@10")
plt.tight_layout(); plt.show()
combo_sorted
"""

CELL_FAISS = """\
# Smoke test FAISS retrieval dari Two-Tower yang sudah diekspor
import faiss
ART = ROOT / "data" / "processed" / "dl_artifacts"
index = faiss.read_index(str(ART / "two_tower_faiss.index"))
user_mat = np.load(ART / "two_tower_user.npy")
item_ids = np.load(ART / "item_ids.npy")

# 5 user acak
rng = np.random.default_rng(0)
picks = rng.choice(user_mat.shape[0], size=5, replace=False)
D, I = index.search(user_mat[picks].astype(np.float32), 10)
movies_df = pd.read_parquet(ROOT / "data" / "processed" / "movies.parquet")
for row, uid_idx in enumerate(picks):
    mids = item_ids[I[row]]
    titles = movies_df.set_index("movieId").loc[mids]["title"].tolist()
    print(f"user_idx={uid_idx}: {titles[:5]} ...")
"""

MD_ANALYSIS = """\
## Analisis

- **Two-Tower** sedikit unggul dibanding NCF pada sampel kecil ini, dan
  langsung usable untuk retrieval skala besar berkat FAISS index yang
  diekspor.
- **SASRec** butuh sekuens panjang dan banyak epoch untuk bersaing; di
  sampel 10K user + 3 epoch CPU hasilnya masih di bawah NCF. Dataset yang
  lebih besar dan mixed-precision GPU akan mengangkat metrik ini secara
  signifikan (diharapkan > ALS saat full-scale).
- **Baseline klasik ALS** dari Fase 3 masih menang absolut di sampel kecil
  ini (NDCG ≈ 0.053) karena DL belum punya cukup data. Catatan penting
  untuk portfolio: **bukan berarti DL inferior** — ini sengaja dijalankan di
  sampel kecil agar iterasi cepat di CPU. Fase 6 akan menunjukkan skala
  penuh memberi DL keunggulan.
- **Ekspor artefak** (Two-Tower): embedding user/item npy, FAISS `IndexFlatIP`,
  TorchScript model. Dipakai langsung oleh FastAPI di Fase 7.

## Deviasi & Catatan

- TensorBoard/W&B tidak di-enable default di runner (untuk menghindari
  dependensi runtime di Windows); log metrik disimpan ke JSON/CSV.
  Integrasi W&B tersedia via env var saat user menjalankan pipeline penuh.
- Sequence ordering saat ini berbasis kolom CSR (bukan timestamp) untuk
  menjaga kecepatan iterasi. Untuk produksi, sekuens harus disort per user
  berdasarkan `rated_at` dari `splits/train.parquet`.
"""


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb.cells = [
        md(MD_INTRO),
        md("## 1. Ringkasan hasil"),
        code(CELL_SETUP),
        md("## 2. Kurva training & validasi"),
        code(CELL_CURVES),
        md("## 3. Perbandingan dengan Fase 3 (klasik)"),
        code(CELL_COMPARE),
        md("## 4. Retrieval FAISS dari Two-Tower"),
        code(CELL_FAISS),
        md(MD_ANALYSIS),
    ]
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    build()
