"""Builder notebook 06_Evaluation_Report.ipynb."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "06_Evaluation_Report.ipynb"


MD_INTRO = """\
# Notebook 06 - Evaluasi Komprehensif

Membandingkan 6 model utama (ALS, BPR, SVD, ItemKNN, ContentBased genome, Hybrid
ALS+Content) dengan set metrik lengkap:

- **Akurasi rating**: RMSE, MAE
- **Ranking**: Precision@10, Recall@10, NDCG@10, MAP@10, MRR
- **Beyond-accuracy**: Coverage, Diversity (rata-rata 1 - cosine antar film top-K
  pakai genome 1128d), Novelty (expected self-information log2 1/p), Serendipity
  (% item non-top-10%-populer yang tetap relevan)

Tambahan:
1. Segmentasi cold-start berdasarkan kuartil aktivitas user (train nnz).
2. Ablation: ALS factors, BPR learning_rate, hybrid alpha (0=pure content, 1=pure ALS).

Dijalankan via `python scripts/run_eval_comprehensive.py`.
"""

CELL_SETUP = """\
import sys, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

FIG = ROOT / "reports" / "figures"
main = pd.read_csv(FIG / "final_benchmark.csv")
cold = pd.read_csv(FIG / "coldstart_segments.csv")
abl = pd.read_csv(FIG / "ablation.csv")
main
"""

MD_MAIN = """\
## 1. Benchmark utama - 6 model, metrik lengkap

ALS masih pemenang NDCG@10 (0.049) dan juga unggul di MAP/MRR; Hybrid
(ALS 0.7 + Content 0.3) nyaris setara -> tanda genome belum banyak menambah
di evaluasi non-cold. ItemKNN menang mutlak di **Coverage** (0.10) dan
memiliki diversity lebih tinggi daripada ALS -> baik sebagai diversifier.
SVD bagus di RMSE (0.93) tapi sangat lemah di ranking: masalah klasik
"predict nilai baik, ranking top-K lemah".
"""

CELL_RADAR = """\
display(Image(filename=str(FIG / "29_radar_benchmark.png")))
"""

MD_COLD = """\
## 2. Cold-start - segmentasi user per kuartil aktivitas

Q1 = user paling sedikit train rating (paling mendekati kondisi cold user).
NDCG@10 per segment memperlihatkan bagaimana tiap model degrade saat data user
sedikit. Content-based relatif lebih tahan di Q1 karena hanya butuh profil
preferensi minimal; ALS perlu history user yang kuat.
"""

CELL_COLD = """\
display(Image(filename=str(FIG / "30_coldstart.png")))
cold.pivot(index="segment", columns="model", values="ndcg@k")
"""

MD_ABL = """\
## 3. Ablation - sweep hyperparameter

- **ALS factors**: kenaikan factor 16 -> 128 menaikkan NDCG tapi menurunkan
  coverage (model over-fit ke long-head).
- **BPR learning_rate**: sweet spot di 5e-3 - 1e-2; terlalu kecil = underfit,
  terlalu besar = ranking jatuh.
- **Hybrid alpha**: alpha=1 (pure ALS) vs alpha=0 (pure content) menegaskan
  genome konten sendirian tidak cukup di evaluasi non-cold; bobot 0.7-0.9 ALS
  ideal untuk kompromi.
"""

CELL_ABL = """\
display(Image(filename=str(FIG / "31_ablation.png")))
abl[['variant', 'ndcg@k', 'coverage', 'diversity', 'novelty']]
"""

MD_CONCL = """\
## 4. Kesimpulan Fase 6

- **ALS tetap anchor ranking** (NDCG 0.049, MAP 0.029); layak jadi retrieval
  utama di FastAPI.
- **ItemKNN** diposisikan sebagai diversifier (coverage 4x ALS).
- **Hybrid** dengan genome tidak mengalahkan ALS pada user aktif, tapi
  lebih defensif di segmen user minim data -> rekomendasi untuk cold-start tab
  di Streamlit.
- **SVD** dipensiunkan dari ranking (bagus di RMSE saja).
- Untuk Fase 7 (serving): expose 3 endpoint - `/recommend` pakai ALS,
  `/similar` pakai ItemKNN/ContentBased, `/cold_start` pakai Hybrid.
"""


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(MD_INTRO),
        nbf.v4.new_code_cell(CELL_SETUP),
        nbf.v4.new_markdown_cell(MD_MAIN),
        nbf.v4.new_code_cell(CELL_RADAR),
        nbf.v4.new_markdown_cell(MD_COLD),
        nbf.v4.new_code_cell(CELL_COLD),
        nbf.v4.new_markdown_cell(MD_ABL),
        nbf.v4.new_code_cell(CELL_ABL),
        nbf.v4.new_markdown_cell(MD_CONCL),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print("wrote", NB_PATH)


if __name__ == "__main__":
    build()
