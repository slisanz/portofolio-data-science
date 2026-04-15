"""Builder notebook 05_NLP_TagGenome.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05_NLP_TagGenome.ipynb"


MD_INTRO = """\
# Notebook 05 - NLP Tag Genome & Free Tags

Pipeline NLP end-to-end di atas dua sumber teks MovieLens:

1. **Free-text tags** (2.33 juta tag-rating mentah) -> pembersihan (lowercase,
   strip non-word, dedup per-user-movie-tag, lemmatization WordNet) ->
   per-movie tag document.
2. **Tag genome** (1128 dimensi relevance per film) -> UMAP 2D -> HDBSCAN
   cluster.

Analitik yang dihasilkan:
- Topic modeling `BERTopic` (sentence-transformers + UMAP + HDBSCAN internal) pada
  tag-document per film.
- Sentence embedding `all-MiniLM-L6-v2` -> FAISS untuk semantic search bebas query.
- WordCloud per genre, co-occurrence network top-80 tag.

Semua artifact tersimpan di `data/processed/nlp/`, figur di `reports/figures/24-28`.
Dijalankan ulang via `python scripts/run_nlp.py`.
"""

CELL_SETUP = """\
import json, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from IPython.display import Image, display

FIG = ROOT / "reports" / "figures"
NLP = ROOT / "data" / "processed" / "nlp"

summary = json.loads((NLP / "summary.json").read_text())
pd.DataFrame([summary]).T.rename(columns={0: "value"})
"""

MD_CLEAN = """\
## 1. Pembersihan tag & pembentukan dokumen per-film

`src/nlp.py::clean_tags` menormalisasi tag bebas: lowercase, hapus karakter
non-kata, collapse whitespace, dedup `(userId, movieId, tag)`, lemmatize
WordNet. Hasil: `tags_clean.parquet` + `movie_tag_docs.parquet` (satu baris
per film, kolom `doc` = gabungan tag yang muncul pada film tersebut).
"""

CELL_CLEAN = """\
tags = pl.read_parquet(NLP / "tags_clean.parquet")
docs = pl.read_parquet(NLP / "movie_tag_docs.parquet")
print("tags_clean:", tags.shape, " |  movie_tag_docs:", docs.shape)
docs.sort("total_tags", descending=True).head(5).to_pandas()
"""

CELL_TOPTAGS = """\
display(Image(filename=str(FIG / "24_top_tags.png")))
"""

MD_BERTOPIC = """\
## 2. Topic modeling - BERTopic

BERTopic di-fit pada 30K film dengan tag terbanyak. Vectorizer: CountVectorizer
bigram + stopwords English + `min_df=5`. Output: 94 topik semantik
(ex: *horror-zombie*, *marvel-superhero*, *anime-studio-ghibli*).
Assignment per film tersimpan di `bertopic_topics.parquet`.
"""

CELL_BERTOPIC = """\
info = pd.read_csv(NLP / "bertopic_info.csv")
print("Jumlah topik (tanpa noise):", (info['Topic'] >= 0).sum())
info[info['Topic'] >= 0].head(15)[['Topic', 'Count', 'Name']]
"""

CELL_BERTOPIC_FIG = """\
display(Image(filename=str(FIG / "25_bertopic_sizes.png")))
"""

MD_SEMANTIC = """\
## 3. Semantic search via sentence-transformers + FAISS

Setiap tag-document dienkode dengan `all-MiniLM-L6-v2` (384-dim, L2-normalized)
-> `IndexFlatIP`. Query natural-language dibandingkan via cosine similarity.
"""

CELL_SEMANTIC = """\
demo = pd.read_csv(NLP / "semantic_search_demo.csv")
for q in demo['query'].unique():
    print(f"\\n>> {q}")
    print(demo[demo['query'] == q][['rank', 'score', 'title']].to_string(index=False))
"""

MD_UMAP = """\
## 4. UMAP + HDBSCAN pada tag-genome (1128-dim)

Reduksi UMAP(cosine, n_neighbors=15) -> 2D, lalu HDBSCAN(min_cluster_size=40).
Catatan: genome embedding sudah cukup padat secara semantik sehingga HDBSCAN
pada default parameter hanya menemukan cluster besar (sisanya ter-labeli noise).
Untuk portfolio ini cluster utama terlihat jelas di scatter plot, tuning
parameter bisa ditambah di Fase 6 (ablation).
"""

CELL_UMAP = """\
clusters = pl.read_parquet(NLP / "genome_clusters.parquet")
print(clusters.group_by('cluster').agg(pl.len().alias('n')).sort('n', descending=True).to_pandas())
display(Image(filename=str(FIG / "26_umap_genome.png")))
"""

MD_WC = """\
## 5. WordCloud per genre & co-occurrence network

WordCloud dibangun dari tag bersih yang difilter per genre utama.
Network tag top-80 menampilkan edge jika dua tag pernah muncul bersama
pada >= 40 film (edge width proporsional terhadap co-occurrence count).
"""

CELL_WC = """\
display(Image(filename=str(FIG / "27_wordcloud_genre.png")))
display(Image(filename=str(FIG / "28_tag_cooccurrence.png")))
"""

MD_CONCLUSION = """\
## 6. Kesimpulan Fase 5

- **2.32M tag bebas** berhasil dibersihkan menjadi **40K tag-document** film.
- **BERTopic** menemukan **94 topik semantik** tanpa supervision, bisa dipakai
  sebagai fitur tambahan (topic id) untuk recommender hybrid di Fase 6.
- **Semantic search** demo menunjukkan retrieval relevan untuk query natural
  ("mind-bending time travel sci-fi", "dark psychological thriller", dst.).
  Bisa dipakai langsung di tab Streamlit Fase 8.
- **Genome UMAP** memperlihatkan manifold koheren; clustering halus menjadi
  agenda Fase 6 (ablation parameter HDBSCAN/min_cluster_size).
- Semua artefak (`tags_clean`, `movie_tag_docs`, `movie_text_embeddings.npy`,
  `movie_text.faiss`, `bertopic_topics`, `genome_clusters`) siap di-load oleh
  FastAPI & Streamlit di fase berikutnya.
"""


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(MD_INTRO),
        nbf.v4.new_code_cell(CELL_SETUP),
        nbf.v4.new_markdown_cell(MD_CLEAN),
        nbf.v4.new_code_cell(CELL_CLEAN),
        nbf.v4.new_code_cell(CELL_TOPTAGS),
        nbf.v4.new_markdown_cell(MD_BERTOPIC),
        nbf.v4.new_code_cell(CELL_BERTOPIC),
        nbf.v4.new_code_cell(CELL_BERTOPIC_FIG),
        nbf.v4.new_markdown_cell(MD_SEMANTIC),
        nbf.v4.new_code_cell(CELL_SEMANTIC),
        nbf.v4.new_markdown_cell(MD_UMAP),
        nbf.v4.new_code_cell(CELL_UMAP),
        nbf.v4.new_markdown_cell(MD_WC),
        nbf.v4.new_code_cell(CELL_WC),
        nbf.v4.new_markdown_cell(MD_CONCLUSION),
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
