# Cara Menjalankan App (Panduan untuk Pengguna Baru)

Repo ini berisi kode, notebook, dan dokumentasi. **Dataset mentah dan artefak model tidak ikut di-commit** (ukurannya gede: ~1.5 GB data + ratusan MB embedding/index). Jadi setelah `git clone` ada satu langkah tambahan untuk menyiapkan data sebelum app bisa jalan.

Pilih salah satu dari dua jalur di bawah.

---

## Persyaratan Awal (dua-duanya butuh ini)

- Python 3.11 (disarankan; 3.10–3.12 kemungkinan besar juga jalan)
- Git
- RAM minimal 8 GB (16 GB disarankan kalau mau re-train)
- OS: Windows / macOS / Linux

### Langkah 0 — Clone & setup environment

```bash
git clone https://github.com/slisanz/Portofolio-Data-Science.git
cd Portofolio-Data-Science/latihan2

# Buat virtual environment
python -m venv .venv

# Aktifkan venv
#   Windows PowerShell:
.venv\Scripts\Activate.ps1
#   Windows Git Bash:
source .venv/Scripts/activate
#   macOS / Linux:
source .venv/bin/activate

# Upgrade pip dulu (penting untuk Windows)
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

> **Catatan untuk Windows**: `scikit-surprise` sudah dikomentari di `requirements.txt` karena butuh Microsoft Visual C++ Build Tools untuk di-compile. Proyek ini sudah menggantinya dengan `scipy.sparse.linalg.svds`, jadi tidak perlu install Surprise. Kalau kamu punya MSVC dan ingin Surprise asli, uncomment baris itu di `requirements.txt`.
>
> **Catatan untuk FAISS di Windows**: pastikan yang ter-install adalah `faiss-cpu` (sudah di requirements). Jangan `pip install faiss` — itu paket lain yang rusak.
>
> **Catatan untuk PyTorch**: default `pip install torch` mengunduh versi CPU (~200 MB). Kalau kamu punya GPU NVIDIA dan ingin CUDA, install manual dari <https://pytorch.org/get-started/locally/> sebelum `pip install -r requirements.txt`.

---

## Jalur A — Pakai Docker (paling cepat, asal sudah punya artefak)

Kalau kamu punya file artefak (dari GitHub Release atau dikirim terpisah), taruh di path sesuai struktur lalu:

```bash
docker compose up --build
```

- API: <http://localhost:8000/docs>
- Streamlit: <http://localhost:8501>

Stop dengan `docker compose down`.

---

## Jalur B — Reproduksi dari Nol (jujur, makan waktu beberapa jam)

Kalau kamu tidak punya artefak, kamu bisa rebuild semuanya dari dataset mentah.

### Langkah 1 — Download dataset MovieLens

1. Kunjungi <https://grouplens.org/datasets/movielens/>
2. Download **ml-latest.zip** (~1.1 GB compressed, ~1.5 GB extracted) — versi Juli 2023
3. Ekstrak ke folder `latihan2/ml-latest/` sehingga strukturnya:
   ```
   latihan2/ml-latest/
   ├── ratings.csv           (891 MB, 33.8M rows)
   ├── tags.csv              (82 MB)
   ├── movies.csv            (4 MB)
   ├── links.csv
   ├── genome-scores.csv     (498 MB)
   └── genome-tags.csv
   ```

### Langkah 2 — Build artefak (pilih sesuai kebutuhan)

Semua perintah di bawah dijalankan dari folder `latihan2/` dengan venv aktif.

**Minimal — cukup untuk menjalankan Streamlit app:**

```bash
# 1. Konversi CSV → Parquet (~5 menit)
python -m src.data_loader

# 2. Feature engineering (~10–20 menit, butuh RAM)
python scripts/run_features.py

# 3. Training Deep Learning + export artefak Two-Tower (~30–60 menit di CPU)
python scripts/run_dl_bench.py

# 4. Pipeline NLP (semantic search + BERTopic, ~20–40 menit)
python scripts/run_nlp.py
```

**Lengkap (termasuk benchmark klasik & evaluasi komprehensif):**

```bash
python scripts/run_classical_bench.py     # RecSys klasik
python scripts/run_eval_comprehensive.py  # Evaluasi + cold-start + ablation
```

Alternatif menggunakan Makefile (macOS/Linux/Git Bash):

```bash
make eda
make features
make train
make nlp
make eval
```

### Langkah 3 — Jalankan app

**Streamlit dashboard (UI utama):**

```bash
streamlit run app/streamlit_app.py
```
Buka <http://localhost:8501>. Ada 4 tab: EDA Explorer, Recommender, Semantic Search, Model Arena.

**FastAPI (opsional, untuk endpoint REST):**

```bash
uvicorn src.serve.api:app --reload --port 8000
```
Buka <http://localhost:8000/docs> untuk OpenAPI UI.

Contoh request:
```bash
curl http://localhost:8000/recommend/1?k=10
curl "http://localhost:8000/semantic?q=dark%20psychological%20thriller&k=5"
```

---

## Struktur Artefak yang Harus Ada Sebelum App Jalan

Kalau kamu menerima artefak pre-built, ini file minimum yang harus ada:

```
latihan2/
├── data/processed/
│   ├── movies.parquet                        (1.4 MB)
│   ├── item_features.parquet                 (3.1 MB)
│   ├── dl_artifacts/                         (~28 MB total)
│   │   ├── two_tower_user.npy
│   │   ├── two_tower_item.npy
│   │   ├── user_ids.npy
│   │   ├── item_ids.npy
│   │   └── two_tower_faiss.index
│   └── nlp/                                  (~133 MB total)
│       ├── movie_text.faiss
│       ├── movie_text_ids.npy
│       └── movie_text_embeddings.npy
└── reports/figures/
    └── final_benchmark.csv                   (untuk tab Model Arena)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'polars'` / modul lain**
Venv belum aktif atau salah Python. Cek:
```bash
python -c "import sys; print(sys.executable)"
```
Path harus menunjuk ke `.venv/Scripts/python.exe` (Windows) atau `.venv/bin/python`. Kalau tidak, aktifkan venv dulu, atau pakai path eksplisit:
```bash
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

**`error: Microsoft Visual C++ 14.0 or greater is required` saat `pip install`**
Itu biasanya dari `scikit-surprise`. Pastikan baris itu masih dikomentari di `requirements.txt`. Kalau masih error, paket lain butuh build — laporkan di issue.

**`FileNotFoundError: .../two_tower_faiss.index`**
Artefak DL belum dibuat. Jalankan `python scripts/run_dl_bench.py`.

**`KeyError: 'year'` / kolom tidak ditemukan**
`item_features.parquet` belum di-regenerate setelah update kode. Jalankan ulang `python scripts/run_features.py`.

**Streamlit lambat / OOM**
Tutup tab lain, atau jalankan dengan sample kecil. RAM <8 GB akan kesulitan membuka `user_features.parquet` (331K user).

**FAISS gagal install di Windows**
Gunakan `pip install faiss-cpu` (bukan `faiss`). Sudah ada di `requirements.txt`.

---

## Test Cepat

Setelah setup selesai, verifikasi dengan:

```bash
pytest tests/                                 # 9 unit test
python scripts/smoke_api.py                   # latensi /recommend
```

---

## Ringkasan Alur

```
clone repo
  → setup venv + install requirements
  → download ml-latest dari GroupLens
  → jalankan 4 script builder (data_loader, features, dl_bench, nlp)
  → streamlit run app/streamlit_app.py
```

Perkiraan total waktu reproduksi dari nol: **1–3 jam** di laptop modern (CPU only), tergantung dataset IO dan CPU.

Untuk pertanyaan / issue: <https://github.com/slisanz/Portofolio-Data-Science/issues>
