===================================================
 PORTOFOLIO DATA SCIENCE - ANALISIS SUPERMARKET
===================================================

Nama Proyek   : Analisis Penjualan Supermarket (3 Cabang)
Penulis       : [Nama Kamu]
Kontak        : [Email / LinkedIn / GitHub]
Tanggal       : [Tanggal]

---------------------------------------------------
 DESKRIPSI SINGKAT
---------------------------------------------------
Proyek ini menganalisis data transaksi supermarket dengan 1.003 baris data (setelah cleaning menjadi ~972 baris). 
Tujuan: mengidentifikasi pola penjualan, performa cabang, produk terlaris, preferensi pelanggan, dan memberikan rekomendasi bisnis berbasis data.

Tools yang digunakan:
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Jupyter Notebook

---------------------------------------------------
 STRUKTUR FOLDER
---------------------------------------------------
PORTOFOLIO DS3/
│
├── .venv/                         # Virtual environment (Python)
├── project.ipynb                  # Notebook utama analisis
├── supermarket_sales.csv          # Dataset mentah
└── README.txt                     # File ini

---------------------------------------------------
 CARA MENJALANKAN
---------------------------------------------------
1. Pastikan Python 3.8+ sudah terinstal.
2. Buat environment dan aktifkan virtual environment:
   - Windows: .venv\Scripts\activate
   - Mac/Linux: source .venv/bin/activate
3. Install library yang dibutuhkan (bisa dijalankan di dalam notebook):
   !python -m pip install pandas numpy matplotlib seaborn scikit-learn
4. Buka Jupyter Notebook:
   jupyter notebook project.ipynb
5. Jalankan semua cell dari atas ke bawah.

Atau cukup buka file project.ipynb di environment yang sudah memiliki library tersebut.

---------------------------------------------------
 HASIL UTAMA & INSIGHT
---------------------------------------------------
Berdasarkan analisis yang dilakukan, diperoleh temuan berikut:

1. Cabang terbaik: Cabang C dengan revenue $96.355,66 dan profit tertinggi.
2. Produk terlaris: Fashion accessories ($49.775,41), disusul Food & beverages.
3. Metode pembayaran paling sering: Cash (35,8% transaksi).
4. Tipe pelanggan paling berkontribusi: Member (50,2% dari total revenue).
   Rata-rata transaksi Member $337,61 vs Normal $321,62.
5. Bulan puncak penjualan: Januari ($102.216,16).
6. Segmentasi: 23,7% pelanggan termasuk High Spender (>$500 per transaksi).

Rekomendasi bisnis:
- Fokus promosi pada Cabang C dan kategori Fashion accessories.
- Tingkatkan program loyalitas untuk Member dan High Spender.
- Optimalisasi stok dan promo pada bulan Januari.
- Dorong penggunaan Ewallet dengan cashback.

---------------------------------------------------
 KESIMPULAN
---------------------------------------------------
Portofolio ini menunjukkan kemampuan end-to-end dalam data cleaning, eksplorasi data, visualisasi, segmentasi pelanggan, dan penyusunan rekomendasi bisnis. Proyek ini siap digunakan sebagai bahan lamaran untuk posisi Data Science / Business Analyst internship.

---------------------------------------------------
 CATATAN
---------------------------------------------------
Dataset berasal dari publik (Kaggle/Supermarket Sales). Seluruh kode ditulis sendiri dalam bahasa Python dengan panduan best practice.

Terima kasih telah membaca.