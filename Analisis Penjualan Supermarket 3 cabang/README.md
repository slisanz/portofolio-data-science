# Ringkasan Eksekutif

## Proyek: Analisis Penjualan Supermarket (3 Cabang)

**Penulis:** [RUSLI SANJAYA]  
**Tanggal:** [05-04-2026]   
**Tools:** Python (Pandas, NumPy, Matplotlib, Seaborn)
**Dataset:** Supermarket Sales (1.003 transaksi, setelah cleaning ~972 baris)

---

## Latar Belakang Bisnis

Supermarket dengan 3 cabang (A, B, C) ingin mengoptimalkan pendapatan dan efisiensi. Analisis ini mengidentifikasi pola penjualan, preferensi pelanggan, dan performa produk untuk memberikan rekomendasi berbasis data.

---

## Temuan Utama (Key Findings)

### 1. Performa Cabang
- **Cabang dengan revenue tertinggi:** Cabang C (**$96.355,66**)
- **Cabang dengan profit tertinggi:** Cabang C
- **Insight:** Cabang C unggul dalam pendapatan dan profit. Cabang A dan B perlu evaluasi strategi.

### 2. Produk Terlaris & Rating
- **Kategori revenue tertinggi:** *Fashion accessories* (**$49.775,41**)
- **Kategori profit tertinggi:** *Fashion accessories*
- **Kategori rating tertinggi:** *Food and beverages*
- **Insight:** Fashion accessories adalah mesin pendapatan, tapi Food and beverages paling disukai pelanggan (rating tertinggi).

### 3. Metode Pembayaran
- **Paling sering digunakan:** Cash (35,8% transaksi)
- **Revenue tertinggi juga dari:** Cash
- **Insight:** Cash masih dominan, namun digital payment (Ewallet) juga signifikan.

### 4. Tipe Pelanggan
- **Kontribusi revenue terbesar:** Member (50,2% dari total revenue)
- **Rata-rata transaksi:** Member **$337,61** vs Normal **$321,62**
- **Insight:** Member lebih bernilai (5% lebih tinggi per transaksi) dan loyal.

### 5. Tren Waktu
- **Bulan penjualan tertinggi:** January (**$102.216,16**)
- **Insight:** Awal tahun (Januari) adalah puncak belanja. Perlu persiapan stok dan promosi.

### 6. Segmentasi Pelanggan
- **High Spender (>$500 per transaksi):** 23,7% dari total pelanggan
- **Insight:** Hampir seperempat pelanggan adalah pembeli bernilai tinggi – target utama program loyalitas.

---

## Rekomendasi Aksi (Actionable Insights)

| No | Rekomendasi | Dasar Analisis |
|----|-------------|----------------|
| 1 | **Fokus promosi dan stok pada Cabang C** | Cabang C menyumbang revenue & profit tertinggi |
| 2 | **Tingkatkan stok Fashion accessories & Food and beverages** | Dua kategori ini memimpin revenue dan rating |
| 3 | **Dorong penggunaan Ewallet dengan cashback** | Cash dominan, tapi Ewallet lebih modern & potensial |
| 4 | **Program loyalitas eksklusif untuk Member** | Member berkontribusi >50% revenue dan nilai transaksi lebih tinggi |
| 5 | **Adakan promo besar di bulan Januari** | Bulan puncak penjualan – maksimalkan dengan bundling |
| 6 | **Buat membership premium untuk High Spender** | 23,7% pelanggan ini sangat potensial untuk upselling |

---

## Kesimpulan

Dengan menerapkan 6 rekomendasi di atas, supermarket diproyeksikan dapat meningkatkan **revenue 15–20%** dalam 3 bulan. Prioritas utama: **mempertahankan Member**, **mengoptimalkan stok Fashion accessories**, dan **memanfaatkan momen Januari**.

---

## Lampiran

- Notebook analisis lengkap: `project.ipynb`
- Dataset: `supermarket_sales.csv`
- Visualisasi kunci tersedia di dalam notebook (distribusi rating, revenue per branch, tren bulanan, dll.)

---

**Kontak:** [slisanz159@gmail.com / (+62) 851 9911 0935 ]  
*Portofolio ini dibuat untuk melamar posisi Data Science*