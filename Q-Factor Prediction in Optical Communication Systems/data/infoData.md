# Dataset — Q-Factor Prediction in Optical Communication Systems

## License

Distributed under **CC BY 4.0** by the original author. See [`LICENSE-DATA.md`](LICENSE-DATA.md)
for the required attribution and modification notes.

## File
`synthetic_qfactor_dataset.csv` — 1,000,000 baris, 6 kolom, semua numerik.

## Domain
Dalam sistem komunikasi serat optik, **Q-Factor** adalah ukuran kualitas sinyal digital yang berkaitan langsung dengan **Bit Error Rate (BER)**. Semakin tinggi Q-Factor (dalam dB), semakin baik kualitas sinyal yang diterima. Hubungan dasarnya:

```
BER ≈ 0.5 · erfc( Q / sqrt(2) )
Q[dB] ≈ 20 · log10(Q_linear)
Q_linear ≈ sqrt(2 · OSNR_linear)   (untuk modulasi sederhana, kondisi ideal)
```

Q-Factor dipengaruhi oleh banyak faktor degradasi sepanjang jalur transmisi: noise dari penguat optik (ASE), efek non-linear pada serat, dispersi kromatik, panjang link, dan daya peluncuran (launch power).

## Skema Kolom

| Kolom | Tipe | Range | Deskripsi singkat |
|-------|------|-------|-------------------|
| `OSNR` | float | [0,1] (normalized) | **Optical Signal-to-Noise Ratio** — rasio daya sinyal vs. daya noise ASE. Faktor dominan kualitas sinyal. |
| `Launch_Power` | float | [0,1] | Daya optik yang diluncurkan ke serat di sisi transmitter (dBm, ter-normalisasi). Trade-off: terlalu rendah → SNR jelek; terlalu tinggi → memicu efek non-linear. |
| `Fiber_Length` | float | [0,1] | Total panjang serat optik link (km, ter-normalisasi). Lebih panjang = akumulasi noise & dispersi lebih besar. |
| `Dispersion` | float | [0,1] | Dispersi kromatik akumulasi (ps/nm, ter-normalisasi). Menyebabkan pelebaran pulsa (inter-symbol interference). |
| `Nonlinear_Effect` | float | [0,1] | Magnitudo gabungan efek non-linear Kerr (SPM, XPM, FWM). Bertambah dengan daya & panjang. |
| `Q_Factor` | float | ~[2, 9] | **Target regresi**. Q-Factor ekuivalen sinyal yang diterima (dB). Nilai ≥ 6 dB umumnya dianggap kualitas memadai untuk transmisi. |

## Sifat Data
- Sintetis (semua fitur input ter-normalisasi ke [0,1] dengan distribusi mendekati uniform).
- Tidak ada missing values.
- Target `Q_Factor` kontinu, perlu dimodelkan sebagai **regresi**.
- Cocok untuk benchmark berbagai pendekatan: linear, tree-based, deep learning tabular, dan physics-informed models.

## Sumber Inspirasi Domain
- Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Wiley.
- Poggiolini et al., "The GN-Model of Fiber Non-Linear Propagation", *J. Lightwave Technology*, 2014.
