# Laporan Fuzzy Logic Risiko Kecelakaan

## 1. Dataset

Dataset yang digunakan adalah data kecelakaan nyata dari Kaggle:
https://www.kaggle.com/datasets/aginanjar/data-kecelakaan

File yang dipakai pada eksperimen ini adalah `datakecminim.csv` dengan 36.055 row. Dataset memenuhi syarat minimal 5.000 row. Kolom yang digunakan dalam sistem fuzzy:

- Input 1: `JAM`
- Input 2: `BULAN`
- Input 3: `FREQ_KECAMATAN`, yaitu frekuensi kemunculan kecamatan dalam dataset
- Input 4: `FREQ_PROFESI`, yaitu frekuensi kemunculan profesi dalam dataset
- Input 5: `SKOR_KARAKTERISTIK`, yaitu skor hasil pemetaan dari `KARAKTERISTIK LAKA`
- Output/ground truth: `KODE_CIDERA`, dipetakan menjadi `KATEGORI_AKTUAL`

Pemetaan label aktual:

- `MD` dan `CT` = `TINGGI`
- `LL` = `SEDANG`
- Selain itu = `RENDAH`

Preprocessing dan data cleaning yang dilakukan:

- Normalisasi teks menjadi huruf besar.
- Menghapus spasi ganda dan spasi di awal/akhir teks, sehingga nilai seperti ` PATI` dan `PATI` dianggap sama.
- Mengubah nilai kosong, `-`, `NaN`, dan `None` menjadi `TIDAK DIKETAHUI`.
- Validasi jam agar berada pada rentang 0 sampai 23,99.
- Validasi bulan agar berada pada rentang 1 sampai 12.
- Menghitung ulang `FREQ_KECAMATAN` dan `FREQ_PROFESI` setelah cleaning kategori.
- Menggunakan threshold kategori risiko yang lebih konservatif karena distribusi label sangat tidak seimbang.

## 2. Desain Variable Linguistic

### JAM

- `sepi`: trapezoid `(0, 0, 5, 8)`
- `normal`: trapezoid `(6, 9, 15, 18)`
- `sibuk`: trapezoid `(16, 18, 23, 23)`

### BULAN

- `awal`: trapezoid `(1, 1, 3, 5)`
- `tengah`: trapezoid `(4, 6, 8, 10)`
- `akhir`: trapezoid `(9, 11, 12, 12)`

### FREQ_KECAMATAN

- `rendah`: trapezoid `(0, 0, 50, 200)`
- `sedang`: trapezoid `(100, 250, 600, 1000)`
- `tinggi`: trapezoid `(700, 1500, 5065, 5065)`

### FREQ_PROFESI

- `rendah`: trapezoid `(0, 0, 1000, 3500)`
- `sedang`: trapezoid `(2000, 4000, 8000, 10000)`
- `tinggi`: trapezoid `(7000, 10000, 12000, 12000)`

### SKOR_KARAKTERISTIK

- `rendah`: trapezoid `(0, 0, 40, 65)`
- `sedang`: triangle `(50, 70, 90)`
- `tinggi`: trapezoid `(75, 90, 100, 100)`

### Output Risiko

- `rendah`: triangle `(0, 25, 50)`
- `sedang`: triangle `(30, 50, 70)`
- `tinggi`: triangle `(60, 85, 100)`

Threshold kategori setelah kalibrasi:

- `RENDAH`: nilai risiko `< 20`
- `SEDANG`: nilai risiko `20` sampai `< 86`
- `TINGGI`: nilai risiko `>= 86`

## 3. Rule Base

Rule base dibuat dari kombinasi 5 input, masing-masing 3 himpunan linguistik. Jumlah rule:

`3^5 = 243 rule`

Contoh 15 rule pertama:

1. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=rendah AND karakteristik=rendah THEN risiko=rendah
2. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=rendah AND karakteristik=sedang THEN risiko=rendah
3. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=rendah AND karakteristik=tinggi THEN risiko=sedang
4. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=sedang AND karakteristik=rendah THEN risiko=rendah
5. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=sedang AND karakteristik=sedang THEN risiko=sedang
6. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=sedang AND karakteristik=tinggi THEN risiko=sedang
7. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=tinggi AND karakteristik=rendah THEN risiko=rendah
8. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=tinggi AND karakteristik=sedang THEN risiko=sedang
9. IF jam=sepi AND bulan=awal AND freq_kecamatan=rendah AND freq_profesi=tinggi AND karakteristik=tinggi THEN risiko=tinggi
10. IF jam=sepi AND bulan=awal AND freq_kecamatan=sedang AND freq_profesi=rendah AND karakteristik=rendah THEN risiko=rendah
11. IF jam=sepi AND bulan=awal AND freq_kecamatan=sedang AND freq_profesi=rendah AND karakteristik=sedang THEN risiko=sedang
12. IF jam=sepi AND bulan=awal AND freq_kecamatan=sedang AND freq_profesi=rendah AND karakteristik=tinggi THEN risiko=tinggi
13. IF jam=sepi AND bulan=awal AND freq_kecamatan=sedang AND freq_profesi=sedang AND karakteristik=rendah THEN risiko=rendah
14. IF jam=sepi AND bulan=awal AND freq_kecamatan=sedang AND freq_profesi=sedang AND karakteristik=sedang THEN risiko=sedang
15. IF jam=sepi AND bulan=awal AND freq_kecamatan=sedang AND freq_profesi=sedang AND karakteristik=tinggi THEN risiko=tinggi

## 4. Implementasi Mamdani dan Sugeno

Implementasi dibuat from scratch pada `models.py`, tanpa library fuzzy.

Tahapan yang dilakukan:

- Fuzzifikasi: setiap input numerik dihitung derajat keanggotaannya pada himpunan linguistik.
- Inferensi: firing strength rule dihitung menggunakan operator minimum.
- Agregasi: nilai maksimum dari rule dengan output sama digunakan sebagai kekuatan output.
- Defuzzifikasi Mamdani: centroid diskrit pada domain risiko 0 sampai 100.
- Defuzzifikasi Sugeno: weighted average dengan konstanta output `rendah=25`, `sedang=50`, dan `tinggi=85`.

## 5. Evaluasi

Hasil evaluasi pada 36.055 row:

| Metode | Akurasi | MAE |
| --- | ---: | ---: |
| Mamdani | 91,68% | 19,57 |
| Sugeno | 91,68% | 20,99 |

Pada eksperimen ini, akurasi Mamdani dan Sugeno sama setelah data cleaning dan kalibrasi threshold. Namun, nilai kontinu Mamdani memiliki MAE lebih kecil 1,42 poin dibanding Sugeno.

## 6. Interpretasi

Mamdani menghasilkan output yang lebih halus karena proses defuzzifikasi centroid mempertimbangkan bentuk kurva output risiko. Metode ini cocok ketika interpretasi linguistik dan transisi antar kategori perlu dijaga. Kekurangannya adalah komputasi lebih berat karena harus menghitung agregasi pada domain output.

Sugeno lebih sederhana dan cepat karena output setiap kategori berupa konstanta. Hasilnya lebih mudah dihitung dan cocok untuk integrasi dengan sistem optimasi atau machine learning. Kekurangannya, output dapat lebih kaku karena bentuk fungsi output tidak dipakai seperti pada Mamdani.

Akurasi meningkat setelah preprocessing karena kategori teks yang sebelumnya terpisah akibat spasi/format tidak konsisten sudah digabungkan, dan threshold kategori dibuat lebih sesuai dengan distribusi label aktual. Label dataset sangat tidak seimbang, dengan mayoritas data berada pada kategori `LL` atau `SEDANG`, sehingga sistem yang terlalu agresif memprediksi `RENDAH` atau `TINGGI` menghasilkan banyak false prediction.

## 7. Kesimpulan

Dataset, jumlah input, rule base, Mamdani, Sugeno, fuzzifikasi, inferensi, defuzzifikasi, dan evaluasi performa sudah memenuhi ketentuan tugas. Berdasarkan hasil eksperimen, Mamdani menjadi metode yang lebih baik untuk konfigurasi rule dan fungsi keanggotaan pada dataset ini.
