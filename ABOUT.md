# QuantMind — Apa Sih Ini?

## Dalam satu kalimat

QuantMind adalah program komputer yang aku bikin untuk **belajar cara pasar
(saham/kripto) bergerak, lalu mencoba menyusun strategi trading yang teruji
secara data** — bukan asal tebak atau ikut feeling.

Anggap saja seperti membangun "laboratorium riset keuangan" versi kecil,
mirip yang dipakai perusahaan hedge fund besar, tapi dikerjakan sendiri
sebagai proyek belajar (portofolio).

---

## Kenapa aku bikin ini?

Untuk membuktikan dan melatih kemampuan di beberapa bidang sekaligus:

- **Mengolah data** (ambil data harga Bitcoin/saham dari internet secara
  otomatis, bersihkan, simpan rapi)
- **Analisis statistik & matematika keuangan** (menghitung indikator seperti
  RSI, moving average, volatilitas, dll — ada 113 indikator berbeda)
- **Machine Learning / AI** (melatih model untuk memprediksi harga naik/turun)
- **Manajemen risiko** (menghitung berapa banyak potensi rugi sebelum ambil
  posisi, supaya tidak "all-in" dan bangkrut)
- **Rekayasa perangkat lunak** (menyusun semua ini jadi satu sistem yang rapi,
  teruji, dan bisa diandalkan — bukan cuma skrip coba-coba)

Ini bukan "bot ajaib yang otomatis kaya", tapi alat riset untuk **menguji ide
trading secara ilmiah** sebelum ide itu dipakai betulan (atau tidak dipakai
sama sekali kalau ternyata idenya jelek).

---

## Cara kerjanya, langkah demi langkah

Bayangkan alurnya seperti pabrik dengan beberapa stasiun kerja:

### 1. Ambil Data (Data Engineering)
Program otomatis mengunduh data harga historis (misalnya harga Bitcoin
setiap hari sejak 2023) dari sumber seperti Yahoo Finance atau Binance,
lalu menyimpannya rapi di database.

### 2. Hitung Indikator (Feature Engineering)
Dari data harga mentah, program menghitung ratusan "petunjuk" teknikal —
semacam RSI, MACD, rata-rata bergerak, volatilitas — yang biasa dipakai
trader profesional untuk membaca arah pasar.

### 3. Latih Model AI (Machine Learning)
Program mencoba melatih model AI (XGBoost, LightGBM, dll — mirip yang
dipakai untuk prediksi cuaca atau rekomendasi produk) untuk menebak: apakah
harga besok akan naik atau turun, berdasarkan pola-pola historis.

### 4. Uji Strategi ke Data Masa Lalu (Backtesting)
Ini bagian paling penting: **sebelum strategi dipakai ke uang sungguhan**,
program mensimulasikan "kalau strategi ini dipakai dari 2023 sampai
sekarang, untung atau rugi?" — lengkap dengan biaya transaksi, slippage,
dan simulasi "seandainya nasibnya lebih apes" (Monte Carlo).

Contoh hasil nyata dari pengujian kemarin:
> Strategi EMA Crossover di Bitcoin (Okt 2023 – Jun 2026): modal 10 juta
> jadi 25 juta (+150%), tapi sempat rugi maksimal -27% di satu titik.

### 5. Atur Risiko (Risk Management)
Sebelum order benar-benar dikirim, ada "penjaga gerbang" yang menghitung:
berapa maksimal uang yang boleh dipertaruhkan di satu posisi, dan otomatis
menghentikan trading kalau rugi sudah kelewat batas (circuit breaker) —
persis seperti sabuk pengaman.

### 6. Eksekusi (Execution)
Ada dua mode:
- **Paper trading**: simulasi jual-beli pakai uang virtual, untuk latihan
  tanpa risiko.
- **Live trading**: terhubung ke bursa kripto sungguhan (lewat testnet
  dulu, bukan uang asli, sampai benar-benar teruji aman).

### 7. Dashboard
Semua hasil bisa dilihat lewat tampilan web yang enak dibaca (grafik untung
rugi, statistik performa), bukan cuma teks di layar hitam.

---

## Apakah ini sudah "siap pakai untuk cari uang"?

**Belum, dan memang belum untuk itu tujuannya.** Statusnya masih tahap
riset/pembelajaran:

- Sudah bisa: download data, hitung indikator, latih model AI, uji strategi
  ke data historis, hitung risiko, simulasi trading.
- Belum/masih perlu hati-hati: pakai uang sungguhan (harus dites lama dulu
  di akun demo/testnet, dan hasil backtest yang bagus **tidak menjamin**
  untung di masa depan — ini prinsip dasar di dunia finance).

Kalau dianalogikan: ini seperti simulator terbang pesawat yang sangat
detail. Bagus untuk belajar dan latihan, tapi belum berarti langsung siap
terbangkan pesawat sungguhan bawa penumpang.

---

## Kenapa ini berharga (meski belum menghasilkan uang)?

Karena yang dibangun bukan cuma "kode yang jalan", tapi:
- Sudah diuji dengan **104 automated test** (program yang mengecek program
  lain, supaya kalau ada bagian yang salah, langsung ketahuan)
- Strukturnya rapi dan bisa dikembangkan terus (bukan tumpukan kode
  berantakan)
- Menggabungkan banyak disiplin ilmu sekaligus (statistik, AI, software
  engineering, manajemen risiko) dalam satu proyek nyata

Ini semacam bukti karya (portfolio) yang bisa ditunjukkan kalau nanti
melamar kerja di bidang teknologi finansial, data science, atau AI
engineering.
