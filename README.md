# commodity-price-forecasting V1
A time series deep learning model for commodity price forecasting

# 🛒 Pantau Pasar - Model Research (LSTM)

## 📌 Deskripsi Proyek
File ini berisi dokumentasi riset model *Machine Learning* untuk memprediksi harga komoditas pangan secara *time-series* menggunakan arsitektur **LSTM (Long Short-Term Memory)**. Riset model ini merupakan bagian dari tahapan pengembangan capstone project **"Pantau Pasar"** (Coding Camp powered by BDT 2026).

## 📥 Input & Output Model (Tensor Shapes)
Berdasarkan proses *sliding window* pada dataset, model dirancang untuk melihat pergerakan harga komoditas selama seminggu terakhir untuk memprediksi harga keesokan harinya.

### 1. Spesifikasi Fitur & Prediksi
- **Jumlah Fitur (N_FEATURES):** **21 Komoditas** yang diproses secara serentak (Multivariate).
- **Periode Input (N_PAST):** **7 Hari** (*look-back window*).
- **Horizon Prediksi (N_FUTURE):** **1 Hari** (*next-day forecast*).

### 2. Dimensi Input (Input Shape)
- **Shape:** `(batch_size, 7, 21)`
- **Penjelasan:** Untuk melakukan satu kali prediksi, model memerlukan input array 3D yang berisi data historis harga dari **21 komoditas** pasar secara paralel selama **7 hari berturut-turut**. `batch_size` yang digunakan saat *training* adalah 8.

### 3. Dimensi Output (Output Shape)
- **Shape:** `(batch_size, 21)`
- **Penjelasan:** Model (melalui *Dense Layer* terakhir) akan mengeluarkan *array* 2D yang berisi ramalan harga untuk ke-**21 komoditas** tersebut untuk **1 hari ke depan**.

## 📊 Sekilas Tentang Dataset
- **Sumber Data:** PIHPS (Pusat Informasi Harga Pangan Strategis Nasional) by Bank Indonesia.
- **Rentang Waktu:** 3 Januari 2022 hingga 30 April 2026
- **Dimensi Data Setelah Preprocessing:** `(1579, 21)` (1.579 hari sekuensial dan 21 fitur).

**Daftar 21 Komoditas (Fitur):**
1. Bawang Merah Ukuran Sedang
2. Bawang Putih Ukuran Sedang
3. Beras Kualitas Bawah I
4. Beras Kualitas Bawah II
5. Beras Kualitas Medium I
6. Beras Kualitas Medium II
7. Beras Kualitas Super I
8. Beras Kualitas Super II
9. Cabai Merah Besar
10. Cabai Merah Keriting
11. Cabai Rawit Hijau
12. Cabai Rawit Merah
13. Daging Ayam Ras Segar
14. Daging Sapi Kualitas 1
15. Daging Sapi Kualitas 2
16. Gula Pasir Kualitas Premium
17. Gula Pasir Lokal
18. Minyak Goreng Curah
19. Minyak Goreng Kemasan Bermerk 1
20. Minyak Goreng Kemasan Bermerk 2
21. Telur Ayam Ras Segar

## ⚙️ Alur Pemrosesan Data (Data Preprocessing)
Sebelum masuk ke tahap *training*, data mentah melalui persiapan berikut:
1. **Pivot Data (`pivot_by_date`):** Transformasi data ke *wide format* (masing-masing komoditas menjadi kolom).
2. **Resampling Harian & Interpolasi (`daily_resampling`):** Mengisi hari yang kosong (akhir pekan/libur) dengan interpolasi linear agar frekuensi waktu (harian) tidak putus.
3. **Normalisasi:** Menggunakan `RobustScaler` (tahan terhadap harga pencilan/outlier). Hanya di-*fit* pada data latih agar tidak terjadi *data leakage*.
4. **Windowing:** Mengubah tabel tabular 2D menjadi format sekuensial 3D menggunakan fungsi `windowed_dataset` (TensorFlow data pipeline).

## 🏗️ Arsitektur & Pelatihan Model
- **Model Utama:** LSTM (Long Short-Term Memory).
  - *Layer 1:* LSTM (64 unit, `return_sequences=True`) + Dropout (0.6)
  - *Layer 2:* LSTM (32 unit) + Dropout (0.4)
  - *Layer 3:* Dense (16 unit, ReLU) + Dropout (0.2)
  - *Output Layer:* Dense (21 unit)
- **Training Details:** Optimizer Adam (`1e-3`), loss fungsi MAE, dilatih selama 200 *epochs* dengan fitur *Early Stopping* untuk mencegah *overfitting*.

## 📤 Evaluasi Model
Berdasarkan hasil pengujian pada *Data Test* (*unseen data*), model memberikan performa:
- **MAE Rata-rata:** Rp 2.909,08
- **RMSE Rata-rata:** Rp 3.350,38
- **MAPE Rata-rata:** **6,35%** (Akurat untuk prediksi harga pangan pasar).

## 🚀 File Utama
- `model_research_v1.ipynb` — Kode sumber (*source code*) keseluruhan penelitian.
