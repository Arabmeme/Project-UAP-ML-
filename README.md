# 🧠 UAP Pembelajaran Mesin  
## Klasifikasi Genre Anime Menggunakan Data Tabular

---

## 📌 Deskripsi Proyek

Proyek ini merupakan tugas **Ujian Akhir Praktikum (UAP)** mata kuliah **Pembelajaran Mesin** yang berfokus pada **klasifikasi genre anime** menggunakan **data tabular** dari MyAnimeList (MAL).

Tujuan utama proyek ini adalah:
- Membandingkan performa **model neural network non-pretrained** dan **dua model pretrained (transfer learning)** pada data tabular
- Melakukan analisis perbandingan performa antar model
- Mengimplementasikan model ke dalam **aplikasi web interaktif berbasis Streamlit**

---

## 📊 Dataset dan Preprocessing

### 🔗 Sumber Dataset
[MyAnimeList (MAL) Anime Dataset](https://www.kaggle.com/datasets/syahrulapriansyah2/myanimelist-2025)

atau

https://drive.google.com/file/d/1T_GLCDbLTzi3rTVut7RHT8S0rsNXuGyi/view?usp=sharing

### 📈 Jumlah Data
Lebih dari **10.000 data anime**

### 🗂️ Jenis Data
- Data tabular
- Fitur numerik dan kategorikal

### ⚙️ Tahapan Preprocessing
Tahapan preprocessing yang dilakukan meliputi:
- Menghapus kolom non-fitur (judul anime, sinopsis, ID, dan kolom deskriptif)
- Menangani missing values (imputasi numerik dan kategorikal)
- Normalisasi / scaling fitur numerik
- Encoding fitur kategorikal
- Sinkronisasi fitur antara proses training dan inference
- Penyesuaian dimensi fitur untuk model MLP

---

## 🤖 Model yang Digunakan

### 1️⃣ MLP (Non-Pretrained)
- Feedforward Neural Network (Multilayer Perceptron)
- Dilatih dari awal tanpa pretrained weight
- Digunakan sebagai baseline model

### 2️⃣ TabNet (Pretrained Model 1)
- Model tabular berbasis attention
- Menggunakan pendekatan transfer learning
- Mampu memilih fitur penting secara adaptif

### 3️⃣ FT-Transformer (Pretrained Model 2)
- Transformer khusus untuk data tabular
- Menggunakan mekanisme self-attention
- Memberikan performa terbaik pada eksperimen ini

---

## 📈 Evaluasi dan Analisis Model

Evaluasi dilakukan menggunakan metrik:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion Matrix
- Grafik Loss dan Accuracy

### 📊 Tabel Perbandingan Model

| Model | Tipe | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) | Analisis Singkat |
|------|------|---------:|---------------------:|------------------:|--------------------:|------------------|
| MLP | Non-Pretrained | 0.91 | 0.91 | 0.91 | 0.91 | Performa baik, namun masih terjadi kesalahan antar genre yang mirip |
| TabNet | Pretrained 1 | 0.89 | 0.89 | 0.89 | 0.89 | Stabil, tetapi performa sedikit di bawah MLP |
| FT-Transformer | Pretrained 2 | **0.99** | **0.99** | **0.99** | **0.99** | Performa terbaik, hampir tanpa kesalahan klasifikasi |

### 🔍 Analisis Singkat
- Model MLP cukup kuat sebagai baseline, tetapi sensitif terhadap kualitas fitur
- TabNet stabil dan interpretatif, namun tidak selalu unggul pada dataset ini
- FT-Transformer unggul karena mampu memodelkan hubungan kompleks antar fitur tabular

---

## 🌐 Panduan Menjalankan Aplikasi Streamlit (Lokal)

1. Clone repository
```bash
git clone https://github.com/Arabmeme/Project-UAP-ML-.git
cd UAP-Streamlit
```
2. Install dependency
```
pip install -r requirements.txt
```
3. Download dan simpan file model File model tidak disertakan di repository karena ukuran besar. Silakan unduh model dari Google Drive berikut:
   https://drive.google.com/file/d/1QPtTDJMMnWCP-eFwkcTXyn7Y4t7qpQPo/view?usp=sharing

   Setelah diunduh, ekstrak folder models/ dan letakkan di:
```
UAP-Streamlit/models/
```
4. Jalankan aplikasi
```
streamlit run app.py
```
Aplikasi akan berjalan pada:
```
http://localhost:8501
```
# 📁 Catatan Repository

Folder models/ tidak disertakan di GitHub karena ukuran file yang besar
Model digunakan secara lokal untuk keperluan demonstrasi
Notebook training tersedia di folder Notebooks/
