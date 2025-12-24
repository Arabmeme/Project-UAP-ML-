# UAP Pembelajaran Mesin  
## Klasifikasi Genre Anime (Data Tabular)

---

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan klasifikasi genre anime menggunakan data tabular
dari MyAnimeList (MAL). Sistem membandingkan tiga model neural network, terdiri dari
satu model non-pretrained dan dua model pretrained, serta diimplementasikan dalam
website interaktif berbasis Streamlit.

---

## Dataset dan Preprocessing

**Sumber Dataset**  
MyAnimeList (MAL) Anime Dataset

**Jumlah Data**  
Lebih dari 5.000 data

**Jenis Data**  
Tabular (numerical dan categorical)

**Preprocessing yang dilakukan**
- Menghapus kolom non-fitur (judul, sinopsis, ID)
- Imputasi nilai kosong
- Scaling fitur numerik
- Encoding fitur kategorikal
- Sinkronisasi fitur antara training dan inference

---

## Model yang Digunakan

### 1. MLP (Non-Pretrained)
- Feedforward Neural Network
- Dilatih dari awal
- Digunakan sebagai baseline model

### 2. TabNet (Pretrained Model 1)
- Model tabular berbasis attention
- Menggunakan transfer learning
- Mampu memilih fitur penting secara adaptif

### 3. FT-Transformer (Pretrained Model 2)
- Transformer khusus data tabular
- Menggunakan self-attention
- Memberikan performa terbaik

---

## Evaluasi Model

Evaluasi dilakukan menggunakan:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion Matrix
- Grafik Loss dan Accuracy

### Tabel Perbandingan Model

| Model | Tipe | Accuracy | Precision | Recall | F1-score | Analisis |
|------|------|---------:|----------:|-------:|---------:|----------|
| MLP | Non-Pretrained | 0.91 | 0.91 | 0.91 | 0.91 | Performa baik, masih keliru pada genre mirip |
| TabNet | Pretrained 1 | 0.89 | 0.89 | 0.89 | 0.89 | Stabil namun sedikit di bawah MLP |
| FT-Transformer | Pretrained 2 | **0.99** | **0.99** | **0.99** | **0.99** | Performa terbaik |

---

## Cara Menjalankan Aplikasi

1. Clone repository
```bash
git clone <url-repository>
cd UAP-Streamlit
```
2. Install dependency
```
pip install -r requirements.txt
```
3. Jalankan aplikasi
```
streamlit run app.py
```
