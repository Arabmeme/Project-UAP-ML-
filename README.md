## Tabel Perbandingan Model

| Model | Tipe | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) | Analisis Singkat |
|---|---|---:|---:|---:|---:|---|
| MLP | Non-Pretrained | 0.91 | 0.91 | 0.91 | 0.91 | Performa baik, namun masih terjadi kesalahan antar genre yang mirip |
| TabNet | Pretrained 1 | 0.89 | 0.89 | 0.89 | 0.89 | Stabil, tetapi performa sedikit di bawah MLP pada dataset ini |
| FT-Transformer | Pretrained 2 | **0.99** | **0.99** | **0.99** | **0.99** | Performa terbaik, hampir tanpa kesalahan klasifikasi |
