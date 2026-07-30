# 📦 FMCG Forecasting App — Streamlit Deployment

| Halaman | Role | Isi |
|---|---|---|
| Data Analyst — EDA & Insights | Data Analyst | 6 visualisasi EDA, restock priority matrix, business impact simulation, key insights & rekomendasi |
| Model 1 — Demand Forecasting (order-level) | Data Scientist | R² = 0,34 |
| Model 2 — Revenue Forecasting (order-level) | Data Scientist | R² = 0,54 |
| Model 3 — Weekly Aggregate Demand Forecasting (per kategori) | Data Scientist | R² = 0,95 |

---

## Struktur File

```
streamlit_app/
├── app.py                        # aplikasi utama Streamlit (5 halaman: EDA + 3 model + overview)
├── requirements.txt              # dependency, versi sudah dipin persis sesuai saat model dilatih
├── app_metadata.json             # daftar kategori, default historis, nama kolom fitur, metrik model (dari DS)
├── eda_metadata.json             # angka kunci, insight, tabel restock, simulasi dampak bisnis (dari DA)
├── assets/
│   └── eda/                      # 6 chart PNG hasil render notebook Data Analyst
│       ├── viz1_revenue_trend.png
│       ├── viz2_revenue_distribution.png
│       ├── viz3_stockout_heatmap.png
│       ├── viz4_waste_heatmap.png
│       ├── viz5_promo_effectiveness.png
│       └── viz6_restock_priority.png
└── models/
    ├── model1_demand_forecasting.pkl
    ├── model1_feature_columns.pkl
    ├── model2_revenue_forecasting.pkl
    ├── model2_feature_columns.pkl
    ├── model3_weekly_demand_forecasting.pkl
    └── model3_feature_columns.pkl
```

---

## Cara Kerja Aplikasi

- `app.py` memuat ulang 3 model `.pkl` (sudah dilatih sebelumnya, jadi TIDAK training ulang saat
  aplikasi jalan — load instan), `app_metadata.json` (kategori & default historis DS), dan
  `eda_metadata.json` (insight & data pendukung DA).
- Fungsi `build_feature_row()` merekonstruksi one-hot encoding secara manual persis sesuai urutan
  kolom saat training (bukan `pd.get_dummies` langsung, karena itu tidak reliable untuk 1 baris input).
- Halaman **Data Analyst — EDA & Insights** menampilkan 6 chart PNG dari `assets/eda/` beserta
  narasi "Apa yang ditampilkan → Cara membaca → Insight → Implikasi bisnis", tabel Top 15 prioritas
  restock, dan simulasi dampak bisnis — semuanya diambil dari `eda_metadata.json`.
- Tiap model forecasting (DS) punya halaman sendiri di sidebar, dengan form input + tombol
  prediksi + panel info (R², MAE, feature importance).
