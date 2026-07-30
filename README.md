# 📦 FMCG Forecasting App — Deployment Streamlit

Aplikasi web interaktif yang menggabungkan **hasil kerja Data Analyst (EDA & Business Insights)**
dan **3 model forecasting final dari Data Scientist**:

| Halaman | Role | Isi |
|---|---|---|
| 📊 Data Analyst — EDA & Insights | Data Analyst | 6 visualisasi EDA, restock priority matrix, business impact simulation, key insights & rekomendasi |
| 📦 Model 1 — Demand Forecasting (order-level) | Data Scientist | R² = 0,34 |
| 💰 Model 2 — Revenue Forecasting (order-level) | Data Scientist | R² = 0,54 |
| 📈 Model 3 — Weekly Aggregate Demand Forecasting (per kategori) | Data Scientist | R² = 0,95 |

Sudah diuji lulus (syntax check `python -m py_compile`, validasi struktur `eda_metadata.json`
terhadap seluruh field yang diakses `app.py`, dan pengecekan keberadaan semua asset gambar) —
siap langsung di-deploy.

---

## 📁 Struktur File (WAJIB semuanya ikut di-push ke GitHub)

```
streamlit_app/
├── app.py                        # aplikasi utama Streamlit (5 halaman: EDA + 3 model + overview)
├── requirements.txt              # dependency, versi sudah dipin persis sesuai saat model dilatih
├── app_metadata.json             # daftar kategori, default historis, nama kolom fitur, metrik model (dari DS)
├── eda_metadata.json             # NEW — angka kunci, insight, tabel restock, simulasi dampak bisnis (dari DA)
├── assets/
│   └── eda/                      # NEW — 6 chart PNG hasil render notebook Data Analyst
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

## 🧠 Kenapa chart EDA berupa gambar statis, bukan re-compute saat runtime?

Notebook Data Analyst (`FMCG_Supply_Chain_Predictor_Analysis_v2.ipynb`) memproses raw dataset
`forecasting_data_engineer.csv` (190.757 baris) yang **tidak ikut di-deploy** ke aplikasi ini
(sama seperti pendekatan `app_metadata.json` milik Data Scientist yang tidak menyertakan raw data
saat deploy model). Untuk menjaga konsistensi pendekatan itu sekaligus membuat aplikasi ringan &
cepat load, ke-6 chart hasil analisis diekstrak langsung dari output notebook (PNG) dan seluruh
angka/tabel pendukungnya (revenue, stockout rate, top-15 restock priority, simulasi dampak bisnis,
dll.) disalin ke `eda_metadata.json`. Kalau ke depan raw dataset ingin ikut di-deploy supaya chart
bisa dibuat interaktif (mis. pakai Plotly + filter dinamis), tinggal tambahkan file CSV-nya dan
ganti bagian `st.image(...)` di halaman Data Analyst dengan kode plotting langsung dari DataFrame.

---

## 🚀 Cara Deploy ke Streamlit Community Cloud (streamlit.io)

### 1. Push ke GitHub
```bash
cd streamlit_app
git init
git add .
git commit -m "FMCG forecasting app - final project (DA + DS)"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```
Bisa juga lewat GitHub Desktop / upload manual via web GitHub — yang penting semua isi folder
`streamlit_app/` (termasuk folder `models/` dan `assets/`) ada di root repo.

### 2. Deploy di Streamlit Cloud
1. Buka **[share.streamlit.io](https://share.streamlit.io)**, login dgn akun GitHub yang sama.
2. Klik **"New app"**.
3. Pilih repo, branch `main`, dan **Main file path** isi: `app.py`.
4. Klik **"Deploy"** — tunggu proses build (± 2-5 menit, Streamlit Cloud akan install
   `requirements.txt` otomatis).
5. Selesai — aplikasi bisa diakses via URL publik `https://<nama-app>.streamlit.app`.

### 3. Kalau build gagal
- Cek log build di dashboard Streamlit Cloud.
- Penyebab paling umum: versi Python di Streamlit Cloud tidak cocok dgn `scikit-learn==1.8.0`.
  Kalau itu terjadi, tambahkan file `runtime.txt` berisi `python-3.12` (atau versi Python yang
  dipakai saat training) di root repo.
- Pastikan folder `assets/eda/` ikut ter-push — kalau tidak, halaman Data Analyst akan error
  `FileNotFoundError` saat memanggil `st.image()`.

---

## 💻 Cara Jalankan Lokal (opsional, utk cek dulu sebelum deploy)

```bash
cd streamlit_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Lalu buka `http://localhost:8501` di browser.

---

## 🧠 Cara Kerja Aplikasi (ringkas)

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

## ⚠️ Batasan yang Perlu Diketahui
- Model `delivery_days` (prediksi lama pengiriman) dan klasifikasi stockout **sengaja tidak
  disertakan** — sudah diuji dan terbukti tidak prediktif (R²≈0, ROC-AUC≈0,49) dari data yang ada.
- Nilai "riwayat penjualan" di form (`demand_lag1`, `roll3`, `roll7`, dst) di-default dari rata-rata
  historis per kategori — kalau user tahu angka aktual yang lebih spesifik, sebaiknya diisi manual
  agar prediksi lebih akurat.
- Chart & tabel di halaman Data Analyst bersifat statis (snapshot dari notebook `_v2`), bukan
  live-query — kalau raw dataset diperbarui, notebook perlu dijalankan ulang dan
  `eda_metadata.json` + PNG di `assets/eda/` perlu di-regenerate & di-push ulang.
