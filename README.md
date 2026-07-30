# 📦 FMCG Forecasting App — Deployment Streamlit

Aplikasi web interaktif untuk 3 model forecasting final:
1. **Demand Forecasting** (order-level) — R² = 0,34
2. **Revenue Forecasting** (order-level) — R² = 0,54
3. **Weekly Aggregate Demand Forecasting** (per kategori) — R² = 0,95

Sudah diuji lulus (syntax check, `streamlit.testing.AppTest` di semua halaman, submit form ketiga
model, dan live-run server) — siap langsung di-deploy.

---

## 📁 Struktur File (WAJIB semuanya ikut di-push ke GitHub)

```
streamlit_app/
├── app.py                 # aplikasi utama Streamlit
├── requirements.txt       # dependency, versi sudah dipin persis sesuai saat model dilatih
├── app_metadata.json      # daftar kategori, default historis, nama kolom fitur, metrik model
└── models/
    ├── model1_demand_forecasting.pkl
    ├── model1_feature_columns.pkl
    ├── model2_revenue_forecasting.pkl
    ├── model2_feature_columns.pkl
    ├── model3_weekly_demand_forecasting.pkl
    └── model3_feature_columns.pkl
```

Total ukuran ± 820 KB — jauh di bawah limit repo GitHub/Streamlit Cloud, aman untuk di-push langsung.

---

## 🚀 Cara Deploy ke Streamlit Community Cloud (streamlit.io)

### 1. Push ke GitHub
```bash
cd streamlit_app
git init
git add .
git commit -m "FMCG forecasting app - final project"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```
Bisa juga lewat GitHub Desktop / upload manual via web GitHub — yang penting semua isi folder
`streamlit_app/` (termasuk folder `models/`) ada di root repo.

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
  aplikasi jalan — load instan) dan `app_metadata.json` (daftar kategori & nilai default historis
  per kategori, supaya form otomatis terisi angka yang masuk akal).
- Fungsi `build_feature_row()` merekonstruksi one-hot encoding secara manual persis sesuai urutan
  kolom saat training (bukan `pd.get_dummies` langsung, karena itu tidak reliable untuk 1 baris input).
- Tiap model punya halaman sendiri di sidebar, dengan form input + tombol prediksi + panel info
  (R², MAE, feature importance).

## ⚠️ Batasan yang Perlu Diketahui
- Model `delivery_days` (prediksi lama pengiriman) dan klasifikasi stockout **sengaja tidak
  disertakan** — sudah diuji dan terbukti tidak prediktif (R²≈0, ROC-AUC≈0,49) dari data yang ada.
- Nilai "riwayat penjualan" di form (`demand_lag1`, `roll3`, `roll7`, dst) di-default dari rata-rata
  historis per kategori — kalau user tahu angka aktual yang lebih spesifik, sebaiknya diisi manual
  agar prediksi lebih akurat.
