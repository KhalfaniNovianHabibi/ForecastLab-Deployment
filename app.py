"""
FMCG Forecasting App — Streamlit deployment utk 3 model final:
1. Demand Forecasting (order-level)
2. Revenue Forecasting (order-level)
3. Weekly Aggregate Demand Forecasting (per kategori)
"""

# ==== IMPORT LIBRARY ====
import json                                                        # baca file metadata (app_metadata.json)
from datetime import date                                          # utk widget input tanggal & hitung fitur tanggal

import joblib                                                      # load model .pkl yg sudah dilatih
import numpy as np                                                 # operasi numerik
import pandas as pd                                                # bikin DataFrame satu baris utk input model
import matplotlib.pyplot as plt                                    # chart feature importance
import streamlit as st                                             # framework utama aplikasi web ini

# ==== KONFIGURASI HALAMAN (WAJIB DIPANGGIL PALING AWAL) ====
st.set_page_config(
    page_title='FMCG Forecasting App',                              # judul tab browser
    page_icon='📦',                                                  # ikon tab browser
    layout='wide',                                                   # gunakan lebar penuh halaman
)


# ==== LOAD METADATA & MODEL (di-cache spy tidak reload berulang setiap interaksi user) ====
@st.cache_data
def load_metadata():
    with open('app_metadata.json') as f:                            # buka file metadata hasil ekstraksi dari notebook
        return json.load(f)                                          # kembalikan sbg dict python


@st.cache_resource
def load_models():
    m1 = joblib.load('models/model1_demand_forecasting.pkl')          # load model 1: demand forecasting (Gradient Boosting)
    m2 = joblib.load('models/model2_revenue_forecasting.pkl')          # load model 2: revenue forecasting (Gradient Boosting)
    m3 = joblib.load('models/model3_weekly_demand_forecasting.pkl')    # load model 3: weekly aggregate demand (Gradient Boosting)
    return m1, m2, m3


@st.cache_data
def load_eda_metadata():
    with open('eda_metadata.json') as f:                              # hasil ekstraksi notebook Data Analyst (EDA & business insight)
        return json.load(f)


META = load_metadata()                                                # metadata: kategori, default historis, kolom fitur, metrik
EDA = load_eda_metadata()                                             # metadata EDA: chart, insight, tabel restock, simulasi dampak bisnis
MODEL1, MODEL2, MODEL3 = load_models()                                # 3 model terlatih siap dipakai prediksi


# ==== HELPER: BANGUN VEKTOR FITUR SESUAI URUTAN KOLOM SAAT TRAINING ====
def build_feature_row(numeric_values: dict, categorical_values: dict, feature_columns: list) -> pd.DataFrame:
    """
    numeric_values     : dict {nama_fitur_numerik: nilai}
    categorical_values : dict {nama_kolom_asli: nilai_terpilih}  -> mis. {'category': 'Yogurt'}
    feature_columns     : urutan kolom fitur persis seperti saat model dilatih (termasuk kolom one-hot)
    Fungsi ini merekonstruksi one-hot encoding manual (bukan pd.get_dummies) supaya konsisten
    walau hanya ada 1 baris input (get_dummies pada 1 baris akan salah krn tdk tau kategori lain).
    """
    row = {}                                                          # dict utk menampung 1 baris fitur final
    for col in feature_columns:                                       # loop sesuai urutan kolom training persis
        is_dummy = False                                               # flag: apakah kolom ini adalah hasil one-hot dari kategori
        for cat_col, cat_val in categorical_values.items():             # cek tiap kolom kategorikal asli (category, channel, dst)
            prefix = f'{cat_col}_'                                      # prefix nama kolom dummy, mis. "category_"
            if col.startswith(prefix):                                  # kalau kolom ini adalah dummy dari cat_col
                row[col] = 1 if col == f'{cat_col}_{cat_val}' else 0    # isi 1 kalau cocok dgn nilai terpilih, else 0
                is_dummy = True                                         # tandai sudah ketemu & terisi
                break                                                    # tidak perlu cek cat_col lain
        if not is_dummy:                                                # kalau bukan kolom dummy -> berarti fitur numerik biasa
            row[col] = numeric_values.get(col, 0)                       # ambil dari input numerik (default 0 kalau tak ada)
    return pd.DataFrame([row])[feature_columns]                        # bungkus jadi DataFrame 1 baris, urutan kolom dijamin sama


def date_features(d: date) -> dict:                                    # turunkan month/quarter/week_of_year/weekend_flag dari tanggal
    return {
        'month': d.month,                                               # bulan (1-12)
        'quarter': (d.month - 1) // 3 + 1,                               # kuartal (1-4)
        'week_of_year': d.isocalendar()[1],                              # minggu ke berapa dlm tahun (ISO week)
        'weekend_flag': 1 if d.weekday() >= 5 else 0,                    # 1 kalau Sabtu/Minggu, else 0
    }


def feature_importance_chart(model, feature_columns, top_n=6):          # bikin chart horizontal bar feature importance
    imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True).tail(top_n)  # ambil top-N fitur
    fig, ax = plt.subplots(figsize=(5, 3))                                # buat figure kecil sesuai proporsi kolom sidebar
    ax.barh(imp.index, imp.values, color='#028090')                       # bar chart horizontal warna teal
    ax.set_xlabel('Importance', fontsize=9)                                # label sumbu x
    ax.tick_params(axis='both', labelsize=8)                               # perkecil font tick spy muat
    fig.tight_layout()                                                     # rapikan layout
    return fig                                                             # kembalikan objek figure utk ditampilkan st.pyplot


# ==== SIDEBAR: NAVIGASI ANTAR MODEL ====
st.sidebar.title('📦 FMCG Forecasting')                                # judul sidebar
st.sidebar.caption('Final Project — Data Science Bootcamp')             # sub-judul kecil
page = st.sidebar.radio(                                                # menu navigasi radio button
    'Pilih halaman:',
    [
        '🏠 Overview',
        '📊 Data Analyst — EDA & Insights',
        '📦 Model 1 — Demand',
        '💰 Model 2 — Revenue',
        '📈 Model 3 — Weekly Aggregate',
    ],
)
st.sidebar.divider()                                                     # garis pemisah
st.sidebar.markdown(
    """
    **Tentang model ini:**
    Semua model dilatih dgn Gradient Boosting Regressor pada data transaksi FMCG 2022-2024
    (190.757 baris), lalu diuji pakai split waktu (bukan acak) agar hasil evaluasi realistis.
    """
)

# =========================================================================
# HALAMAN: OVERVIEW
# =========================================================================
if page == '🏠 Overview':
    st.title('📦 FMCG Demand & Revenue Forecasting')
    st.markdown(
        'Aplikasi ini men-deploy **3 model forecasting terbaik** dari hasil analisis data transaksi FMCG '
        '(2022-2024) — masing-masing menyasar kebutuhan bisnis yang berbeda.'
    )

    col1, col2, col3 = st.columns(3)                                    # 3 kartu ringkasan model sejajar
    with col1:
        st.metric('Model 1 — Demand', f"R² = {META['model_metrics']['model1']['r2']:.2f}", 'Supply Chain')
        st.caption('Prediksi unit terjual per order — basis reorder & safety stock harian.')
    with col2:
        st.metric('Model 2 — Revenue', f"R² = {META['model_metrics']['model2']['r2']:.2f}", 'Finance')
        st.caption('Prediksi revenue per order — basis budgeting & evaluasi ROI promosi.')
    with col3:
        st.metric('Model 3 — Weekly Agg.', f"R² = {META['model_metrics']['model3']['r2']:.2f}", 'Supply Planning')
        st.caption('Prediksi total demand mingguan per kategori — model paling akurat.')

    st.divider()
    st.subheader('Kenapa hanya 3 model?')
    st.markdown(
        """
        Sebelum dibangun penuh, setiap kandidat target diuji kelayakannya secara data-driven.
        **2 target terbukti tidak punya sinyal prediktif** dan sengaja tidak dilanjutkan:

        | Target yang diuji | Metric | Hasil | Keputusan |
        |---|---|---|---|
        | `delivery_days` | R² | ≈ 0,00 | ❌ Tidak dipakai |
        | Klasifikasi stockout | ROC-AUC | ≈ 0,49 | ❌ Tidak dipakai |
        | `units_sold` per order | R² | 0,34 | ✅ Model 1 |
        | `revenue` per order | R² | 0,54 | ✅ Model 2 |
        | Total demand mingguan/kategori | R² | 0,95 | ✅ Model 3 |

        Silakan pilih model di menu sebelah kiri untuk mencoba prediksi secara interaktif.
        """
    )

# =========================================================================
# HALAMAN: DATA ANALYST — EDA & BUSINESS INSIGHTS
# Konten halaman ini diekstrak dari notebook Data Analyst
# (FMCG_Supply_Chain_Predictor_Analysis_v2.ipynb) — 6 visualisasi EDA inti,
# restock priority matrix, dan simulasi dampak bisnis. Karena raw dataset
# (forecasting_data_engineer.csv) tidak ikut di-deploy, chart ditampilkan
# sbg gambar statis hasil render notebook (bukan re-compute saat runtime),
# sedangkan angka & tabel diambil dari eda_metadata.json.
# =========================================================================
elif page == '📊 Data Analyst — EDA & Insights':
    st.title('📊 Data Analyst — Exploratory Data Analysis & Business Insights')
    st.caption(
        'Baseline analitik sebelum model dibangun — kuantifikasi pain point overstock & stockout, '
        'serta rekomendasi restock berbasis data historis 2022-2024.'
    )

    bu = EDA['business_understanding']
    with st.expander('🎯 Business Understanding', expanded=False):
        st.markdown(f"**Ringkasan proyek:** {bu['project_summary']}")
        st.markdown('**Pain points utama:**')
        for p in bu['pain_points']:
            st.markdown(f'- {p}')
        st.markdown(f"**Business objective:** {bu['objective']}")

    st.divider()
    st.subheader('Ringkasan Angka Kunci')
    km = EDA['key_metrics']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Revenue', f"$ {km['total_revenue']:,.0f}")
    c2.metric('Total Unit Terjual', f"{km['total_units_sold']:,.0f} unit")
    c3.metric('Tingkat Stockout', f"{km['stockout_rate_pct']:.2f}%")
    c4.metric('Data Quality', 'Bersih ✅', f"{km['missing_values']} missing, {km['duplicate_rows']} duplikat")
    st.caption(
        f"Dataset: {EDA['dataset_info']['ukuran']} · Periode {EDA['dataset_info']['periode']} · "
        f"Sumber: {EDA['dataset_info']['sumber']}"
    )

    st.divider()
    st.subheader('6 Visualisasi Inti')
    for viz in EDA['visualizations']:                                  # loop tiap visualisasi: gambar + insight
        st.markdown(f"### {viz['title']}")
        img_col, text_col = st.columns([1.4, 1])
        with img_col:
            st.image(f"assets/eda/{viz['file']}", width='stretch')
        with text_col:
            st.markdown(f"**Apa yang ditampilkan:** {viz['apa_yang_ditampilkan']}")
            st.markdown(f"**Cara membaca:** {viz['cara_membaca']}")
            st.info(f"**Insight utama:** {viz['insight']}")
            st.success(f"**Implikasi bisnis:** {viz['implikasi']}")
        st.divider()

    st.subheader('📋 Top 15 Prioritas Restock (SKU x Channel x Region)')
    rp = EDA['restock_priority']
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric('Total Kombinasi Dianalisis', rp['total_combinations_analyzed'])
    rc2.metric('🔴 Critical', rp['risk_distribution']['Critical'])
    rc3.metric('🟠 High', rp['risk_distribution']['High'])
    rc4.metric('🟡 Medium / 🟢 Safe', f"{rp['risk_distribution']['Medium']} / {rp['risk_distribution']['Safe']}")
    st.caption(rp['methodology'])
    top15_df = pd.DataFrame(rp['top15_table'])
    st.dataframe(top15_df, width='stretch', hide_index=True)

    st.divider()
    st.subheader('💡 Business Impact Simulation')
    bis = EDA['business_impact_simulation']
    sim1, sim2 = st.columns(2)
    with sim1:
        st.markdown('**Sisi Overstock (Inventory Waste)**')
        st.metric('Baseline idle stock / hari', f"$ {bis['overstock_side']['baseline_avg_idle_value_per_day_usd']:,.0f}")
        st.metric('Target setelah reduksi 50%', f"$ {bis['overstock_side']['target_after_50pct_reduction_usd']:,.0f}")
        st.metric('Potensi efisiensi modal / hari', f"$ {bis['overstock_side']['potential_capital_efficiency_per_day_usd']:,.0f}")
    with sim2:
        st.markdown('**Sisi Stockout (Lost Revenue)**')
        st.metric('Estimasi total revenue hilang (2022-2024)', f"$ {bis['stockout_side']['total_lost_revenue_estimate_usd']:,.0f}")
        st.metric(
            f"Potensi revenue dipulihkan ({bis['stockout_side']['recoverable_pct_range']})",
            f"$ {bis['stockout_side']['recoverable_revenue_low_usd']:,.0f} - $ {bis['stockout_side']['recoverable_revenue_high_usd']:,.0f}",
        )
    st.caption(bis['note'])

    st.divider()
    st.subheader('🔑 Key Insights & Rekomendasi')
    for insight in EDA['key_insights']:
        st.markdown(f'- {insight}')

    st.markdown('#### Rekomendasi per Role')
    rec_df = pd.DataFrame(EDA['recommendations']).rename(columns={'untuk': 'Untuk', 'rekomendasi': 'Rekomendasi'})
    st.dataframe(rec_df, width='stretch', hide_index=True)

    st.markdown('#### Kesimpulan')
    st.markdown(EDA['conclusion'])
    st.caption(
        'Sumber: FMCG_Supply_Chain_Predictor_Analysis_v2.ipynb — Data Analyst Notebook, '
        'Team: Khalfani Novian Habibi, Rendy Azly, Dennis Wirawan · FTDS 2026.'
    )

# =========================================================================
# HALAMAN: MODEL 1 — DEMAND FORECASTING
# =========================================================================
elif page == '📦 Model 1 — Demand':
    st.title('📦 Model 1 — Demand Forecasting (Order-Level)')
    st.caption('Memprediksi jumlah unit terjual (`units_sold`) utk satu kombinasi SKU × Channel × Region × Pack Type.')

    left, right = st.columns([1.3, 1])                                  # kolom kiri: form input, kanan: info model

    with left:
        with st.form('form_model1'):                                    # form input, submit sekali jalan (efisien, tdk rerun tiap widget)
            c1, c2 = st.columns(2)                                        # 2 kolom dlm form utk hemat tempat
            category = c1.selectbox('Kategori', META['categories'], key='m1_cat')       # pilih kategori produk
            channel = c2.selectbox('Channel', META['channels'], key='m1_chn')            # pilih channel penjualan
            region = c1.selectbox('Region', META['regions'], key='m1_reg')               # pilih wilayah
            pack_type = c2.selectbox('Pack Type', META['pack_types'], key='m1_pack')     # pilih tipe kemasan

            defaults = META['category_defaults'][category]                # ambil default historis sesuai kategori terpilih
            price_unit = c1.number_input('Harga per Unit', min_value=0.5, max_value=20.0,
                                          value=float(defaults['avg_price']), step=0.1, key='m1_price')
            delivery_days = c2.slider('Estimasi Lead Time Pengiriman (hari)', 1, 5,
                                       value=int(round(defaults['avg_delivery_days'])), key='m1_lt')
            promotion_flag = st.checkbox('Promosi sedang aktif?', key='m1_promo')
            order_date = st.date_input('Tanggal Order', value=date(2025, 6, 15), key='m1_date')

            with st.expander('⚙️ Riwayat penjualan (opsional — sudah auto-terisi rata-rata kategori)'):
                st.caption('Kosongkan / biarkan default kalau tidak tahu riwayat penjualan spesifik.')
                demand_lag1 = st.number_input('Unit terjual order sebelumnya', min_value=0.0,
                                               value=float(defaults['avg_units_sold']), key='m1_lag1')
                demand_roll3 = st.number_input('Rata-rata 3 order sebelumnya', min_value=0.0,
                                                value=float(defaults['avg_units_sold']), key='m1_roll3')
                demand_roll7 = st.number_input('Rata-rata 7 order sebelumnya', min_value=0.0,
                                                value=float(defaults['avg_units_sold']), key='m1_roll7')

            submitted1 = st.form_submit_button('🔮 Prediksi Unit Terjual', width='stretch')

        if submitted1:
            dfeat = date_features(order_date)                             # turunkan month/quarter/week_of_year/weekend_flag
            numeric_values = {
                'price_unit': price_unit, 'promotion_flag': int(promotion_flag), 'delivery_days': delivery_days,
                **dfeat, 'demand_lag1': demand_lag1, 'demand_roll3': demand_roll3, 'demand_roll7': demand_roll7,
            }
            categorical_values = {'category': category, 'channel': channel, 'region': region, 'pack_type': pack_type}
            X = build_feature_row(numeric_values, categorical_values, META['model1_columns'])  # susun fitur sesuai urutan training
            pred = MODEL1.predict(X)[0]                                    # prediksi unit terjual
            pred = max(0, pred)                                            # jaga2 tidak negatif (unit tidak mungkin < 0)
            st.success(f'### 📦 Prediksi: **{pred:,.0f} unit terjual**')
            st.caption(f'MAE model ≈ {META["model_metrics"]["model1"]["mae"]:.1f} unit — perkiraan ini bisa meleset sekitar segitu secara rata-rata.')

    with right:
        st.markdown('#### Info Model')
        st.metric('R² (data test)', f"{META['model_metrics']['model1']['r2']:.3f}")
        st.metric('MAE (data test)', f"{META['model_metrics']['model1']['mae']:.2f} unit")
        st.markdown('#### Feature Importance')
        st.pyplot(feature_importance_chart(MODEL1, META['model1_columns']), width='stretch')
        st.caption('Status promosi & riwayat demand adalah prediktor paling dominan.')

# =========================================================================
# HALAMAN: MODEL 2 — REVENUE FORECASTING
# =========================================================================
elif page == '💰 Model 2 — Revenue':
    st.title('💰 Model 2 — Revenue Forecasting (Order-Level)')
    st.caption('Memprediksi revenue (`price_unit × units_sold`) utk satu kombinasi SKU × Channel × Region × Pack Type.')

    left, right = st.columns([1.3, 1])

    with left:
        with st.form('form_model2'):
            c1, c2 = st.columns(2)
            category = c1.selectbox('Kategori', META['categories'], key='m2_cat')
            channel = c2.selectbox('Channel', META['channels'], key='m2_chn')
            region = c1.selectbox('Region', META['regions'], key='m2_reg')
            pack_type = c2.selectbox('Pack Type', META['pack_types'], key='m2_pack')

            defaults = META['category_defaults'][category]
            price_unit = c1.number_input('Harga per Unit', min_value=0.5, max_value=20.0,
                                          value=float(defaults['avg_price']), step=0.1, key='m2_price')
            delivery_days = c2.slider('Estimasi Lead Time Pengiriman (hari)', 1, 5,
                                       value=int(round(defaults['avg_delivery_days'])), key='m2_lt')
            promotion_flag = st.checkbox('Promosi sedang aktif?', key='m2_promo')
            order_date = st.date_input('Tanggal Order', value=date(2025, 6, 15), key='m2_date')

            with st.expander('⚙️ Riwayat penjualan (opsional — sudah auto-terisi rata-rata kategori)'):
                demand_lag1 = st.number_input('Unit terjual order sebelumnya', min_value=0.0,
                                               value=float(defaults['avg_units_sold']), key='m2_lag1')
                demand_roll3 = st.number_input('Rata-rata 3 order sebelumnya', min_value=0.0,
                                                value=float(defaults['avg_units_sold']), key='m2_roll3')
                demand_roll7 = st.number_input('Rata-rata 7 order sebelumnya', min_value=0.0,
                                                value=float(defaults['avg_units_sold']), key='m2_roll7')

            submitted2 = st.form_submit_button('🔮 Prediksi Revenue', width='stretch')

        if submitted2:
            dfeat = date_features(order_date)
            numeric_values = {
                'price_unit': price_unit, 'promotion_flag': int(promotion_flag), 'delivery_days': delivery_days,
                **dfeat, 'demand_lag1': demand_lag1, 'demand_roll3': demand_roll3, 'demand_roll7': demand_roll7,
            }
            categorical_values = {'category': category, 'channel': channel, 'region': region, 'pack_type': pack_type}
            X = build_feature_row(numeric_values, categorical_values, META['model2_columns'])
            pred = MODEL2.predict(X)[0]
            pred = max(0, pred)
            st.success(f'### 💰 Prediksi: **Rp {pred:,.0f}** (satuan mata uang dataset)')
            st.caption(f'MAE model ≈ {META["model_metrics"]["model2"]["mae"]:.1f} — perkiraan ini bisa meleset sekitar segitu secara rata-rata.')

    with right:
        st.markdown('#### Info Model')
        st.metric('R² (data test)', f"{META['model_metrics']['model2']['r2']:.3f}")
        st.metric('MAE (data test)', f"{META['model_metrics']['model2']['mae']:.2f}")
        st.markdown('#### Feature Importance')
        st.pyplot(feature_importance_chart(MODEL2, META['model2_columns']), width='stretch')
        st.caption('`units_sold` sengaja TIDAK dipakai sbg fitur (leakage) — harga & promosi jadi prediktor utama.')

# =========================================================================
# HALAMAN: MODEL 3 — WEEKLY AGGREGATE DEMAND
# =========================================================================
elif page == '📈 Model 3 — Weekly Aggregate':
    st.title('📈 Model 3 — Weekly Aggregate Demand Forecasting')
    st.caption('Memprediksi TOTAL unit terjual per kategori dalam satu minggu — model paling akurat (R² = 0,95).')

    left, right = st.columns([1.3, 1])

    with left:
        with st.form('form_model3'):
            category = st.selectbox('Kategori', META['categories'], key='m3_cat')
            wdef = META['weekly_defaults'][category]                      # default historis mingguan sesuai kategori

            c1, c2 = st.columns(2)
            lag1 = c1.number_input('Total unit minggu lalu (t-1)', min_value=0.0,
                                    value=float(wdef['avg_weekly_units']), key='m3_lag1')
            lag2 = c2.number_input('Total unit 2 minggu lalu (t-2)', min_value=0.0,
                                    value=float(wdef['avg_weekly_units']), key='m3_lag2')
            roll4 = c1.number_input('Rata-rata 4 minggu terakhir', min_value=0.0,
                                     value=float(wdef['avg_weekly_units']), key='m3_roll4')
            n_orders = c2.number_input('Jumlah order minggu ini', min_value=0,
                                        value=int(wdef['avg_n_orders']), key='m3_norders')
            avg_price = c1.number_input('Rata-rata harga minggu ini', min_value=0.5,
                                         value=float(wdef['avg_price']), step=0.1, key='m3_price')
            promo_rate = c2.slider('Proporsi order dgn promosi minggu ini', 0.0, 1.0,
                                    value=float(wdef['avg_promo_rate']), step=0.05, key='m3_promo')
            week_date = st.date_input('Tanggal Awal Minggu yang Diprediksi', value=date(2025, 6, 16), key='m3_date')

            submitted3 = st.form_submit_button('🔮 Prediksi Total Demand Mingguan', width='stretch')

        if submitted3:
            numeric_values = {
                'lag1': lag1, 'lag2': lag2, 'roll4': roll4, 'promo_rate': promo_rate,
                'avg_price': avg_price, 'n_orders': n_orders,
                'month': week_date.month, 'weekofyear': week_date.isocalendar()[1],
            }
            categorical_values = {'category': category}
            X = build_feature_row(numeric_values, categorical_values, META['model3_columns'])
            pred = MODEL3.predict(X)[0]
            pred = max(0, pred)
            st.success(f'### 📈 Prediksi: **{pred:,.0f} unit** total minggu ini utk kategori {category}')
            st.caption(f'MAE model ≈ {META["model_metrics"]["model3"]["mae"]:.0f} unit — model paling akurat dari ketiganya.')

    with right:
        st.markdown('#### Info Model')
        st.metric('R² (data test)', f"{META['model_metrics']['model3']['r2']:.3f}")
        st.metric('MAE (data test)', f"{META['model_metrics']['model3']['mae']:.0f} unit")
        st.markdown('#### Feature Importance')
        st.pyplot(feature_importance_chart(MODEL3, META['model3_columns']), width='stretch')
        st.caption('Agregasi mingguan meredam noise antar-order — akurasi jauh lebih tinggi dari Model 1.')

# ==== FOOTER ====
st.divider()
st.caption('FMCG Forecasting App · Final Project Data Science Bootcamp · Model: Gradient Boosting Regressor')
