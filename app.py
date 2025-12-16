import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/lr_pipeline.joblib"

st.set_page_config(
    page_title="Prediksi & Optimasi GMV Iklan (Linear Regression)",
    layout="centered",
)

st.title("Prediksi dan Optimasi GMV Iklan Menggunakan Linear Regression")

st.info(
    "Catatan validasi:\n"
    "- Dataset aslinya memakai kolom **TV/Radio/Newspaper → Sales**.\n"
    "- Di aplikasi ini, ketiganya direpresentasikan sebagai **Meta Ads / Google Ads / TikTok Ads** "
    "(mapping label saja) agar relevan ke iklan digital.\n"
    "- `Sales` dipakai sebagai **proxy GMV**."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model belum tersedia.\n\n"
        "Langkah:\n"
        "1) Pastikan `data/advertising.csv` ada\n"
        "2) Jalankan: `python train.py`\n"
        "3) Jalankan: `streamlit run app.py`"
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_names = bundle["feature_names"]
metrics = bundle.get("metrics", {})
mapping = bundle.get("channel_mapping", {})


UI_LABELS = {
    "TV": "Meta Ads Spend",
    "Radio": "Google Ads Spend",
    "Newspaper": "TikTok Ads Spend",
}

with st.expander("Kualitas model (test set)"):
    st.write(f"**RMSE:** {metrics.get('rmse','-')}")
    st.write(f"**R²:** {metrics.get('r2','-')}")

tabs = st.tabs(["Prediksi GMV", "Optimasi Budget (Simulasi)", "Interpretasi Model"])


def to_dataframe(tv: float, radio: float, news: float) -> pd.DataFrame:
    return pd.DataFrame([[tv, radio, news]], columns=feature_names)



with tabs[0]:
    st.subheader("Prediksi GMV dari rencana budget iklan")

    c1, c2, c3 = st.columns(3)
    with c1:
        tv = st.number_input(UI_LABELS["TV"], min_value=0.0, value=100.0, step=1.0)
    with c2:
        radio = st.number_input(UI_LABELS["Radio"], min_value=0.0, value=25.0, step=1.0)
    with c3:
        news = st.number_input(UI_LABELS["Newspaper"], min_value=0.0, value=10.0, step=1.0)

    X_new = to_dataframe(tv, radio, news)
    pred_gmv = float(pipeline.predict(X_new)[0])

    total_spend = float(tv + radio + news)
    roas = pred_gmv / total_spend if total_spend > 0 else np.nan

    st.success(f"Prediksi GMV (proxy Sales): **{pred_gmv:,.3f}**")
    st.write(f"Total spend: **{total_spend:,.3f}**")
    if np.isfinite(roas):
        st.write(f"Estimated ROAS (GMV/Spend): **{roas:,.3f}**")
    else:
        st.warning("ROAS tidak bisa dihitung jika total spend = 0.")

    with st.expander("Lihat input (kolom asli dataset)"):
        st.dataframe(X_new, use_container_width=True)
        st.caption(f"Mapping label: {mapping}")



with tabs[1]:
    st.subheader("Optimasi sederhana (simulasi banyak skenario)")

    st.write(
        "Karena Linear Regression adalah model prediksi, optimasi dilakukan dengan **simulasi**: "
        "kita generate banyak kombinasi budget yang memenuhi batasan, prediksi GMV untuk tiap kombinasi, "
        "lalu pilih kombinasi terbaik."
    )

    total_budget = st.number_input("Total budget maksimal", min_value=0.0, value=150.0, step=1.0)
    n_samples = st.slider("Jumlah skenario simulasi", min_value=200, max_value=5000, value=1500, step=100)

    st.markdown("### Batasan per channel (opsional)")
    col1, col2, col3 = st.columns(3)
    with col1:
        tv_min = st.number_input(f"{UI_LABELS['TV']} min", min_value=0.0, value=0.0, step=1.0)
        tv_max = st.number_input(f"{UI_LABELS['TV']} max", min_value=0.0, value=float(total_budget), step=1.0)
    with col2:
        radio_min = st.number_input(f"{UI_LABELS['Radio']} min", min_value=0.0, value=0.0, step=1.0)
        radio_max = st.number_input(f"{UI_LABELS['Radio']} max", min_value=0.0, value=float(total_budget), step=1.0)
    with col3:
        news_min = st.number_input(f"{UI_LABELS['Newspaper']} min", min_value=0.0, value=0.0, step=1.0)
        news_max = st.number_input(f"{UI_LABELS['Newspaper']} max", min_value=0.0, value=float(total_budget), step=1.0)

    objective = st.selectbox("Kriteria terbaik", ["GMV tertinggi", "ROAS tertinggi"])

    if st.button("Jalankan Optimasi"):
        rng = np.random.default_rng(42)

        low = np.array([tv_min, radio_min, news_min], dtype=float)
        high = np.array([tv_max, radio_max, news_max], dtype=float)
        raw = rng.uniform(low=low, high=high, size=(n_samples, 3))

        sums = raw.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        scaled = raw / sums * float(total_budget)

        X_sim = pd.DataFrame(scaled, columns=feature_names)
        pred_sim = pipeline.predict(X_sim).astype(float)

        spend_sim = X_sim.sum(axis=1).values.astype(float)
        roas_sim = np.where(spend_sim > 0, pred_sim / spend_sim, np.nan)

        res = X_sim.copy()
        res["pred_gmv"] = pred_sim
        res["total_spend"] = spend_sim
        res["pred_roas"] = roas_sim

        if objective == "GMV tertinggi":
            res_sorted = res.sort_values("pred_gmv", ascending=False)
        else:
            res_sorted = res.sort_values("pred_roas", ascending=False)

        best = res_sorted.iloc[0]

        st.success("Rekomendasi budget terbaik (berdasarkan simulasi + model):")
        st.write(
            f"- {UI_LABELS['TV']}: **{best['TV']:.3f}**\n"
            f"- {UI_LABELS['Radio']}: **{best['Radio']:.3f}**\n"
            f"- {UI_LABELS['Newspaper']}: **{best['Newspaper']:.3f}**\n"
            f"- Total spend: **{best['total_spend']:.3f}**\n"
            f"- Prediksi GMV: **{best['pred_gmv']:.3f}**\n"
            f"- Prediksi ROAS: **{best['pred_roas']:.3f}**"
        )

        st.markdown("### Top 10 skenario terbaik (kolom asli dataset)")
        st.dataframe(res_sorted.head(10), use_container_width=True)

        st.caption("Hasil optimasi ini berbasis pola dataset historis (bukan data akun user tertentu).")



with tabs[2]:
    st.subheader("Interpretasi: channel mana yang paling berpengaruh?")

    model = pipeline.named_steps["model"]
    coef_df = pd.DataFrame({
        "Kolom Dataset": feature_names,
        "Label di Aplikasi": [UI_LABELS.get(f, f) for f in feature_names],
        "Koefisien (Linear Regression)": model.coef_.astype(float),
    }).sort_values("Koefisien (Linear Regression)", ascending=False)

    st.dataframe(coef_df, use_container_width=True)

    st.write(
        "Cara baca cepat:\n"
        "- Koefisien lebih besar → channel itu lebih kuat menaikkan GMV (berdasarkan data latih).\n"
        "- Ini interpretasi model linier, bukan jaminan hasil real untuk tiap akun."
    )
