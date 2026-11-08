from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "final_64"

TRAINING = BASE / "training_cost_benchmark_seed42.csv"
SCALABILITY = BASE / "scalability" / "scalability_summary_seed42.csv"
OVERALL = BASE / "model_comparison" / "final_5seed_overall_comparison.csv"

COLORS = {
    "Fixed GWN": "#8096ba",
    "Adaptive GWN": "#43a6cc",
    "Hybrid GWN": "#2879ed",
    "GC-LSTM-style": "#ed9860",
}

MODEL_ORDER = list(COLORS)
NODE_ORDER = [8, 16, 32, 64]


@st.cache_data
def load_results():
    for file in [TRAINING, SCALABILITY, OVERALL]:
        if not file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {file}")

    train = pd.read_csv(TRAINING)
    scale = pd.read_csv(SCALABILITY)
    overall = pd.read_csv(OVERALL)

    # Samakan label model untuk visualisasi.
    aliases = {
        "GC-LSTM": "GC-LSTM-style",
        "GC-LSTM Fair": "GC-LSTM-style",
    }

    train["model"] = train["model"].replace(aliases)
    scale["model"] = scale["model"].replace(aliases)
    overall["model_clean"] = overall["model_clean"].replace(aliases)

    required_train = [
        "model",
        "mean_sec_per_epoch",
        "total_training_time_sec",
        "best_epoch_benchmark",
    ]
    required_scale = [
        "model",
        "num_nodes",
        "avg_mae_kmh",
        "avg_rmse_kmh",
        "parameters",
        "mean_sec_per_epoch",
        "inference_ms_per_sample",
    ]
    required_overall = ["model_clean", "avg_mae", "avg_rmse"]

    for df, cols, name in [
        (train, required_train, "Training"),
        (scale, required_scale, "Scalability"),
        (overall, required_overall, "Overall"),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom {name} belum sesuai: {missing}")

    if train["model"].duplicated().any():
        raise ValueError("Model duplikat pada benchmark training.")

    if scale.duplicated(["model", "num_nodes"]).any():
        raise ValueError("Model-node duplikat pada scalability summary.")

    return train, scale, overall


def styled_chart(chart, height=300):
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#587092",
            titleColor="#29476f",
            gridColor="#eaf0f8",
        )
        .configure_legend(
            labelColor="#29476f",
            titleColor="#29476f",
            orient="top",
        )
    )


def palette():
    return alt.Scale(
        domain=MODEL_ORDER,
        range=[COLORS[name] for name in MODEL_ORDER],
    )


def bar_chart(df, field, title, height=300):
    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "model:N",
                sort=MODEL_ORDER,
                title=None,
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y(
                f"{field}:Q",
                title=title,
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "model:N",
                scale=palette(),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(
                    f"{field}:Q",
                    title=title,
                    format=".4f",
                ),
            ],
        )
    )

    st.altair_chart(
        styled_chart(chart, height),
        width="stretch",
    )


def scale_chart(df, field, title, height=310):
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=75), strokeWidth=3)
        .encode(
            x=alt.X(
                "num_nodes:O",
                sort=NODE_ORDER,
                title="Jumlah Node",
            ),
            y=alt.Y(
                f"{field}:Q",
                title=title,
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "model:N",
                scale=palette(),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("num_nodes:O", title="Node"),
                alt.Tooltip(
                    f"{field}:Q",
                    title=title,
                    format=".4f",
                ),
            ],
        )
    )

    st.altair_chart(
        styled_chart(chart, height),
        width="stretch",
    )


def section_intro(label, description):
    st.markdown(f"### {label}")
    st.caption(description)


def render_computational_page():

    st.markdown(
        """
        <div style="padding:8px 0 20px">
          <div style="color:#2879ed;font-size:13px;
                      font-weight:800;letter-spacing:1px">
            COMPUTATIONAL & SCALABILITY
          </div>
          <div style="color:#102655;font-weight:800;
                      font-size:clamp(27px,3vw,39px);
                      line-height:1.2;margin-top:8px">
            Analisis Efisiensi dan Skalabilitas
          </div>
          <div style="color:#5b7090;font-size:16px;
                      margin-top:9px">
            Perbandingan biaya komputasi, kompleksitas
            model, dan pengaruh ukuran jaringan pada Final64.
          </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    try:
        train, scale, overall = load_results()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    st.info(
        "Benchmark training dan scalability menggunakan "
        "eksperimen seed 42. Evaluasi akurasi keseluruhan "
        "berasal dari lima seed. Pengukuran bergantung "
        "pada perangkat dan konfigurasi eksperimen."
    )

    # ========================================================
    # TOP METRICS
    # ========================================================

    hybrid = scale.loc[(scale["model"] == "Hybrid GWN") & (scale["num_nodes"] == 64)]

    baseline = scale.loc[(scale["model"] == "GC-LSTM-style") & (scale["num_nodes"] == 64)]

    if hybrid.empty or baseline.empty:
        st.error("Data Hybrid GWN atau GC-LSTM pada 64 node tidak tersedia.")
        return

    hybrid = hybrid.iloc[0]
    baseline = baseline.iloc[0]

    train_hybrid = train.loc[train["model"] == "Hybrid GWN"]

    train_baseline = train.loc[train["model"] == "GC-LSTM-style"]

    if train_hybrid.empty or train_baseline.empty:
        st.error("Benchmark training kedua model belum lengkap.")
        return

    train_hybrid = train_hybrid.iloc[0]
    train_baseline = train_baseline.iloc[0]

    st.markdown("### Ringkasan Komputasi")

    k1, k2, k3, k4 = st.columns(4, gap="small")

    with k1:
        with st.container(border=True):
            st.caption("Parameter Hybrid (64 node)")
            st.metric(
                "Parameter",
                f"{int(hybrid['parameters']):,}",
            )

    with k2:
        with st.container(border=True):
            st.caption("Parameter GC-LSTM (64 node)")
            st.metric(
                "Parameter",
                f"{int(baseline['parameters']):,}",
            )

    with k3:
        with st.container(border=True):
            st.caption("Training Hybrid")
            st.metric(
                "Detik / epoch",
                f"{train_hybrid['mean_sec_per_epoch']:.3f}",
            )

    with k4:
        with st.container(border=True):
            st.caption("Training GC-LSTM-style")
            st.metric(
                "Detik / epoch",
                f"{train_baseline['mean_sec_per_epoch']:.3f}",
            )

    st.caption(
        "Jumlah parameter mengacu pada eksperimen "
        "skalabilitas 64 node. Waktu per epoch berasal "
        "dari benchmark training khusus seed 42."
    )

    # ========================================================
    # SUBTABS
    # ========================================================

    tab_train, tab_scale, tab_accuracy, tab_insight = st.tabs(
        [
            "⏱️ Computational Benchmark",
            "📈 Scalability",
            "🎯 Accuracy vs Network Size",
            "📋 Research Insights",
        ]
    )

    # ========================================================
    # TRAINING
    # ========================================================

    with tab_train:
        section_intro(
            "Biaya Training Model", "Perbandingan empat model pada benchmark training seed 42."
        )

        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
                st.markdown("#### Waktu Rata-rata per Epoch")

                bar_chart(
                    train,
                    "mean_sec_per_epoch",
                    "Detik / epoch",
                )

        with right:
            with st.container(border=True):
                st.markdown("#### Total Waktu Training")

                bar_chart(
                    train,
                    "total_training_time_sec",
                    "Total training (detik)",
                )

        st.markdown("#### Detail Benchmark Training")

        table = train.rename(
            columns={
                "model": "Model",
                "mean_sec_per_epoch": "Detik per Epoch",
                "total_training_time_sec": "Total Training (s)",
                "best_epoch_benchmark": "Best Epoch",
            }
        )

        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Total training time dipengaruhi jumlah epoch "
            "yang dijalankan. Perbandingan detik per epoch "
            "lebih langsung untuk menilai biaya setiap epoch."
        )

    # ========================================================
    # SCALABILITY
    # ========================================================

    with tab_scale:
        section_intro(
            "Skalabilitas Ukuran Jaringan",
            "Perbandingan Hybrid Graph WaveNet dan GC-LSTM-style pada 8, 16, 32, dan 64 node.",
        )

        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
                st.markdown("#### Jumlah Parameter")

                scale_chart(
                    scale,
                    "parameters",
                    "Jumlah parameter",
                )

        with right:
            with st.container(border=True):
                st.markdown("#### Inference Time per Sample")

                scale_chart(
                    scale,
                    "inference_ms_per_sample",
                    "Milidetik / sampel",
                )

        with st.container(border=True):
            st.markdown("#### Waktu Training per Epoch")

            scale_chart(
                scale,
                "mean_sec_per_epoch",
                "Detik / epoch",
                height=280,
            )

        st.markdown("#### Data Eksperimen Skalabilitas")

        table = scale[
            [
                "model",
                "num_nodes",
                "parameters",
                "mean_sec_per_epoch",
                "inference_ms_per_sample",
            ]
        ].rename(
            columns={
                "model": "Model",
                "num_nodes": "Node",
                "parameters": "Parameter",
                "mean_sec_per_epoch": "Detik / Epoch",
                "inference_ms_per_sample": "Inference (ms / sample)",
            }
        )

        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Hasil berasal dari eksperimen scalability seed 42. "
            "Ukuran jaringan menggunakan subset node berbeda "
            "dan graph KNN yang dibangun ulang."
        )

    # ========================================================
    # ACCURACY
    # ========================================================

    with tab_accuracy:
        section_intro(
            "Akurasi Berdasarkan Ukuran Jaringan",
            "MAE dan RMSE Hybrid GWN dibandingkan dengan "
            "GC-LSTM-style pada masing-masing subset node.",
        )

        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
                st.markdown("#### Average MAE")

                scale_chart(
                    scale,
                    "avg_mae_kmh",
                    "MAE (km/jam)",
                )

        with right:
            with st.container(border=True):
                st.markdown("#### Average RMSE")

                scale_chart(
                    scale,
                    "avg_rmse_kmh",
                    "RMSE (km/jam)",
                )

        st.dataframe(
            scale[
                [
                    "model",
                    "num_nodes",
                    "avg_mae_kmh",
                    "avg_rmse_kmh",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "num_nodes": "Node",
                    "avg_mae_kmh": "MAE (km/jam)",
                    "avg_rmse_kmh": "RMSE (km/jam)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        st.warning(
            "Perubahan MAE/RMSE saat jumlah node bertambah "
            "tidak boleh dianggap sebagai efek ukuran graph "
            "semata, karena komposisi corridor pada subset "
            "8, 16, 32, dan 64 juga berbeda."
        )

    # ========================================================
    # INSIGHTS
    # ========================================================

    with tab_insight:
        st.markdown("### Temuan Utama Penelitian")

        with st.container(border=True):
            st.markdown("**1. Jumlah parameter Hybrid relatif stabil**")
            st.write(
                "Dari 8 ke 64 node, parameter Hybrid GWN "
                "berubah dari 292.835 menjadi 293.955. "
                "GC-LSTM-style berubah dari 83.800 menjadi "
                "553.472 parameter."
            )

        with st.container(border=True):
            st.markdown("**2. GC-LSTM-style lebih cepat**")
            st.write(
                "Pada benchmark training seed 42, "
                "GC-LSTM-style membutuhkan sekitar 0,254 "
                "detik per epoch, sedangkan Hybrid GWN "
                "sekitar 3,820 detik per epoch."
            )

        with st.container(border=True):
            st.markdown("**3. Hybrid terbaik dalam keluarga GWN**")
            st.write(
                "Pada evaluasi keseluruhan lima seed, "
                "Hybrid memiliki rata-rata MAE lebih rendah "
                "dibandingkan Fixed dan Adaptive GWN."
            )

        with st.container(border=True):
            st.markdown("**4. Baseline tetap kompetitif**")
            st.write(
                "GC-LSTM-style memiliki rata-rata MAE "
                "keseluruhan yang sedikit lebih rendah "
                "dibandingkan Hybrid GWN, sehingga "
                "tidak tepat menyatakan Hybrid selalu unggul."
            )

        st.markdown("### Evaluasi Akurasi Lima Seed")

        overall_table = overall.rename(
            columns={
                "model_clean": "Model",
                "avg_mae": "Average MAE",
                "avg_mae_std": "MAE Std",
                "avg_rmse": "Average RMSE",
                "avg_rmse_std": "RMSE Std",
            }
        )

        st.dataframe(
            overall_table,
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Evaluasi akurasi lima seed merupakan eksperimen "
            "berbeda dari scalability benchmark seed 42."
        )

    st.divider()

    st.caption(
        "Sumber: training_cost_benchmark_seed42.csv · "
        "scalability_summary_seed42.csv · "
        "final_5seed_overall_comparison.csv"
    )
