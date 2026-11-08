from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/final_64"
GWN = RESULT / "graphwavenet_audited/graphwavenet_multiseed_summary.csv"
BASELINE = RESULT / "gclstm_baseline/gclstm_fair_multiseed_summary.csv"

COLORS = {
    "Fixed GWN": "#8096ba",
    "Adaptive GWN": "#43a6cc",
    "Hybrid GWN": "#2879ed",
    "GC-LSTM-style": "#ed9860",
}

HORIZONS = ["t+1", "t+3", "t+6"]


def matching_column(df, *names):
    lookup = {str(col).lower().replace("_", "").replace(" ", ""): col for col in df.columns}
    for name in names:
        key = name.lower().replace("_", "").replace(" ", "")
        if key in lookup:
            return lookup[key]
    return None


def normalize_horizon(value):
    text = str(value).strip().lower().replace(" ", "")
    aliases = {
        "t+1": "t+1",
        "+1": "t+1",
        "1": "t+1",
        "t1": "t+1",
        "h1": "t+1",
        "t+3": "t+3",
        "+3": "t+3",
        "3": "t+3",
        "t3": "t+3",
        "h3": "t+3",
        "t+6": "t+6",
        "+6": "t+6",
        "6": "t+6",
        "t6": "t+6",
        "h6": "t+6",
    }
    return aliases.get(text)


def normalize_table(df, default_model=None):
    horizon_col = matching_column(df, "horizon", "forecast_horizon", "step")
    model_col = matching_column(df, "mode", "model_name", "model", "strategy")

    required = {
        "MAE_mean": matching_column(df, "MAE_mean", "mean_mae", "mae_avg"),
        "MAE_std": matching_column(df, "MAE_std", "std_mae", "mae_sd"),
        "RMSE_mean": matching_column(df, "RMSE_mean", "mean_rmse", "rmse_avg"),
        "RMSE_std": matching_column(df, "RMSE_std", "std_rmse", "rmse_sd"),
    }

    missing = [key for key, col in required.items() if col is None]
    if horizon_col is None or missing:
        raise ValueError(
            "Kolom ringkasan belum cocok. Kolom tersedia: " + ", ".join(map(str, df.columns))
        )

    if model_col is None and default_model is None:
        raise ValueError("Kolom model tidak ditemukan.")

    result = pd.DataFrame()
    result["Horizon"] = df[horizon_col].map(normalize_horizon)

    if default_model:
        result["Model"] = default_model
    else:

        def model_name(raw):
            value = str(raw).lower()
            if "hybrid" in value:
                return "Hybrid GWN"
            if "adaptive" in value:
                return "Adaptive GWN"
            if "fixed" in value:
                return "Fixed GWN"
            if "gclstm" in value or "gc-lstm" in value:
                return "GC-LSTM-style"
            return str(raw)

        result["Model"] = df[model_col].map(model_name)

    for key, col in required.items():
        result[key] = pd.to_numeric(df[col], errors="coerce")

    result = result[result["Horizon"].isin(HORIZONS)].dropna(subset=["MAE_mean", "RMSE_mean"])

    return result


@st.cache_data
def read_evaluation():
    if not GWN.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {GWN}")

    gwn = normalize_table(pd.read_csv(GWN))

    if gwn.duplicated(["Model", "Horizon"]).any():
        raise ValueError("Ada duplikasi model-horizon dalam GWN.")

    note = None
    if BASELINE.exists():
        raw_baseline = pd.read_csv(BASELINE)
        try:
            baseline = normalize_table(
                raw_baseline,
                default_model="GC-LSTM-style",
            )
            if len(baseline) != 3 or baseline["Horizon"].nunique() != 3:
                raise ValueError("Baseline tidak berisi tiga horizon unik.")
            result = pd.concat(
                [gwn, baseline],
                ignore_index=True,
            )
        except ValueError as exc:
            result = gwn
            note = "Baseline belum bisa dibaca otomatis: " + str(exc)
    else:
        result = gwn
        note = "File ringkasan GC-LSTM-style belum ditemukan."

    result["Horizon"] = pd.Categorical(
        result["Horizon"],
        categories=HORIZONS,
        ordered=True,
    )

    return result.sort_values(["Horizon", "Model"]), note


def bar_chart(df, metric, selected_horizon):
    mean_col = metric + "_mean"
    std_col = metric + "_std"

    subset = df.copy()
    if selected_horizon != "Semua Horizon":
        subset = subset[subset["Horizon"] == selected_horizon].copy()

    subset["lower"] = subset[mean_col] - subset[std_col].fillna(0)
    subset["upper"] = subset[mean_col] + subset[std_col].fillna(0)

    base = alt.Chart(subset).encode(
        x=alt.X(
            "Horizon:N",
            sort=HORIZONS,
            title="Horizon Prediksi",
        ),
        xOffset="Model:N",
        y=alt.Y(
            f"{mean_col}:Q",
            title=f"{metric} (km/jam)",
            scale=alt.Scale(zero=True),
        ),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(
                domain=list(COLORS),
                range=list(COLORS.values()),
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
                columns=2,
            ),
        ),
        tooltip=[
            "Model:N",
            "Horizon:N",
            alt.Tooltip(
                f"{mean_col}:Q",
                title="Mean",
                format=".3f",
            ),
            alt.Tooltip(
                f"{std_col}:Q",
                title="Std",
                format=".3f",
            ),
        ],
    )

    return (
        base.mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        )
        .properties(height=320)
        .configure_view(stroke=None)
    )


def render_model_comparison(
    compare_models,
    selected_sample,
    selected_node,
    selected_corridor_display,
    display_time,
    model_label,
):
    st.markdown(
        """
        <div style="margin:8px 0 20px">
          <div style="color:#2879ed;font-size:13px;
                      font-weight:800;letter-spacing:1px">
            MODEL COMPARISON
          </div>
          <div style="color:#102655;font-size:36px;
                      font-weight:800;margin-top:8px">
            Perbandingan Performa Model
          </div>
          <div style="color:#5b7090;font-size:16px">
            Evaluasi Fixed, Adaptive, dan Hybrid Graph WaveNet
            serta GC-LSTM-style baseline pada dataset Final64.
          </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    try:
        df, note = read_evaluation()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Hasil eksperimen belum bisa dimuat: {exc}")
        return

    if note:
        st.warning(note)

    st.caption(
        "Metrik berasal dari eksperimen 5 seed. "
        "Semakin rendah MAE/RMSE, semakin kecil error prediksi. "
        "Hasil GC-LSTM-style masih berasal dari protokol loss historis yang berbeda; "
        "perbandingan ini bersifat deskriptif hingga hasil training koreksi tersedia."
    )

    # RINGKASAN KESELURUHAN
    average = (
        df.groupby("Model", observed=True)[["MAE_mean", "RMSE_mean"]].mean().sort_values("MAE_mean")
    )

    st.markdown("### Ringkasan Evaluasi")

    cols = st.columns(len(average), gap="small")

    for col, (name, row) in zip(cols, average.iterrows()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.metric(
                    "Rata-rata MAE",
                    f"{row['MAE_mean']:.3f}",
                )
                st.caption(f"Rata-rata RMSE: {row['RMSE_mean']:.3f}")

    # FILTER + GRAFIK
    st.markdown("### Perbandingan per Horizon")

    horizon = st.selectbox(
        "Filter Horizon",
        ["Semua Horizon"] + HORIZONS,
        key="mc_horizon",
    )

    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("#### Mean Absolute Error (MAE)")
            st.altair_chart(
                bar_chart(df, "MAE", horizon),
                width="stretch",
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Root Mean Squared Error (RMSE)")
            st.altair_chart(
                bar_chart(df, "RMSE", horizon),
                width="stretch",
            )

    # TABEL RINCI
    st.markdown("### Hasil Multi-Seed")

    table = df[["Model", "Horizon", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"]].copy()

    table["MAE (mean ± std)"] = table.apply(
        lambda r: f"{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}",
        axis=1,
    )
    table["RMSE (mean ± std)"] = table.apply(
        lambda r: f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f}",
        axis=1,
    )

    st.dataframe(
        table[["Model", "Horizon", "MAE (mean ± std)", "RMSE (mean ± std)"]],
        hide_index=True,
        width="stretch",
    )

    # PEMENANG PER HORIZON
    st.markdown("### Model dengan Error Terendah")

    winners = []

    for h in HORIZONS:
        part = df[df["Horizon"] == h]
        if part.empty:
            continue
        mae_best = part.loc[part["MAE_mean"].idxmin()]
        rmse_best = part.loc[part["RMSE_mean"].idxmin()]

        winners.append(
            {
                "Horizon": h,
                "MAE Terendah": mae_best["Model"],
                "MAE": round(float(mae_best["MAE_mean"]), 3),
                "RMSE Terendah": rmse_best["Model"],
                "RMSE": round(float(rmse_best["RMSE_mean"]), 3),
            }
        )

    st.dataframe(
        pd.DataFrame(winners),
        hide_index=True,
        width="stretch",
    )

    st.info(
        "Hybrid GWN dibandingkan dengan Fixed dan Adaptive "
        "untuk menilai strategi graph. GC-LSTM-style adalah "
        "baseline eksternal, sehingga kesimpulan terbaik "
        "harus memperhatikan model pembanding dan metriknya."
    )

    # PERBANDINGAN PREDIKSI TIGA GWN (FITUR LAMA)
    st.divider()
    st.markdown("### Perbandingan Prediksi pada Satu Sampel")

    st.caption(
        f"Corridor: {selected_corridor_display} · "
        f"Node: {selected_node} · Sampel: {selected_sample}. "
        "Ubah pilihan melalui Pengaturan Prediksi di sidebar."
    )

    if st.button(
        "Bandingkan Fixed, Adaptive, Hybrid",
        type="primary",
        key="mc_compare_button",
    ):
        with st.spinner("Menjalankan tiga model..."):
            st.session_state.comparison_result = compare_models(
                sample_idx=selected_sample,
                node_idx=selected_node,
            )
            st.session_state["mc_comparison_context"] = (selected_sample, selected_node)

    comparison_result = st.session_state.get("comparison_result")

    if comparison_result is not None:
        context = st.session_state.get("mc_comparison_context")
        if context != (selected_sample, selected_node):
            st.warning(
                "Hasil tersimpan berasal dari konfigurasi sebelumnya. Jalankan perbandingan ulang."
            )
        else:
            comparison = comparison_result["comparison"].copy()

            comparison["Waktu Target"] = comparison["target_time"].apply(display_time)
            comparison["Model"] = comparison["model"].apply(model_label)

            st.dataframe(
                comparison[
                    [
                        "Model",
                        "label",
                        "predicted_speed",
                        "Waktu Target",
                    ]
                ].rename(
                    columns={
                        "label": "Horizon",
                        "predicted_speed": "Prediksi (km/jam)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

            chart_data = comparison.rename(
                columns={
                    "label": "Horizon",
                    "predicted_speed": "Kecepatan",
                }
            )

            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Horizon:N",
                        sort=["+1 jam", "+3 jam", "+6 jam"],
                    ),
                    xOffset="Model:N",
                    y=alt.Y(
                        "Kecepatan:Q",
                        title="Prediksi (km/jam)",
                    ),
                    color="Model:N",
                    tooltip=[
                        "Model:N",
                        "Horizon:N",
                        alt.Tooltip(
                            "Kecepatan:Q",
                            format=".2f",
                        ),
                    ],
                )
                .properties(height=290)
            )

            st.altair_chart(
                chart,
                width="stretch",
            )

            timing = pd.DataFrame(
                [
                    {
                        "Model": model_label(mode),
                        "Inference (ms)": value,
                    }
                    for mode, value in comparison_result["inference_ms"].items()
                ]
            )

            st.dataframe(
                timing,
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "Waktu inference di atas berasal dari "
                "pengujian pada perangkat saat ini, "
                "bukan benchmark multi-seed."
            )

    st.caption(
        "GC-LSTM-style baseline tercakup dalam evaluasi "
        "agregat di atas, tetapi tidak dipanggil oleh "
        "fungsi compare_models() lama."
    )
