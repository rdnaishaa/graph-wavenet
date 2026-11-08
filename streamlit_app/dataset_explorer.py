from datetime import timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "final_64"

BLUE = "#2879ed"
RED = "#f46776"
GRAY = "#a5b3c7"


@st.cache_data(show_spinner="Memuat dataset Final64...")
def load_final64():
    frames = []

    for split in ("train", "val", "test"):
        path = DATA / f"forecasting_{split}_preprocessed.csv"

        if not path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

        part = pd.read_csv(path)
        part["split"] = split
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)

    df["obs_time_utc"] = pd.to_datetime(
        df["obs_time_utc"],
        utc=True,
        errors="coerce",
    )

    df["timestamp_wib"] = df["obs_time_utc"].dt.tz_convert("Asia/Jakarta")

    df["date_wib"] = df["timestamp_wib"].dt.date
    df["hour_wib"] = df["timestamp_wib"].dt.hour

    df["current_speed"] = pd.to_numeric(
        df["current_speed"],
        errors="coerce",
    )

    for name in ("is_observed", "was_imputed"):
        df[name] = df[name].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    valid = df["current_speed"].notna()

    df["status"] = np.select(
        [
            valid & df["is_observed"],
            valid & df["was_imputed"],
        ],
        ["Observed", "Imputed"],
        default="Missing",
    )

    return df.sort_values(["timestamp_wib", "corridor_file"]).reset_index(drop=True)


def chart_theme(chart, height=280):
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
        )
    )


def time_series(frame):
    valid = frame[frame["current_speed"].notna()].copy()

    if valid.empty:
        st.info("Tidak ada data kecepatan pada filter ini.")
        return

    base = alt.Chart(valid).encode(
        x=alt.X(
            "timestamp_wib:T",
            title="Waktu (WIB)",
            axis=alt.Axis(format="%d %b %H:%M"),
        ),
        y=alt.Y(
            "current_speed:Q",
            title="Kecepatan (km/jam)",
            scale=alt.Scale(zero=False),
        ),
    )

    line = base.mark_line(
        color=BLUE,
        strokeWidth=2,
        opacity=0.65,
    )

    dots = base.mark_circle(size=34).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=["Observed", "Imputed", "Missing"],
                range=[BLUE, RED, GRAY],
            ),
            legend=alt.Legend(title="Status"),
        ),
        tooltip=[
            alt.Tooltip(
                "timestamp_wib:T",
                title="Waktu WIB",
                format="%d %b %Y %H:%M",
            ),
            alt.Tooltip(
                "current_speed:Q",
                title="Kecepatan",
                format=".2f",
            ),
            "status:N",
            "split:N",
        ],
    )

    st.altair_chart(
        chart_theme(line + dots, 300),
        width="stretch",
    )


def hourly_profile(frame):
    observed = frame[(frame["status"] == "Observed") & frame["current_speed"].notna()].copy()

    if observed.empty:
        st.info("Tidak ada observasi asli pada tanggal terpilih.")
        return

    profile = (
        observed.groupby("hour_wib")["current_speed"].agg(["mean", "std", "count"]).reset_index()
    )

    profile["lower"] = profile["mean"] - profile["std"].fillna(0)
    profile["upper"] = profile["mean"] + profile["std"].fillna(0)

    base = alt.Chart(profile).encode(
        x=alt.X(
            "hour_wib:Q",
            title="Jam (WIB)",
            scale=alt.Scale(domain=[0, 23]),
            axis=alt.Axis(values=list(range(0, 24, 2))),
        )
    )

    band = base.mark_area(
        color="#bad8ff",
        opacity=0.4,
    ).encode(
        y=alt.Y(
            "lower:Q",
            title="Kecepatan (km/jam)",
        ),
        y2="upper:Q",
    )

    line = base.mark_line(
        color=BLUE,
        strokeWidth=2.5,
        point=True,
    ).encode(
        y="mean:Q",
        tooltip=[
            alt.Tooltip("hour_wib:Q", title="Jam WIB"),
            alt.Tooltip("mean:Q", title="Rata-rata", format=".2f"),
            alt.Tooltip("std:Q", title="Std Dev", format=".2f"),
            alt.Tooltip("count:Q", title="Jumlah observed"),
        ],
    )

    st.altair_chart(
        chart_theme(band + line, 300),
        width="stretch",
    )

    st.caption(
        "Rata-rata per jam WIB dihitung hanya dari data observed. Area biru = ±1 standar deviasi."
    )


def distribution(frame, height=210):
    valid = frame[frame["current_speed"].notna()]

    if valid.empty:
        st.info("Tidak ada nilai kecepatan.")
        return

    chart = (
        alt.Chart(valid)
        .mark_bar(color="#5596f2")
        .encode(
            x=alt.X(
                "current_speed:Q",
                bin=alt.Bin(maxbins=24),
                title="Kecepatan (km/jam)",
            ),
            y=alt.Y(
                "count():Q",
                title="Frekuensi",
            ),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah")],
        )
    )

    st.altair_chart(
        chart_theme(chart, height),
        width="stretch",
    )

    st.caption(
        f"Mean {valid['current_speed'].mean():.2f} · "
        f"Median {valid['current_speed'].median():.2f} · "
        f"Std {valid['current_speed'].std():.2f} km/jam"
    )


def quality_donut(frame):
    counts = (
        frame["status"]
        .value_counts()
        .reindex(
            ["Observed", "Imputed", "Missing"],
            fill_value=0,
        )
        .rename_axis("status")
        .reset_index(name="jumlah")
    )

    chart = (
        alt.Chart(counts)
        .mark_arc(
            innerRadius=55,
            outerRadius=87,
        )
        .encode(
            theta="jumlah:Q",
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["Observed", "Imputed", "Missing"],
                    range=[BLUE, RED, GRAY],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["status:N", "jumlah:Q"],
        )
    )

    st.altair_chart(
        chart_theme(chart, 200),
        width="stretch",
    )

    total = len(frame)

    for _, item in counts.iterrows():
        pct = (100 * item["jumlah"] / total) if total else 0

        st.caption(f"{item['status']}: {int(item['jumlah']):,} ({pct:.2f}%)")


def observed_vs_imputed(frame):
    valid = frame[frame["current_speed"].notna()]

    if valid.empty:
        st.info("Tidak ada data.")
        return

    chart = (
        alt.Chart(valid)
        .mark_tick(
            size=23,
            thickness=2,
        )
        .encode(
            x=alt.X(
                "timestamp_wib:T",
                title="Waktu (WIB)",
                axis=alt.Axis(format="%d %b"),
            ),
            y=alt.Y(
                "current_speed:Q",
                title="Kecepatan (km/jam)",
            ),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["Observed", "Imputed", "Missing"],
                    range=[BLUE, RED, GRAY],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip(
                    "timestamp_wib:T",
                    title="Waktu",
                    format="%d %b %Y %H:%M",
                ),
                "current_speed:Q",
                "status:N",
            ],
        )
    )

    st.altair_chart(
        chart_theme(chart, 185),
        width="stretch",
    )


def corridor_label(filename):
    return filename.removeprefix("tt_").removesuffix(".csv").replace("_", " ").title()


def render_dataset_explorer():

    st.markdown(
        """
        <div style="margin:5px 0 22px;">
            <div style="font-weight:800;color:#2879ed;
                        font-size:13px;letter-spacing:1px;">
                DATASET EXPLORER
            </div>
            <div style="font-weight:800;color:#102655;
                        font-size:clamp(25px,3vw,38px);
                        line-height:1.2;margin:8px 0;">
                Eksplorasi Data TomTom (Final64)
            </div>
            <div style="color:#5b7090;font-size:16px;">
                Lihat dan analisis data kecepatan lalu lintas
                pada 64 corridor Jabodetabek.
            </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    df = load_final64()

    vis, stats, pre, splits = st.tabs(
        [
            "📈 Visualisasi Data",
            "📊 Statistik Dataset",
            "🧩 Preprocessing",
            "🗂️ Pembagian Data",
        ]
    )

    with vis:
        corridors = sorted(df["corridor_file"].dropna().unique())

        idx = corridors.index("tt_sudirman.csv") if "tt_sudirman.csv" in corridors else 0

        f1, f2, f3, f4 = st.columns(4, gap="small")

        with f1:
            corridor = st.selectbox(
                "Pilih Corridor",
                corridors,
                index=idx,
                format_func=corridor_label,
                key="dataset_corridor",
            )

        corridor_df = df[df["corridor_file"] == corridor].copy()

        first_date = corridor_df["date_wib"].min()
        last_date = corridor_df["date_wib"].max()

        with f2:
            dates = st.date_input(
                "Rentang Tanggal (WIB)",
                value=(
                    max(first_date, last_date - timedelta(days=6)),
                    last_date,
                ),
                min_value=first_date,
                max_value=last_date,
                key="dataset_dates",
            )

        with f3:
            status = st.selectbox(
                "Jenis Data",
                [
                    "Semua Data",
                    "Observed",
                    "Imputed",
                    "Missing",
                ],
                key="dataset_status",
            )

        with f4:
            view = st.selectbox(
                "Tampilan",
                ["Time Series", "Distribusi Kecepatan"],
                key="dataset_view",
            )

        if not isinstance(dates, (tuple, list)) or len(dates) != 2:
            st.info("Pilih tanggal mulai dan akhir.")
            return

        start_date, end_date = dates

        filtered = corridor_df[
            (corridor_df["date_wib"] >= start_date) & (corridor_df["date_wib"] <= end_date)
        ].copy()

        quality_frame = filtered.copy()

        if status != "Semua Data":
            filtered = filtered[filtered["status"] == status].copy()

        if filtered.empty:
            st.warning("Tidak ada data sesuai pilihan filter.")
            return

        name = corridor_label(corridor)

        left, right = st.columns(
            [1.35, 1],
            gap="small",
        )

        with left:
            with st.container(border=True):
                st.markdown(f"#### Time Series Kecepatan — {name}")

                if view == "Time Series":
                    time_series(filtered)
                else:
                    distribution(filtered, 300)

        with right:
            with st.container(border=True):
                st.markdown(f"#### Profil Rata-rata 24 Jam ({name})")
                hourly_profile(quality_frame)

        bottom_left, bottom_right = st.columns(
            [1.35, 1],
            gap="small",
        )

        with bottom_left:
            with st.container(border=True):
                st.markdown("#### Data Table (Sampel)")

                table = filtered[
                    [
                        "timestamp_wib",
                        "corridor_file",
                        "current_speed",
                        "status",
                        "split",
                    ]
                ].copy()

                table["timestamp_wib"] = table["timestamp_wib"].dt.strftime("%Y-%m-%d %H:%M")

                table.columns = [
                    "Timestamp (WIB)",
                    "Corridor",
                    "Speed (km/jam)",
                    "Status",
                    "Split",
                ]

                st.dataframe(
                    table,
                    height=360,
                    hide_index=True,
                    width="stretch",
                )

                st.caption(f"Menampilkan {len(table):,} baris sesuai filter.")

        with bottom_right:
            a, b = st.columns(2, gap="small")

            with a:
                with st.container(border=True):
                    st.markdown("#### Distribusi Kecepatan")
                    distribution(filtered)

            with b:
                with st.container(border=True):
                    st.markdown("#### Kualitas Data")
                    quality_donut(quality_frame)

            with st.container(border=True):
                st.markdown("#### Observed vs Imputed")
                observed_vs_imputed(quality_frame)

    with stats:
        st.subheader("Statistik Dataset Final64")

        a, b, c, d = st.columns(4)

        a.metric("Total Baris Grid", f"{len(df):,}")
        b.metric("Jumlah Corridor", df["corridor_file"].nunique())
        c.metric(
            "Kecepatan Valid",
            f"{df['current_speed'].notna().sum():,}",
        )
        d.metric(
            "Missing",
            f"{df['current_speed'].isna().sum():,}",
        )

        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                st.markdown("#### Distribusi Kecepatan")
                distribution(df, 290)

        with right:
            with st.container(border=True):
                st.markdown("#### Status Data")
                quality_donut(df)

        per_corridor = (
            df.groupby("corridor_file")
            .agg(
                total_rows=("current_speed", "size"),
                valid_speed=("current_speed", "count"),
                mean_speed=("current_speed", "mean"),
                min_speed=("current_speed", "min"),
                max_speed=("current_speed", "max"),
            )
            .reset_index()
        )

        st.markdown("#### Ringkasan per Corridor")

        st.dataframe(
            per_corridor,
            hide_index=True,
            width="stretch",
        )

    with pre:
        st.subheader("Preprocessing Final64")

        st.markdown(
            """
            **1. Hourly alignment:** data diselaraskan
            menjadi interval waktu satu jam.

            **2. Causal forward-fill:** imputasi dilakukan
            pada gap terisolasi satu jam menggunakan
            informasi historis sebelumnya.

            **3. Segmentasi:** gap panjang dipisahkan
            agar sliding window tidak melewati
            deret waktu yang terputus.

            **4. Sliding window:** 12 jam riwayat input
            digunakan untuk memprediksi kecepatan
            pada +1, +3, dan +6 jam.
            """
        )

        a, b, c = st.columns(3)

        a.metric(
            "Observed",
            f"{(df['status'] == 'Observed').sum():,}",
        )
        b.metric(
            "Imputed",
            f"{(df['status'] == 'Imputed').sum():,}",
        )
        c.metric(
            "Missing",
            f"{(df['status'] == 'Missing').sum():,}",
        )

        st.info("Missing tidak disamakan dengan imputed. Statistik berasal dari CSV preprocessing.")

    with splits:
        st.subheader("Pembagian Data")

        st.markdown(
            "Pembagian dilakukan secara kronologis agar tidak terjadi pengacakan urutan waktu."
        )

        summary = (
            df.groupby("split")
            .agg(
                total_rows=("current_speed", "size"),
                valid_speed=("current_speed", "count"),
                first_time=("timestamp_wib", "min"),
                last_time=("timestamp_wib", "max"),
            )
            .reset_index()
        )

        summary["sliding_windows"] = summary["split"].map({"train": 415, "val": 92, "test": 93})

        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
        )

        x, y, z = st.columns(3)

        x.metric("Train Windows", 415)
        y.metric("Validation Windows", 92)
        z.metric("Test Windows", 93)

        st.caption(
            "600 sliding windows berbeda dari jumlah baris "
            "CSV hourly. Satu baris CSV mewakili satu "
            "corridor pada satu timestamp."
        )
