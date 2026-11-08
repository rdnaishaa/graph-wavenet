import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BLUE = "#2879ed"
RED = "#f46776"
GRAY = "#a8b3c4"


def speed_chart(frame):
    valid = frame.dropna(subset=["current_speed"]).copy()

    if valid.empty:
        st.info("Tidak ada kecepatan valid untuk divisualisasikan.")
        return

    base = alt.Chart(valid).encode(
        x=alt.X(
            "time_wib:T",
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
        opacity=0.55,
        strokeWidth=2,
    )

    points = base.mark_circle(size=32).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=["Observed", "Imputed", "Missing"],
                range=[BLUE, RED, GRAY],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip(
                "time_wib:T",
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
        (line + points).properties(height=315).configure_view(stroke=None),
        width="stretch",
    )


def hourly_chart(frame):
    observed = frame[(frame["status"] == "Observed") & frame["current_speed"].notna()].copy()

    if observed.empty:
        st.info("Tidak ada observasi asli pada rentang ini.")
        return

    observed["hour"] = observed["time_wib"].dt.hour

    profile = observed.groupby("hour")["current_speed"].agg(["mean", "std", "count"]).reset_index()

    profile["low"] = profile["mean"] - profile["std"].fillna(0)
    profile["high"] = profile["mean"] + profile["std"].fillna(0)

    base = alt.Chart(profile).encode(
        x=alt.X(
            "hour:Q",
            title="Jam (WIB)",
            scale=alt.Scale(domain=[0, 23]),
            axis=alt.Axis(values=list(range(0, 24, 2))),
        )
    )

    band = base.mark_area(
        color="#bad8ff",
        opacity=0.45,
    ).encode(
        y=alt.Y("low:Q", title="Kecepatan (km/jam)"),
        y2="high:Q",
    )

    line = base.mark_line(
        color=BLUE,
        strokeWidth=2.8,
        point=True,
    ).encode(
        y="mean:Q",
        tooltip=[
            alt.Tooltip("hour:Q", title="Jam WIB"),
            alt.Tooltip("mean:Q", title="Rata-rata", format=".2f"),
            alt.Tooltip("std:Q", title="Std", format=".2f"),
            alt.Tooltip("count:Q", title="Observasi"),
        ],
    )

    st.altair_chart(
        (band + line).properties(height=280).configure_view(stroke=None),
        width="stretch",
    )

    fastest = profile.loc[profile["mean"].idxmax()]
    slowest = profile.loc[profile["mean"].idxmin()]

    a, b = st.columns(2)

    a.metric(
        "Rata-rata Tertinggi",
        f"{fastest['mean']:.2f} km/jam",
    )
    a.caption(f"Pukul {int(fastest['hour']):02d}.00 WIB")

    b.metric(
        "Rata-rata Terendah",
        f"{slowest['mean']:.2f} km/jam",
    )
    b.caption(f"Pukul {int(slowest['hour']):02d}.00 WIB")

    st.caption(
        "Profil memakai data observed saja, sesuai rentang "
        "tanggal terpilih. Area biru = ±1 standar deviasi."
    )


def daily_chart(frame):
    valid = frame.dropna(subset=["current_speed"]).copy()

    if valid.empty:
        st.info("Tidak ada data harian.")
        return

    valid["date_wib"] = valid["time_wib"].dt.date

    daily = valid.groupby("date_wib", as_index=False)["current_speed"].mean()

    daily["date_wib"] = pd.to_datetime(daily["date_wib"])

    chart = (
        alt.Chart(daily)
        .mark_line(
            color=BLUE,
            point=True,
            strokeWidth=2.7,
        )
        .encode(
            x=alt.X(
                "date_wib:T",
                title="Tanggal (WIB)",
                axis=alt.Axis(format="%d %b"),
            ),
            y=alt.Y(
                "current_speed:Q",
                title="Rata-rata kecepatan (km/jam)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip(
                    "date_wib:T",
                    title="Tanggal",
                    format="%d %b %Y",
                ),
                alt.Tooltip(
                    "current_speed:Q",
                    title="Rata-rata",
                    format=".2f",
                ),
            ],
        )
        .properties(height=260)
        .configure_view(stroke=None)
    )

    st.altair_chart(chart, width="stretch")

    st.caption(
        "Rata-rata harian menggunakan nilai kecepatan valid "
        "(observed dan imputed) pada rentang terpilih."
    )


def render_history_page(
    corridor_records,
    get_history_period,
    get_corridor_date_range,
    pretty_corridor_name,
):
    st.markdown(
        """
        <div style="padding:8px 0 20px">
          <div style="color:#2879ed;font-size:13px;
                      font-weight:800;letter-spacing:1px">
            HISTORICAL TRAFFIC ANALYSIS
          </div>
          <div style="color:#102655;font-weight:800;
                      font-size:clamp(27px,3vw,39px);
                      line-height:1.2;margin-top:8px">
            Riwayat Kondisi Lalu Lintas
          </div>
          <div style="color:#5b7090;font-size:16px;
                      margin-top:9px">
            Eksplorasi pola kecepatan historis,
            karakteristik waktu, dan kualitas data
            corridor pada dataset TomTom Final64.
          </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    period = get_history_period()

    minimum = pd.Timestamp(period["start"]).date()
    maximum = pd.Timestamp(period["end"]).date()

    st.markdown("### Filter Riwayat")

    with st.container(border=True):
        f1, f2, f3 = st.columns([1.4, 1, 1], gap="medium")

        with f1:
            corridor = st.selectbox(
                "Pilih Corridor",
                corridor_records,
                format_func=lambda row: pretty_corridor_name(row["corridor_name"]),
                key="history_page_corridor",
            )

        with f2:
            start_date = st.date_input(
                "Tanggal Awal (UTC)",
                value=minimum,
                min_value=minimum,
                max_value=maximum,
                key="history_page_start",
            )

        with f3:
            end_date = st.date_input(
                "Tanggal Akhir (UTC)",
                value=maximum,
                min_value=minimum,
                max_value=maximum,
                key="history_page_end",
            )

    if start_date > end_date:
        st.error("Tanggal awal tidak boleh melewati tanggal akhir.")
        return

    frame = get_corridor_date_range(
        corridor_file=corridor["corridor_file"],
        start_date=start_date,
        end_date=end_date,
    ).copy()

    if frame.empty:
        st.warning("Data historis tidak tersedia untuk pilihan ini.")
        return

    frame["obs_time_utc"] = pd.to_datetime(
        frame["obs_time_utc"],
        utc=True,
        errors="coerce",
    )

    frame["time_wib"] = frame["obs_time_utc"].dt.tz_convert("Asia/Jakarta")

    frame["current_speed"] = pd.to_numeric(
        frame["current_speed"],
        errors="coerce",
    )

    for col in ("is_observed", "was_imputed"):
        frame[col] = frame[col].astype(str).str.lower().isin(["true", "1", "yes"])

    has_speed = frame["current_speed"].notna()

    frame["status"] = np.select(
        [
            has_speed & frame["is_observed"],
            has_speed & frame["was_imputed"],
        ],
        ["Observed", "Imputed"],
        default="Missing",
    )

    valid_speed = frame["current_speed"].dropna()

    if valid_speed.empty:
        st.info("Tidak ada nilai kecepatan valid dalam periode ini.")
        return

    name = pretty_corridor_name(corridor["corridor_name"])

    st.markdown(f"### Ringkasan — {name}")

    st.caption(
        f"Filter tanggal berdasarkan UTC: "
        f"{start_date:%d %b %Y} – {end_date:%d %b %Y}. "
        "Grafik waktu ditampilkan dalam WIB."
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Kecepatan Rata-rata",
        f"{valid_speed.mean():.2f} km/jam",
    )
    k2.metric(
        "Kecepatan Minimum",
        f"{valid_speed.min():.2f} km/jam",
    )
    k3.metric(
        "Kecepatan Maksimum",
        f"{valid_speed.max():.2f} km/jam",
    )
    k4.metric(
        "Data Valid",
        f"{len(valid_speed):,}",
    )

    st.markdown("### Kualitas Data")

    quality = (
        frame["status"].value_counts().reindex(["Observed", "Imputed", "Missing"], fill_value=0)
    )

    q1, q2, q3 = st.columns(3)

    q1.metric("Observed", f"{quality['Observed']:,}")
    q2.metric("Imputed", f"{quality['Imputed']:,}")
    q3.metric("Missing", f"{quality['Missing']:,}")

    st.caption(
        "Observed = pengamatan asli; Imputed = nilai yang "
        "diisi melalui preprocessing; Missing = baris tanpa "
        "kecepatan valid. Ketiganya dibedakan secara eksplisit."
    )

    st.markdown("### Visualisasi Historis")

    left, right = st.columns([1.35, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("#### Perubahan Kecepatan terhadap Waktu")
            speed_chart(frame)

    with right:
        with st.container(border=True):
            st.markdown("#### Profil Rata-rata 24 Jam")
            hourly_chart(frame)

    with st.container(border=True):
        st.markdown("#### Pola Harian Kecepatan")
        daily_chart(frame)

    st.markdown("### Detail Data Historis")

    with st.expander("Lihat Tabel Riwayat", expanded=False):
        columns = [
            "obs_time_utc",
            "time_wib",
            "current_speed",
            "free_flow_speed",
            "congestion_ratio",
            "status",
            "split",
        ]

        available = [c for c in columns if c in frame.columns]

        table = frame[available].copy()

        table["obs_time_utc"] = table["obs_time_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")
        table["time_wib"] = table["time_wib"].dt.strftime("%Y-%m-%d %H:%M WIB")

        table = table.rename(
            columns={
                "obs_time_utc": "Waktu UTC",
                "time_wib": "Waktu WIB",
                "current_speed": "Speed (km/jam)",
                "free_flow_speed": "Free Flow (km/jam)",
                "congestion_ratio": "Congestion Ratio",
                "status": "Status",
                "split": "Split",
            }
        )

        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=370,
        )

        st.caption(f"{len(table):,} baris historis.")

    st.info(
        "Halaman ini menampilkan analisis historis "
        "gabungan train, validation, dan test. "
        "Prediksi model tetap dievaluasi pada window test set."
    )
