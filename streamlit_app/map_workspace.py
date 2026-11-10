"""Map-first exploration of the frozen test set; no live traffic claims."""

from html import escape

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from streamlit_app.data_loader import get_historical_series, get_target_values, load_test_data
from streamlit_app.forecasting import run_forecast


@st.cache_data(show_spinner=False)
def map_predictions(model, sample):
    return run_forecast(model, sample)["prediction"]


def wib(value):
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
    return timestamp.tz_convert("Asia/Jakarta").strftime("%d %b %Y · %H:%M WIB")


def render_map_workspace(metadata, references, models):
    st.markdown(
        '<div class="location-strip"><div><strong>Jabodetabek</strong>'
        "<span>64 koridor · Prediksi kecepatan</span></div>"
        '<span class="archive-tag">DATA HISTORIS / SEPTEMBER 2026</span></div>',
        unsafe_allow_html=True,
    )
    nodes = metadata.node_index.astype(int).tolist()
    names = metadata.set_index("node_index").corridor_name.to_dict()
    with st.container(key="map-stage"):
        with st.container(key="map-panel"):
            st.markdown(
                '<div class="panel-heading">Jelajahi <mark class="ink-highlight">jalan.</mark></div>'
                '<div class="panel-hint">Pilih koridor, lihat beberapa jam ke depan.</div>',
                unsafe_allow_html=True,
            )
            node = st.selectbox(
                "Cari koridor", nodes, format_func=lambda n: names[n], key="map_node"
            )
            sample = st.selectbox(
                "Waktu acuan",
                list(range(len(references))),
                format_func=lambda i: wib(references[i]["reference_time"]),
                key="map_sample",
            )
            horizon = st.radio(
                "Horizon prediksi",
                [1, 3, 6],
                horizontal=True,
                format_func=lambda h: f"+{h} jam",
                key="map_horizon",
            )
            with st.popover("Model & tampilan", use_container_width=True):
                model = st.selectbox(
                    "Model prediksi",
                    models,
                    key="map_model",
                    index=next((i for i, n in enumerate(models) if "hybrid" in n.lower()), 0),
                )
                focus = st.toggle("Fokus ke koridor terpilih", value=False, key="map_focus")
                st.caption(
                    "Warna titik menunjukkan kecepatan absolut, bukan klasifikasi kemacetan."
                )
            with st.spinner("Menyiapkan prediksi…"):
                predictions = map_predictions(model, sample)
            hidx = [1, 3, 6].index(horizon)
            speed = float(predictions[hidx, node])
            target_time = load_test_data()["target_times"][sample, hidx]
            actual = get_target_values(sample, node).iloc[hidx].actual_speed
            actual_text = (
                f"Aktual {actual:.1f} km/jam"
                if pd.notna(actual)
                else "Observasi aktual tidak tersedia"
            )
            st.markdown(
                f'<div class="forecast-readout"><div class="place">{escape(names[node])}</div>'
                f'<div class="number">{speed:.1f}<small>km/jam</small></div>'
                f'<div class="when">Prediksi +{horizon} jam · {wib(target_time)}</div>'
                f'<div class="comparison-inline">{actual_text}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("Simulasi data uji. Bukan lalu lintas real-time.")
        points = metadata.copy()
        points["speed"] = [round(float(predictions[hidx, int(n)]), 1) for n in points.node_index]
        points["color"] = points.speed.apply(
            lambda v: (
                [207, 107, 92, 235]
                if v < 20
                else [216, 171, 73, 240]
                if v < 40
                else [68, 151, 133, 235]
            )
        )
        selected = points[points.node_index == node]
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                id="corridors",
                data=points,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius=160,
                radius_min_pixels=5,
                radius_max_pixels=9,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1.5,
                stroked=True,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                id="selected-halo",
                data=selected,
                get_position="[longitude, latitude]",
                get_radius=440,
                radius_min_pixels=18,
                get_fill_color=[40, 94, 232, 35],
                pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                id="selected",
                data=selected,
                get_position="[longitude, latitude]",
                get_radius=220,
                radius_min_pixels=8,
                get_fill_color=[40, 94, 232],
                get_line_color=[255, 255, 255],
                line_width_min_pixels=2.5,
                stroked=True,
                pickable=False,
            ),
            pdk.Layer(
                "TextLayer",
                id="label",
                data=selected,
                get_position="[longitude, latitude]",
                get_text="corridor_name",
                get_size=13,
                get_color=[30, 45, 70],
                get_pixel_offset=[0, -26],
                background=True,
                get_background_color=[255, 255, 255, 235],
                background_padding=[7, 4],
                pickable=False,
            ),
        ]
        center = selected.iloc[0] if focus else points[["latitude", "longitude"]].median()
        chart_key = f"network_map_{sample}_{horizon}_{model}_{focus}_{node if focus else 'all'}"

        def select_map_node():
            picked = st.session_state[chart_key].selection.get("objects", {}).get("corridors", [])
            if picked:
                st.session_state["map_node"] = int(picked[0]["node_index"])

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=float(center.latitude),
                    longitude=float(center.longitude) - (0.012 if focus else 0.055),
                    zoom=12 if focus else 10,
                ),
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                tooltip={
                    "html": "<b>{corridor_name}</b><br>{speed} km/jam · prediksi",
                    "style": {
                        "backgroundColor": "#18232f",
                        "color": "white",
                        "borderRadius": "6px",
                    },
                },
            ),
            height=620,
            on_select=select_map_node,
            key=chart_key,
        )
    st.markdown(
        '<div class="map-legend"><span>KECEPATAN</span>'
        '<span><i style="background:#cf6b5c"></i>&lt;20</span>'
        '<span><i style="background:#d8ab49"></i>20–&lt;40</span>'
        '<span><i style="background:#449785"></i>≥40 km/jam</span>'
        '<span class="map-tip">Klik titik untuk memilih · Geser dan zoom untuk menjelajah</span></div>',
        unsafe_allow_html=True,
    )
    heading, summary = st.columns([1, 2], vertical_alignment="center")
    with heading:
        st.markdown(
            f'<div class="summary-title">{escape(names[node])}, beberapa jam ke depan.</div>'
            '<div class="summary-subtitle">12 jam riwayat masukan · 3 horizon prediksi</div>',
            unsafe_allow_html=True,
        )
    with summary:
        for col, h, value in zip(st.columns(3), [1, 3, 6], predictions[:, node]):
            col.metric(f"+{h} jam", f"{value:.1f} km/jam")
    with st.expander("Lihat riwayat & observasi aktual", expanded=False):
        st.caption(
            "Riwayat masukan dapat memuat imputasi. Target yang hilang tidak ditampilkan sebagai observasi."
        )
        render_history(sample, node, predictions)


def render_history(sample, node, predictions):
    history = get_historical_series(sample, node).rename(
        columns={"speed": "Kecepatan", "time": "Waktu"}
    )
    history["Seri"] = "Riwayat masukan"
    targets = load_test_data()["target_times"][sample]
    future = pd.DataFrame(
        {"Waktu": pd.to_datetime(targets), "Kecepatan": predictions[:, node], "Seri": "Prediksi"}
    )
    observed = get_target_values(sample, node).rename(
        columns={"target_time": "Waktu", "actual_speed": "Kecepatan"}
    )
    observed["Seri"] = "Observasi aktual"
    chart_data = pd.concat([history, future, observed[["Waktu", "Kecepatan", "Seri"]]]).dropna(
        subset=["Kecepatan"]
    )
    chart_data["Waktu"] = (
        pd.to_datetime(chart_data.Waktu, utc=True)
        .dt.tz_convert("Asia/Jakarta")
        .dt.tz_localize(None)
    )
    base = alt.Chart(chart_data).encode(
        x=alt.X("Waktu:T", title="Waktu (WIB)"),
        y=alt.Y("Kecepatan:Q", title="Kecepatan (km/jam)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "Seri:N",
            scale=alt.Scale(
                domain=["Riwayat masukan", "Prediksi", "Observasi aktual"],
                range=["#9aa5b1", "#285ee8", "#d8ab49"],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("Waktu:T", format="%d %b %H:%M"),
            "Seri:N",
            alt.Tooltip("Kecepatan:Q", format=".1f"),
        ],
    )
    lines = base.transform_filter(alt.datum.Seri != "Observasi aktual").mark_line(
        point=True, strokeWidth=2
    )
    dots = base.transform_filter(alt.datum.Seri == "Observasi aktual").mark_point(
        size=90, filled=True
    )
    st.altair_chart((lines + dots).properties(height=200), width="stretch")
