"""Single-purpose research screens without nested tab navigation."""

from html import escape

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from streamlit_app.computational_page import load_results
from streamlit_app.data_loader import get_target_values
from streamlit_app.full_history_loader import load_full_history
from streamlit_app.graph_explorer import load_adaptive_graph, load_fixed_graph
from streamlit_app.map_workspace import map_predictions, render_history, wib
from streamlit_app.model_comparison_page import read_evaluation
from streamlit_app.routing_utils import get_route_between_corridors

MODEL_COLORS = ["#e5ba36", "#55998b", "#e9803f", "#647bb5"]
MODEL_NAMES = ["Fixed GWN", "Adaptive GWN", "Hybrid GWN", "GC-LSTM-style"]
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


def heading(title, description):
    st.markdown(
        f'<div class="page-heading"><h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def chart(value, height=290):
    st.altair_chart(
        value.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(gridColor="#eef0f3", labelColor="#778390", titleColor="#5c6875")
        .configure_legend(orient="bottom", title=None),
        width="stretch",
    )


def model_color(field="Model:N"):
    return alt.Color(field, scale=alt.Scale(domain=MODEL_NAMES, range=MODEL_COLORS), title=None)


def download(frame, name):
    st.download_button(
        "Unduh data CSV", frame.to_csv(index=False).encode(), name, "text/csv", key=name
    )


@st.cache_data(show_spinner=False)
def history_data():
    return load_full_history()


def render_data(metadata):
    heading("Arsip lalu lintas.", "Dari observasi asli sampai data yang siap digunakan model.")
    history = history_data()
    controls, period = st.columns([1, 2], vertical_alignment="bottom")
    with controls:
        name = st.selectbox("Koridor", metadata.corridor_name.tolist(), key="archive_corridor")
    with period:
        st.caption(
            f"Periode arsip · {wib(history.obs_time_utc.min())} — {wib(history.obs_time_utc.max())}"
        )
    filename = metadata.loc[metadata.corridor_name == name, "corridor_file"].iloc[0]
    frame = history[history.corridor_file == filename].sort_values("obs_time_utc").copy()
    frame["Waktu"] = frame.obs_time_utc.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    frame["Status"] = np.where(
        frame.is_observed,
        "Observasi asli",
        np.where(frame.current_speed.notna(), "Imputasi", "Tidak tersedia"),
    )
    cols = st.columns(4)
    values = [
        ("Koridor penelitian", "64"),
        ("Observasi asli", f"{frame.is_observed.sum():,} jam"),
        (
            "Kecepatan diimputasi",
            f"{((~frame.is_observed) & frame.current_speed.notna()).sum():,} jam",
        ),
        ("Kecepatan hilang", f"{frame.current_speed.isna().sum():,} jam"),
    ]
    for c, (label, value) in zip(cols, values):
        c.metric(label, value)
    st.markdown("### Satu bulan di jalan ini")
    st.caption(
        "Garis mengikuti segmen data valid. Titik membedakan observasi asli dan hasil imputasi."
    )
    frame["Segmen"] = frame.segment_id.astype("string")
    base = alt.Chart(frame).encode(
        x=alt.X("Waktu:T", title="Waktu (WIB)"),
        y=alt.Y("current_speed:Q", title="Kecepatan (km/jam)"),
        tooltip=[
            alt.Tooltip("Waktu:T", format="%d %b %H:%M"),
            alt.Tooltip("current_speed:Q", title="km/jam"),
            "Status:N",
        ],
    )
    line = (
        base.transform_filter("isValid(datum.segment_id)")
        .mark_line(color="#c5ced9", strokeWidth=1.5)
        .encode(detail="Segmen:N")
    )
    dots = base.mark_point(size=20, filled=True).encode(
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=["Observasi asli", "Imputasi", "Tidak tersedia"],
                range=["#285ee8", "#d5a65b", "#e6e9ed"],
            ),
        )
    )
    chart(line + dots, 300)
    st.markdown("### Ketersediaan per jam")
    chart(
        alt.Chart(frame)
        .mark_tick(thickness=2, size=25)
        .encode(
            x=alt.X("Waktu:T", title=None),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Observasi asli", "Imputasi", "Tidak tersedia"],
                    range=["#285ee8", "#d5a65b", "#e6e9ed"],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Waktu:T", format="%d %b %H:%M"), "Status:N"],
        ),
        45,
    )
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### Dari arsip ke model")
        st.markdown(
            """<div class="process-list"><div><b>01</b><span><strong>Selaraskan waktu</strong>Grid satu jam, timestamp UTC, dan duplikasi dibersihkan.</span></div>
        <div><b>02</b><span><strong>Tangani kekosongan</strong>Gap terisolasi diisi dari jam sebelumnya; gap panjang memisahkan segmen.</span></div>
        <div><b>03</b><span><strong>Bentuk masukan</strong>Split kronologis, standardisasi dari training, lalu window 12 jam.</span></div></div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "Seleksi koridor memakai seluruh arsip. Penentuan gap terisolasi juga memeriksa ketersediaan jam berikutnya; ini bukan pipeline online yang sepenuhnya kausal."
        )
    with right:
        st.markdown("### Pembagian waktu")
        split = history.groupby("split").obs_time_utc.agg(["min", "max"]).sort_values("min")
        for key, row in split.iterrows():
            st.markdown(f"**{key.capitalize()}**")
            st.caption(f"{wib(row['min'])} → {wib(row['max'])}")
        st.caption(
            "Target evaluasi hanya memakai observasi asli. Kecepatan imputasi tidak dianggap sebagai target aktual."
        )
    download(
        frame[
            ["obs_time_utc", "current_speed", "is_observed", "was_imputed", "segment_id", "split"]
        ],
        "arsip-koridor.csv",
    )


def render_graph(metadata):
    heading("Hubungan antarkoridor.", "Lihat hubungan geografis dan bobot yang dipelajari model.")
    a, b = st.columns(2)
    with a:
        kind = st.selectbox(
            "Representasi graf", ["Fixed · kedekatan geografis", "Adaptive · hasil pembelajaran"]
        )
    with b:
        node = st.selectbox(
            "Koridor pusat",
            metadata.node_index.tolist(),
            format_func=lambda n: metadata.set_index("node_index").loc[n, "corridor_name"],
        )
    adaptive = kind.startswith("Adaptive")
    matrix, _ = load_adaptive_graph() if adaptive else (load_fixed_graph(), None)
    if matrix is None:
        st.info("Bobot graf adaptif belum tersedia.")
        return
    row = matrix[node].copy()
    row[node] = 0
    dest = np.argsort(row)[::-1][:5] if adaptive else np.flatnonzero(row > 0)
    dest = [int(n) for n in dest if row[n] > 0]
    lookup = metadata.set_index("node_index")
    source = lookup.loc[node]
    edges = [
        {
            "start": [float(source.longitude), float(source.latitude)],
            "end": [float(lookup.loc[n].longitude), float(lookup.loc[n].latitude)],
        }
        for n in dest
    ]
    points = metadata.copy()
    points["color"] = points.node_index.apply(
        lambda n: [40, 94, 232] if n == node else [67, 167, 153] if n in dest else [184, 194, 205]
    )
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.pydeck_chart(
            pdk.Deck(
                layers=[
                    pdk.Layer(
                        "LineLayer",
                        data=edges,
                        get_source_position="start",
                        get_target_position="end",
                        get_color=[40, 94, 232, 140],
                        get_width=2,
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=points,
                        get_position="[longitude, latitude]",
                        get_fill_color="color",
                        get_radius=200,
                        radius_min_pixels=5,
                        pickable=True,
                    ),
                ],
                initial_view_state=pdk.ViewState(
                    latitude=float(source.latitude), longitude=float(source.longitude), zoom=10
                ),
                map_style=MAP_STYLE,
                tooltip={"text": "{corridor_name}"},
            ),
            height=440,
        )
    with right:
        st.markdown("### " + ("Lima bobot keluar terbesar" if adaptive else "Tetangga keluar"))
        st.caption(source.corridor_name)
        for n in dest:
            st.markdown(
                f'<div class="connection-row"><span>{escape(lookup.loc[n, "corridor_name"])}</span><b>{row[n]:.4f}</b></div>',
                unsafe_allow_html=True,
            )
        st.caption(
            "Garis merupakan relasi graf, bukan rute jalan. Bobot ditampilkan dari simpul sumber ke simpul tujuan."
        )
        if adaptive:
            st.caption(
                "Adaptive GWN · seed 42. Hanya lima bobot ditampilkan; matriks asli tidak dipangkas."
            )
    st.markdown("### Seluruh hubungan, dalam satu matriks")
    i, j = np.indices(matrix.shape)
    grid = pd.DataFrame({"Sumber": i.ravel(), "Tujuan": j.ravel(), "Bobot": matrix.ravel()})
    chart(
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("Tujuan:O", axis=alt.Axis(values=list(range(0, 64, 8)))),
            y=alt.Y("Sumber:O", axis=alt.Axis(values=list(range(0, 64, 8)))),
            color=alt.Color("Bobot:Q", scale=alt.Scale(range=["#f3f6fb", "#285ee8"])),
            tooltip=["Sumber:O", "Tujuan:O", "Bobot:Q"],
        ),
        340,
    )


def forecast_controls(metadata, references, models, prefix):
    a, b, c = st.columns([1, 1.2, 1])
    with a:
        node = st.selectbox(
            "Koridor",
            metadata.node_index.tolist(),
            key=prefix + "_node",
            format_func=lambda n: metadata.set_index("node_index").loc[n, "corridor_name"],
        )
    with b:
        sample = st.selectbox(
            "Waktu acuan",
            list(range(len(references))),
            key=prefix + "_sample",
            format_func=lambda i: wib(references[i]["reference_time"]),
        )
    with c:
        model = st.selectbox(
            "Model",
            models,
            index=next((i for i, n in enumerate(models) if "hybrid" in n.lower()), 0),
            key=prefix + "_model",
        )
    return node, sample, model


def render_prediction(metadata, references, models):
    heading(
        "Dua belas jam masuk. Tiga prediksi keluar.",
        "Bandingkan prediksi kecepatan dengan observasi aktual pada data uji.",
    )
    node, sample, model = forecast_controls(metadata, references, models, "prediction")
    with st.spinner("Memuat prediksi…"):
        pred = map_predictions(model, sample)
    actual = get_target_values(sample, node)
    for c, (h, value) in zip(st.columns(3), zip([1, 3, 6], pred[:, node])):
        c.metric(f"+{h} jam", f"{value:.1f} km/jam")
        v = actual[actual.horizon == h].actual_speed.iloc[0]
        c.caption(
            f"Aktual {v:.1f} · Selisih {value - v:+.1f} km/jam"
            if pd.notna(v)
            else "Observasi aktual tidak tersedia"
        )
    st.markdown("### Riwayat dan hasil prediksi")
    render_history(sample, node, pred)
    st.caption(
        "Riwayat masukan dapat memuat imputasi. Observasi target yang hilang tidak ditampilkan; hasil ini bukan lalu lintas real-time."
    )
    actual["predicted_speed"] = pred[:, node]
    download(actual, "prediksi-koridor.csv")


def render_models():
    heading(
        "Tiga graf. Satu tugas prediksi.",
        "Performa dan variasi hasil pada lima random seed; galat lebih kecil berarti lebih baik.",
    )
    frame, note = read_evaluation()
    if note:
        st.caption(note)
    for column, metric in zip(st.columns(2, gap="large"), ["MAE", "RMSE"]):
        with column:
            st.markdown(f"### {metric}")
            data = frame.copy()
            data["lower"] = data[metric + "_mean"] - data[metric + "_std"]
            data["upper"] = data[metric + "_mean"] + data[metric + "_std"]
            base = alt.Chart(data).encode(
                x=alt.X("Horizon:N", title="Horizon"), xOffset="Model:N", color=model_color()
            )
            bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                y=alt.Y(metric + "_mean:Q", title="km/jam"),
                tooltip=[
                    "Model:N",
                    "Horizon:N",
                    alt.Tooltip(metric + "_mean:Q", format=".3f"),
                    alt.Tooltip(metric + "_std:Q", format=".3f"),
                ],
            )
            intervals = base.mark_rule(color="#253348").encode(y="lower:Q", y2="upper:Q")
            chart(bars + intervals, 330)
    st.caption(
        "Batang = rata-rata lima seed. Garis = ±1 standar deviasi antar-seed. Himpunan target valid dapat berbeda antarhorizon."
    )
    st.markdown(
        '<div class="research-note"><b>Batas perbandingan baseline</b>GC-LSTM-style memakai hasil historis dengan normalisasi loss berbeda. Perbandingan ini deskriptif, bukan bukti keunggulan arsitektur secara mutlak.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Angka di balik grafik")
    table = frame.copy()
    for metric in ["MAE", "RMSE"]:
        table[metric + " (km/jam)"] = table.apply(
            lambda r: f"{r[metric + '_mean']:.3f} ± {r[metric + '_std']:.3f}", axis=1
        )
    st.dataframe(
        table[["Model", "Horizon", "MAE (km/jam)", "RMSE (km/jam)"]],
        hide_index=True,
        width="stretch",
    )
    download(frame, "perbandingan-model.csv")


def render_compute():
    heading(
        "Saat jaringan bertambah besar.", "Perubahan kebutuhan komputasi dari 8 hingga 64 koridor."
    )
    training, scale, _ = load_results()
    st.caption("Eksperimen seed 42 · Waktu bergantung pada perangkat dan konfigurasi benchmark.")
    for pair in [
        [
            ("parameters", "Jumlah parameter"),
            ("inference_ms_per_sample", "Inferensi per sampel (ms)"),
        ],
        [
            ("mean_sec_per_epoch", "Training per epoch (detik)"),
            ("avg_mae_kmh", "MAE rata-rata horizon (km/jam)"),
        ],
    ]:
        for col, (field, title) in zip(st.columns(2, gap="large"), pair):
            with col:
                st.markdown("### " + title)
                chart(
                    alt.Chart(scale)
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("num_nodes:O", title="Jumlah koridor"),
                        y=alt.Y(field + ":Q", title=None, scale=alt.Scale(zero=True)),
                        color=model_color("model:N"),
                        tooltip=["model:N", "num_nodes:O", alt.Tooltip(field + ":Q", format=".3f")],
                    ),
                    240,
                )
    st.markdown("### Benchmark training terpisah")
    st.caption(
        "Angka ini berasal dari run benchmark training, terpisah dari eksperimen scalability dan evaluasi lima seed."
    )
    st.dataframe(
        training.rename(
            columns={
                "model": "Model",
                "mean_sec_per_epoch": "Detik / epoch",
                "total_training_time_sec": "Total detik",
                "best_epoch_benchmark": "Epoch terbaik",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    download(scale, "skalabilitas-model.csv")


def render_trip(metadata, references, models):
    heading(
        "Dari satu koridor ke koridor lain.",
        "Lihat prediksi kedua titik perjalanan dan, bila diperlukan, geometri rute jalan.",
    )
    lookup = metadata.set_index("node_index")
    nodes = lookup.index.tolist()
    a, b = st.columns(2)
    with a:
        origin = st.selectbox(
            "Dari", nodes, format_func=lambda n: lookup.loc[n, "corridor_name"], key="trip_from"
        )
    with b:
        destination = st.selectbox(
            "Ke",
            nodes,
            index=1,
            format_func=lambda n: lookup.loc[n, "corridor_name"],
            key="trip_to",
        )
    a, b = st.columns(2)
    with a:
        sample = st.selectbox(
            "Waktu historis",
            range(len(references)),
            format_func=lambda i: wib(references[i]["reference_time"]),
            key="trip_time",
        )
    with b:
        model = st.selectbox("Model", models, key="trip_model_new")
    if origin == destination:
        st.info("Pilih dua koridor yang berbeda.")
        return
    pred = map_predictions(model, sample)
    points = metadata[metadata.node_index.isin([origin, destination])].copy()
    points["color"] = points.node_index.apply(
        lambda n: [40, 94, 232] if n == origin else [67, 167, 153]
    )
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=350,
            radius_min_pixels=9,
            pickable=True,
        )
    ]
    key = f"route_{origin}_{destination}"
    if st.button("Tampilkan rute jalan", type="primary"):
        try:
            with st.spinner("Mengambil geometri jalan…"):
                st.session_state[key] = get_route_between_corridors(
                    lookup.loc[origin], lookup.loc[destination], traffic=False
                )
        except Exception:
            st.warning(
                "Rute belum bisa dimuat. Periksa koneksi dan konfigurasi TomTom. Prediksi koridor tetap tersedia."
            )
    route = st.session_state.get(key)
    if route is not None:
        path = route["route_df"][["longitude", "latitude"]].values.tolist()
        layers.insert(
            0,
            pdk.Layer(
                "PathLayer",
                data=[{"path": path}],
                get_path="path",
                get_color=[40, 94, 232],
                get_width=4,
                width_min_pixels=4,
            ),
        )
        st.caption(
            f"Jarak rute TomTom: {route['distance_km']:.1f} km · Geometri saat diminta; bukan rekonstruksi lalu lintas historis."
        )
    else:
        st.caption(
            "Dua titik menunjukkan koridor asal dan tujuan. Tombol rute memerlukan koneksi TomTom."
        )
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            map_style=MAP_STYLE,
            initial_view_state=pdk.ViewState(
                latitude=float(points.latitude.mean()),
                longitude=float(points.longitude.mean()),
                zoom=10,
            ),
            tooltip={"text": "{corridor_name}"},
        ),
        height=400,
    )
    rows = pd.DataFrame(
        {
            "Horizon": ["+1 jam", "+3 jam", "+6 jam"],
            lookup.loc[origin, "corridor_name"]: pred[:, origin],
            lookup.loc[destination, "corridor_name"]: pred[:, destination],
        }
    )
    st.markdown("### Prediksi pada kedua koridor")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Kecepatan dalam km/jam. Prediksi dua koridor bukan prediksi waktu tempuh seluruh rute atau rekomendasi waktu berangkat terbaik."
    )
