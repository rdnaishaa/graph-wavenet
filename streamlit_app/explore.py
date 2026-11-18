"""Public map explorer. Calendar scenarios are synthetic, never observations."""
from html import escape
import time
import base64
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_app.data_loader import ROOT, get_corridor_metadata, load_scaler
from streamlit_app.map_workspace import map_predictions, wib


@st.cache_data(show_spinner=False)
def typical_window(reference):
    """Observed training values only; weekday/hour -> hour -> node fallback."""
    meta = get_corridor_metadata()
    df = pd.read_csv(ROOT / 'data/final_64/forecasting_train_preprocessed.csv')
    df = df[df.is_observed.astype(str).str.lower().eq('true')].copy()
    times = pd.to_datetime(df.obs_time_utc, utc=True).dt.tz_convert('Asia/Jakarta')
    df['weekday'], df['hour'] = times.dt.dayofweek, times.dt.hour
    detailed = df.groupby(['corridor_file', 'weekday', 'hour']).current_speed.mean()
    hourly = df.groupby(['corridor_file', 'hour']).current_speed.mean()
    fallback = df.groupby('corridor_file').current_speed.mean()
    timeline = pd.date_range(end=pd.Timestamp(reference), periods=12, freq='h')
    values = np.empty((12, len(meta), 1), dtype=np.float32)
    for t, timestamp in enumerate(timeline):
        for n, corridor in enumerate(meta.corridor_file):
            values[t, n, 0] = detailed.get((corridor, timestamp.dayofweek, timestamp.hour),
                hourly.get((corridor, timestamp.hour), fallback.get(corridor, np.nan)))
    if not np.isfinite(values).all():
        raise ValueError('Riwayat training belum cukup untuk membentuk profil semua koridor.')
    return values


@st.cache_data(show_spinner=False)
def scenario_prediction(reference, multiplier):
    from streamlit_app.model_loader import load_graphwavenet_model
    history = typical_window(reference) * multiplier
    scaler = load_scaler()
    x = (history - scaler['mean']) / scaler['scale']
    prediction = load_graphwavenet_model('hybrid')(x[None], training=False).numpy()[0]
    return prediction * scaler['scale'] + scaler['mean'], history


@st.fragment(run_every='1s')
def live_clock():
    now = pd.Timestamp(datetime.now(ZoneInfo('Asia/Jakarta')))
    if st.session_state.get('clock_reference_hour') != now.floor('h').isoformat():
        st.rerun()
    st.markdown(f'<div style="background:#dff2ec;color:#1e6256;padding:10px 12px;border-radius:10px;font-weight:650">◷ {now.strftime("%H:%M:%S")} WIB <small>· Jam sekarang</small></div>',unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def road_route(origin, destination):
    from streamlit_app.routing_utils import get_alternative_routes
    metadata = get_corridor_metadata().set_index('node_index')
    return get_alternative_routes(metadata.loc[origin], metadata.loc[destination])


def estimate_route_minutes(route, metadata, speeds):
    coords = route['route_df'][['latitude','longitude']].to_numpy(float)
    mid = (coords[:-1]+coords[1:])/2
    scale = np.array([111.195, 111.195*np.cos(np.deg2rad(coords[:,0].mean()))])
    lengths = np.linalg.norm(np.diff(coords,axis=0)*scale,axis=1)
    if lengths.sum() <= 0:
        raise ValueError('Panjang rute tidak valid')
    nodes = metadata[['latitude','longitude']].to_numpy(float)
    nearest = np.argmin(np.sum(((mid[:,None,:]-nodes[None,:,:])*scale)**2,axis=2),axis=1)
    segment_speeds = np.asarray(speeds)[nearest]
    if np.any(segment_speeds <= 0):
        raise ValueError('Kecepatan prediksi tidak valid')
    # Rescale polyline lengths to the provider's road distance.
    return float(np.sum(lengths / lengths.sum() * route['distance_km'] / segment_speeds) * 60)


def render_explorer(metadata, references):
    st.markdown('''<style>
    .block-container {max-width:none;padding:0}
    .st-key-explorer-canvas [data-testid="stDeckGlJsonChart"] {height:100dvh!important}
    .st-key-explorer-canvas [data-testid="stDeckGlJsonChart"]>div {height:100%!important}
    .explore-top {position:fixed;top:12px;right:60px;z-index:8;background:#ffffffeb;border-radius:16px;padding:9px 14px!important;box-shadow:0 3px 15px #183f4815}
    .explore-legend {position:fixed;bottom:20px;right:15px;z-index:8;background:#fffffff0;padding:9px 12px!important;border-radius:10px}

    [data-testid="stSidebar"] {background:#fff;border-right:1px solid #e7e9ec;min-width:330px;max-width:360px}
    [data-testid="stSidebarUserContent"] {padding:1rem 1.25rem}
    [data-testid="stSidebarContent"] {padding-top:0}
    [data-testid="stSidebarHeader"] {height:1rem;min-height:1rem;padding:0}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.65rem}
    [data-testid="stHeader"] {background:transparent;height:0}
    [data-testid="stSidebar"] {background:#fffcf5;border-right:1px solid #e9dfc9}
    [data-testid="stSidebar"] label p {font-size:14px!important;font-weight:600;color:#263e46}
    [data-testid="stSidebar"] [data-testid="stButton"] button {background:#183f48;color:white;border:0;border-radius:12px;min-height:48px;font-weight:650}
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {background:#236273;color:white}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:4px;flex-wrap:wrap}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:8px 6px;border:1px solid #e4ddca;border-radius:10px;background:white;margin:0}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#ffdb69;border-color:#edba31;color:#253e43}
    [data-testid="stSidebar"] [data-testid="stExpander"] {background:#fff;border:1px solid #e8e1d3;border-radius:12px}
    [data-testid="stSidebar"] [role="radiogroup"] label p {font-size:12px!important}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {background:#fff0df;border-radius:11px;color:#865322}
    .nav-section {font-size:12px;font-weight:750;letter-spacing:.06em;color:#608084;margin:12px 0 0}
    .nav-map-active {background:#ffdb69;color:#233e44;border:1px solid #efc64b;border-radius:12px;padding:13px 15px;font-size:16px;font-weight:700;margin-bottom:8px}
    .nav-map-active span {float:right;background:#fff7d9;padding:2px 8px;border-radius:20px;font-size:11px}
    .explore-brand {font-size:34px;font-weight:850;letter-spacing:-2px;color:#17272d}
    .explore-brand span {color:#e29b27}.explore-tag {font-size:11px;letter-spacing:2px;color:#76838b;margin-bottom:20px}
    .explore-top {display:flex;align-items:center;justify-content:space-between;padding:5px 5px 14px;gap:12px}
    .explore-top strong {font-size:20px;letter-spacing:-.5px}.explore-badge {background:#fff0c7;color:#735218;border-radius:20px;padding:6px 12px;font-size:12px}
    .explore-readout {border-top:1px solid #e7e9ec;margin-top:18px;padding-top:16px}
    .explore-readout h2 {font-size:23px;margin:0}.explore-speed {font-size:48px;font-weight:750;letter-spacing:-2px}.explore-speed small {font-size:13px;font-weight:400;letter-spacing:0;color:#75818a}
    .explore-legend {font-size:12px;color:#65767d;padding:8px 0}.explore-legend b {padding:0 7px}
    [data-testid="stSidebar"] [data-testid="stMetric"] {background:transparent;border:0;padding:0}
    @media(max-width:760px) {.block-container{padding:3rem .5rem .5rem}.explore-top strong{font-size:17px}}
    /* Editorial accents: tilted backing, readable upright controls. */
    [data-testid="stAppViewContainer"] {background:#f1eee5}
    [data-testid="stSidebar"] {margin:14px 0 14px 14px;border:1px solid #ddd6c5;border-radius:22px;box-shadow:5px 5px 0 #d7d3c6;overflow:hidden;min-width:344px;max-width:360px}
    [data-testid="stSidebarUserContent"] {padding:1.1rem 1.4rem 2rem}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:1rem}
    .explore-brand {display:inline-block;position:relative;isolation:isolate;padding:2px 10px;margin:8px 0 8px}
    .explore-brand:before {content:"";position:absolute;inset:5px -4px 0;background:#ffda65;transform:rotate(-5deg);z-index:-1;border-radius:3px}
    .explore-brand span {color:#c55731}
    .explore-tag {margin:5px 0 22px!important;line-height:1.6}
    .nav-map-active {position:relative;isolation:isolate;background:transparent;border:0;margin:4px 2px 14px;padding:15px 14px;box-shadow:none}
    .nav-map-active:before {content:"";position:absolute;inset:0;background:#ffda65;border:1px solid #e6b63c;border-radius:6px;transform:none;z-index:-1;box-shadow:3px 3px 0 #253e43}
    [data-testid="stSidebar"] [data-testid="stButton"] button {margin:2px 0 10px;min-height:48px;box-shadow:3px 3px 0 #c6d5ce;border-radius:9px}
    .nav-section {display:inline-block;position:relative;isolation:isolate;padding:6px 9px;margin:16px 0 4px;color:#62432c;letter-spacing:.08em}
    .nav-section:before {content:"";position:absolute;inset:0;background:#ffd1ae;transform:rotate(-2deg);z-index:-1;border-radius:2px}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:6px;padding:4px 0}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 6px;border-radius:8px}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#ffdf77;box-shadow:2px 2px 0 #d0a952}
    [data-testid="stSidebar"] [data-testid="stExpander"] {margin:5px 0}
    .explore-readout {margin-top:24px;padding:22px 4px 8px}
    .explore-readout h2 {display:inline-block;position:relative;isolation:isolate;padding:0 6px}
    .explore-readout h2:before {content:"";position:absolute;left:0;right:0;bottom:0;height:45%;background:#bce4d5;transform:none;z-index:-1}
    .explore-speed {margin-top:12px}
    .explore-top {top:24px;right:65px;padding:13px 20px!important;border:1px solid #e6ddc9;box-shadow:4px 4px 0 #d8cfb6;gap:18px;border-radius:12px}
    .explore-top strong {font-size:20px;line-height:1.5}
    .explore-badge {transform:none;border-radius:4px;background:#ffdc6c;padding:7px 11px}
    .explore-legend {bottom:24px;right:24px;padding:12px 18px!important;border:1px solid #ded7c8;box-shadow:3px 3px 0 #d6d0c3;line-height:1.6;max-width:calc(100vw - 420px)}
    .st-key-explorer-canvas {margin:14px 14px 14px 12px;border-radius:20px;overflow:hidden;border:1px solid #ded9ca}
    .st-key-explorer-canvas [data-testid="stDeckGlJsonChart"] {height:calc(100dvh - 28px)!important}
    @media(max-width:760px) {
      [data-testid="stSidebar"] {margin:8px;border-radius:16px;min-width:300px;max-width:330px}
      .explore-top {top:52px;right:20px;left:20px;gap:8px;padding:8px 12px!important}
      .explore-top strong{font-size:15px}.explore-badge{font-size:10px}
      .explore-legend{max-width:calc(100vw - 40px);font-size:10px;bottom:16px;right:20px}
      .st-key-explorer-canvas {margin:8px}
    }
    .nav-map-active {margin-bottom:22px}
    .explore-top {right:72px;top:18px;max-width:calc(100vw - 470px)}
    .st-key-explorer-canvas {margin:0;border:0;border-radius:0}
    .st-key-explorer-canvas [data-testid="stDeckGlJsonChart"] {height:100dvh!important}
    [data-testid="stMainBlockContainer"] {padding:0!important}
    .explore-legend {position:fixed;bottom:18px;right:18px;z-index:50}
    .st-key-explorer-canvas [data-stale="true"] {opacity:1!important}
    [data-testid="stToolbar"] {display:none}
    .st-key-explorer-canvas [data-testid="stDeckGlJsonChart"] {
      position:fixed!important;inset:0!important;width:100vw!important;height:100dvh!important;
      max-width:none!important;border-radius:0!important;z-index:0;
    }
    .st-key-explorer-canvas [data-testid="stDeckGlJsonChart"]>div {position:absolute!important;inset:0!important;width:100%!important;height:100%!important}
    [data-testid="stSidebar"] {z-index:100!important;position:fixed!important;left:0;top:0;bottom:0;height:calc(100dvh - 28px)!important}
    .explore-top,.explore-legend {z-index:101!important}
    @media(max-width:760px) {
      [data-testid="stSidebar"] {position:fixed!important;top:auto!important;bottom:0!important;left:0!important;right:0!important;width:100vw!important;min-width:0!important;max-width:100vw!important;height:43dvh!important;max-height:43dvh!important;margin:0!important;border-radius:22px 22px 0 0!important;transform:none!important;box-shadow:0 -4px 22px #16394520!important;z-index:100!important}
      [data-testid="stSidebar"]:before {content:"";display:block;width:38px;height:4px;background:#b4b9b5;border-radius:9px;margin:10px auto 0}
      [data-testid="stSidebarHeader"] {display:none!important}
      [data-testid="stSidebarUserContent"] {padding:12px 20px 32px!important}
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.7rem!important}
      .explore-brand {font-size:27px;margin:0 0 4px}
      .explore-tag {margin-bottom:14px!important}
      .nav-section {margin-top:10px}
      .explore-top {left:12px;right:54px;top:12px!important;max-width:none!important;padding:8px 10px!important;gap:7px;border-radius:10px}
      .explore-top strong {font-size:13px;line-height:1.35}
      .explore-badge {font-size:9px;padding:5px 6px;white-space:nowrap}
      .explore-legend {left:10px;right:10px;bottom:calc(43dvh + 10px)!important;max-width:none!important;padding:6px 9px!important;font-size:9px;line-height:1.4;border-radius:8px}
      [data-testid="stMain"] {margin-left:0!important}
    }
    </style>''', unsafe_allow_html=True)
    names = metadata.set_index('node_index').corridor_name.to_dict()
    graph = np.load(ROOT / 'outputs/final_64/fixed_graph/fixed_graph_64.npz')['directed_knn_adjacency']
    with st.sidebar:
        st.markdown('<div class="explore-brand">arus<span>↗</span></div><div class="explore-tag" style="letter-spacing:.02em;font-size:13px;color:#56757b">Jelajahi lalu lintas Jabodetabek</div><div class="nav-map-active">↗ &nbsp; Peta & simulasi <span>Aktif</span></div>', unsafe_allow_html=True)
        if st.button('Buka dashboard penelitian ↗', use_container_width=True):
            st.session_state['app_surface'] = 'dashboard'
            st.rerun()
        st.markdown('<div class="nav-section">01 / SUMBER DATA</div>', unsafe_allow_html=True)
        mode = st.radio('Sumber simulasi', ['Pola historis', 'Putar data uji'], horizontal=True)
        st.markdown('<div class="nav-section">02 / ASAL & TUJUAN</div>', unsafe_allow_html=True)
        node = st.selectbox('Dari koridor', list(names), format_func=names.get, key='explore_node')
        neighbors = [int(n) for n in names if n != node]
        if st.session_state.get('explore_destination') not in neighbors:
            st.session_state['explore_destination'] = neighbors[0]
        destination = st.selectbox('Ke koridor', neighbors, format_func=names.get, key='explore_destination')
        st.caption('Semua koridor tersedia · simulasi dari pola historis.')
        if mode == 'Pola historis':
            now = datetime.now(ZoneInfo('Asia/Jakarta'))
            reference = pd.Timestamp(now).floor('h')
            st.session_state['clock_reference_hour'] = reference.isoformat()
            live_clock()
            with st.expander('Atur kondisi masukan'):
                factor = st.slider('Kecepatan relatif terhadap pola biasa (%)', 50, 150, 100, step=10)
                st.caption('Mengubah seluruh masukan 12 jam, bukan mengubah hasil prediksi secara langsung.')
            with st.spinner('Menghitung skenario…'):
                predictions, history = scenario_prediction(reference.isoformat(), factor / 100)
            source = 'Simulasi · pola historis'
        else:
            sample = st.selectbox('Waktu historis', range(len(references)), format_func=lambda i: wib(references[i]['reference_time']))
            reference = pd.Timestamp(references[sample]['reference_time'])
            reference = reference.tz_localize('UTC') if reference.tzinfo is None else reference
            predictions = map_predictions('hybrid', sample)
            source = 'Simulasi · data uji'
        st.markdown('<div class="nav-section">03 / PREDIKSI & PETA</div>', unsafe_allow_html=True)
        horizon = st.radio('Lihat ke depan', [1, 3, 6], format_func=lambda h:f'+{h} jam', horizontal=True)
        edges_on = st.toggle('Hubungan antarkoridor', value=True)
        focus = st.toggle('Fokus koridor terpilih', value=False)
        hidx = [1, 3, 6].index(horizon)
        departure = pd.Timestamp(datetime.now(ZoneInfo('Asia/Jakarta'))) if mode == 'Pola historis' else reference
        target = reference + pd.Timedelta(hours=horizon)
        trip_departure = departure + pd.Timedelta(hours=horizon)
        st.markdown(f'<div class="explore-readout"><h2>{escape(names[node])}</h2><div class="explore-speed">{predictions[hidx,node]:.1f} <small>km/jam</small></div></div>', unsafe_allow_html=True)
        st.caption(f'Prediksi Hybrid Graph WaveNet · {wib(target)}')
        with st.expander('Tiga horizon prediksi', expanded=True):
            for h, speed in zip([1,3,6], predictions[:,node]):
                st.write(f'**+{h} jam**  ·  {speed:.1f} km/jam')
        with st.expander('Dari mana angkanya?'):
            st.write('Tanggal bebas memakai rata-rata observasi training pada hari dan jam yang sama. Jika tidak tersedia, dipakai rata-rata jam, lalu rata-rata koridor. Profil 12 jam ini menjadi masukan model Hybrid yang sudah dilatih. Tanggal tidak dibaca langsung oleh model.')
            st.caption('Bukan data aktual, bukan prakiraan operasional hari ini. Garis adalah relasi graf geografis, bukan rute jalan atau arah perjalanan.')
        if st.button('Hitung perjalanan →', use_container_width=True):
            st.session_state.pop('explore_routes_v2', None)
            if node == destination:
                st.info('Asal dan tujuan sama. Pilih koridor tujuan lain.')
            else:
                try:
                    with st.spinner('Mencari rute jalan…'):
                        route = road_route(node, destination)
                    st.session_state['explore_routes_v2'] = (node, destination, route)
                except Exception:
                    st.warning('Rute jalan belum tersedia. Periksa koneksi dan konfigurasi TomTom. Estimasi durasi belum dapat dihitung.')
        saved_route = st.session_state.get('explore_routes_v2')
        route = saved_route[2] if saved_route and saved_route[:2] == (node, destination) else None
        alternatives = route or []
        route = None
        if alternatives:
            comparisons = []
            for index, candidate in enumerate(alternatives):
                minutes = estimate_route_minutes(candidate, metadata, predictions[hidx])
                comparisons.append((minutes, index, candidate))
            comparisons.sort(key=lambda item: item[0])
            best_minutes = comparisons[0][0]
            st.success(f'Tercepat dalam simulasi: Rute {comparisons[0][1]+1} · {best_minutes:.0f} menit')
            chosen = st.radio('Bandingkan rute', [entry[1] for entry in comparisons],
                format_func=lambda i: f"Rute {i+1} · {next(m for m,j,c in comparisons if j==i):.0f} menit", key=f'route_choice_{node}_{destination}')
            route = alternatives[chosen]
            for minutes, index, candidate in comparisons:
                via = ' → '.join(candidate['streets'][:4]) or 'Ikuti jalur berwarna di peta'
                extra = minutes - best_minutes
                st.markdown(f"**Rute {index+1} · {candidate['distance_km']:.1f} km** — " + ('tercepat' if extra < .01 else f'+{extra:.1f} menit'))
                st.caption(via)
            duration = next(m for m,j,c in comparisons if j==chosen)
            arrival = trip_departure + pd.Timedelta(minutes=duration)
            st.write(f"Berangkat **{trip_departure.strftime('%H:%M')}** → tiba **{arrival.strftime('%d %b, %H:%M WIB')}**")
            if len(alternatives)==1:
                st.caption('Penyedia hanya mengembalikan satu rute; belum ada alternatif untuk dibandingkan.')
            st.caption('Estimasi simulasi memakai prediksi koridor terdekat pada setiap segmen rute. Koridor bukan sensor setiap ruas; lampu merah dan kejadian aktual belum dimodelkan.')
    points = metadata.copy()
    points['speed'] = np.round(predictions[hidx],1)
    points['color'] = points.speed.map(lambda s:[211,104,83] if s<20 else [224,171,47] if s<40 else [45,146,128])
    selected = points[points.node_index == node]
    graph = np.load(ROOT / 'outputs/final_64/fixed_graph/fixed_graph_64.npz')['directed_knn_adjacency']
    with st.sidebar.expander('Koridor yang terhubung'):
        st.caption('Tetangga keluar pada fixed graph geografis.')
        for neighbor in np.flatnonzero(graph[node] > 0):
            st.write(f'**{names[int(neighbor)]}** · {predictions[hidx,neighbor]:.1f} km/jam')
    links=[]
    for i,j in zip(*np.where(graph>0)):
        a,b=points.iloc[i],points.iloc[j]
        links.append(dict(source=[a.longitude,a.latitude],target=[b.longitude,b.latitude],
            color=[235,124,44,255] if i==node and j==destination else [36,113,130,210] if i==node else [116,139,148,65],
            width=6 if i==node and j==destination else 3 if i==node else 1))
    layers=[]
    for alternative in alternatives:
        if alternative is not route:
            layers.append(pdk.Layer('PathLayer',data=[{'path':alternative['route_df'][['longitude','latitude']].values.tolist()}],get_path='path',get_color=[146,119,167,170],get_width=4,width_min_pixels=3))
    if route:
        layers.append(pdk.Layer('PathLayer',data=[{'path':route['route_df'][['longitude','latitude']].values.tolist()}],get_path='path',get_color=[27,113,173],get_width=5,width_min_pixels=4))
    destination_point = points[points.node_index == destination]
    layers.append(pdk.Layer('ScatterplotLayer',data=destination_point,get_position='[longitude, latitude]',get_fill_color=[240,137,55],get_radius=240,radius_min_pixels=10,stroked=True,get_line_color=[255,255,255],line_width_min_pixels=2))
    layers.append(pdk.Layer('TextLayer',data=destination_point,get_position='[longitude, latitude]',get_text='corridor_name',get_pixel_offset=[0,-28],get_size=14,get_color=[135,65,25],background=True,get_background_color=[255,245,230,245]))

    if edges_on:
        layers.append(pdk.Layer('LineLayer',data=links,get_source_position='source',get_target_position='target',get_color='color',get_width='width'))
    layers.extend([
        pdk.Layer('ScatterplotLayer',id='explore-corridors',data=points,get_position='[longitude, latitude]',get_fill_color='color',get_radius=150,radius_min_pixels=5,stroked=True,get_line_color=[255,255,255],line_width_min_pixels=1.5,pickable=True),
        pdk.Layer('ScatterplotLayer',data=selected,get_position='[longitude, latitude]',get_fill_color=[29,76,99,45],get_radius=450,radius_min_pixels=19),
        pdk.Layer('TextLayer',data=selected,get_position='[longitude, latitude]',get_text='corridor_name',get_size=14,get_pixel_offset=[0,-28],get_color=[22,49,61],background=True,get_background_color=[255,255,255,245],background_padding=[8,5])])
    center = selected.iloc[0] if focus else points[['latitude','longitude']].median()
    st.markdown(f'<div class="explore-top"><strong>{escape(names[node])} → {escape(names[destination])}</strong><span class="explore-badge">{source}</span></div>', unsafe_allow_html=True)
    def pick_node():
        objects=st.session_state['explore_map'].selection.get('objects',{}).get('explore-corridors',[])
        if objects:
            st.session_state['explore_node']=int(objects[0]['node_index'])
    @st.fragment(run_every='2s' if route else None)
    def draw_map():
        animated_layers = list(layers)
        if route:
            path = route['route_df'][['longitude','latitude']].to_numpy(float)
            lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cumulative = np.r_[0, np.cumsum(lengths)]
            distance = (time.time() % 60) / 60 * cumulative[-1]
            position = [float(np.interp(distance,cumulative,path[:,axis])) for axis in [0,1]]
            svg = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="40"><rect x="2" y="10" width="60" height="22" rx="8" fill="#f4b633" stroke="#243d48" stroke-width="3"/><path d="M16 11L23 2H43L50 11" fill="#69b9cc" stroke="#243d48" stroke-width="3"/><circle cx="16" cy="32" r="6" fill="#243d48"/><circle cx="49" cy="32" r="6" fill="#243d48"/></svg>'
            icon = 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()
            animated_layers.append(pdk.Layer('IconLayer',data=[{'position':position,'icon':{'url':icon,'width':64,'height':40,'anchorY':20}}],get_position='position',get_icon='icon',get_size=36,pickable=False))
        with st.container(key='explorer-canvas'):
            st.pydeck_chart(pdk.Deck(layers=animated_layers,initial_view_state=pdk.ViewState(latitude=float(center.latitude),longitude=float(center.longitude),zoom=12 if focus else 9.7),map_style='https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',tooltip={'html':'<b>{corridor_name}</b><br>{speed} km/jam · simulasi'}),height=900,key='explore_map',on_select=pick_node)
    draw_map()
    if route:
        st.sidebar.caption('Mobil mengulang rute dalam 60 detik untuk ilustrasi. Bukan posisi kendaraan aktual atau durasi perjalanan sebenarnya.')
    st.markdown('<div class="explore-legend"><b style="color:#d36853">●</b> &lt;20 <b style="color:#e0ab2f">●</b> 20–&lt;40 <b style="color:#2d9280">●</b> ≥40 km/jam · Klik titik untuk memilih koridor · Garis = relasi graf, bukan rute</div>',unsafe_allow_html=True)
    st.sidebar.caption(f'64 koridor · {wib(target)} · Rata-rata prediksi jaringan {predictions[hidx].mean():.1f} km/jam')
