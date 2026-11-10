"""Arus: traffic forecasting workspace and research evidence."""

# Ensure repository root is available for absolute package imports.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from streamlit_app.data_loader import get_corridor_metadata, get_reference_times
from streamlit_app.map_workspace import render_map_workspace
from streamlit_app.model_info import get_model_names
from streamlit_app.research_views import (
    render_compute,
    render_data,
    render_graph,
    render_models,
    render_prediction,
    render_trip,
)
from streamlit_app.theme import apply_theme, render_header

st.set_page_config(page_title="arus — Traffic explorer", page_icon="↗", layout="wide")
apply_theme()
metadata = get_corridor_metadata()
references = get_reference_times()
models = get_model_names()

page_options = [
    "🏠 Overview",
    "🗃️ Dataset Explorer",
    "🕸️ Graph Explorer",
    "📈 Traffic Forecasting",
    "📊 Model Comparison",
    "🗺️ Analisis Perjalanan",
    "⚙️ Computational & Scalability",
]

with st.container(key="photo-header"):
    render_header()

    with st.container(key="topbar"):
        brand, navigation, spacer = st.columns([1, 3, 1], vertical_alignment="center")
        with brand:
            st.markdown(
                '<div class="wordmark">arus<b>↗</b><small>TRAFFIC EXPLORER</small></div>',
                unsafe_allow_html=True,
            )
        with navigation:
            active_page = st.radio(
                "Navigasi",
                page_options,
                horizontal=True,
                key="main_navigation",
                format_func=lambda page: {
                    "🏠 Overview": "Peta",
                    "🗃️ Dataset Explorer": "Data",
                    "🕸️ Graph Explorer": "Graf",
                    "📈 Traffic Forecasting": "Prediksi",
                    "📊 Model Comparison": "Model",
                    "🗺️ Analisis Perjalanan": "Perjalanan",
                    "⚙️ Computational & Scalability": "Komputasi",
                }[page],
                label_visibility="collapsed",
            )


if active_page == "🏠 Overview":
    render_map_workspace(metadata, references, models)
elif active_page == "🗃️ Dataset Explorer":
    render_data(metadata)
elif active_page == "🕸️ Graph Explorer":
    render_graph(metadata)
elif active_page == "📈 Traffic Forecasting":
    render_prediction(metadata, references, models)
elif active_page == "📊 Model Comparison":
    render_models()
elif active_page == "🗺️ Analisis Perjalanan":
    render_trip(metadata, references, models)
elif active_page == "⚙️ Computational & Scalability":
    render_compute()
