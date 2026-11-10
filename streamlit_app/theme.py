"""Responsive navigation and map workspace styling."""

import base64
from pathlib import Path

import streamlit as st


def apply_theme():
    st.markdown(
        """<style>
    :root {--ink:#18232f;--muted:#79838d;--blue:#285ee8;--line:#e6e9ed}
    .stApp {background:#fff;color:var(--ink)}
    .block-container {max-width:1680px;padding:3.4rem 24px 28px}
    [data-testid="stHeader"] {background:transparent;height:2.4rem}
    h1,h2,h3,h4 {font-family:Arial,sans-serif!important;color:var(--ink);letter-spacing:-.035em}
    h1 {font-size:1.9rem!important} h3 {font-size:1.2rem!important}
    .st-key-topbar {border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:6px}
    .wordmark {font-size:27px;font-weight:750;letter-spacing:-1.5px;line-height:42px;white-space:nowrap}
    .wordmark b {color:var(--blue)}
    .wordmark small {font-size:10px;letter-spacing:.1em;font-weight:500;color:var(--muted);margin-left:14px}
    .st-key-topbar [data-testid="stRadio"] {width:100%!important}
    .st-key-topbar [data-testid="stRadio"] > div {width:100%!important}
    .st-key-topbar [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {width:100%!important;align-self:stretch}
    .st-key-topbar [role="radiogroup"] {width:100%;justify-content:center!important;gap:4px;flex-wrap:wrap}
    .st-key-topbar [role="radiogroup"] label {padding:9px 13px;border-radius:7px;margin:0!important;min-height:40px}
    .st-key-topbar [role="radiogroup"] label:has(input:checked) {background:#f0f4fe;color:var(--blue)}
    .st-key-topbar [role="radiogroup"] label [data-testid="stMarkdownContainer"] p {font-size:13px;font-weight:500}
    .st-key-topbar [role="radiogroup"] label>div>div:first-child:not([data-testid="stMarkdownContainer"]) {display:none!important}
    .jakarta-banner {position:relative;height:156px;overflow:hidden;border-radius:12px;margin:0 0 4px;background:#263747}
    .jakarta-banner img {width:100%;height:100%;object-fit:cover;object-position:center 46%;display:block}
    .jakarta-banner::after {content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(12,23,35,.3),rgba(12,23,35,.55),rgba(12,23,35,.3))}
    .jakarta-banner-caption {position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:20px;z-index:1;color:white}
    .jakarta-banner-caption span {display:block;font-size:10px;letter-spacing:.16em;margin-bottom:8px;color:#eee}
    .jakarta-banner-caption strong {font-size:23px;font-weight:500;letter-spacing:-.5px}
    .st-key-photo-header {position:relative;border-radius:12px;overflow:hidden;margin-bottom:14px}
    .st-key-photo-header .jakarta-banner {height:190px;margin:0;border-radius:0}
    .st-key-photo-header .jakarta-banner::after {background:rgba(8,17,27,.68)}
    .st-key-photo-header .jakarta-banner-caption {padding-bottom:65px}
    .st-key-photo-header .st-key-topbar {position:absolute!important;bottom:0;left:0;right:0;width:100%;padding:10px 22px;margin:0;border-bottom:0;border-top:1px solid #ffffff25;background:#0a142338;z-index:2}
    .st-key-photo-header .wordmark,.st-key-photo-header .wordmark b {color:#fff}
    .st-key-photo-header .wordmark small {color:#d5dce5}
    .st-key-photo-header [role="radiogroup"] label {color:#e7edf5}
    .st-key-photo-header [role="radiogroup"] label:has(input:checked) {background:#ffffff24;color:#fff}
    .st-key-photo-header [role="radiogroup"] label:hover {background:#ffffff17}
    .ink-highlight {background:#f6d94b;color:#172332;padding:.03em .22em;box-decoration-break:clone;-webkit-box-decoration-break:clone}
    .hero-title {font-size:32px!important;font-weight:750!important;letter-spacing:-1px!important;line-height:1.55}
    .hero-title .ink-highlight {display:inline-block;transform:rotate(-2deg);padding:0 .24em;box-shadow:3px 3px 0 #172332}
    .panel-heading .ink-highlight {background:linear-gradient(transparent 48%,#f6d94b 48%,#f6d94b 94%,transparent 94%);padding:0 .08em}
    .page-heading h1 {display:inline;line-height:1.5;background:linear-gradient(transparent 64%,#f6d94b 64%,#f6d94b 94%,transparent 94%);box-decoration-break:clone;-webkit-box-decoration-break:clone}
    .page-heading p {margin-top:12px!important}
    .st-key-photo-header [role="radiogroup"] label:has(input:checked) {background:#f6d94b;color:#172332;border-radius:2px;box-shadow:2px 2px 0 #172332}
    @media(max-width:760px) {.hero-title {font-size:24px!important}}
    /* Poster accents across the research views; traffic colors retain their meaning. */
    .page-heading h1 {background:linear-gradient(transparent 62%,#ffd564 62%,#ffd564 94%,transparent 94%)}
    .page-heading p {border-left:3px solid #ee874a;padding-left:12px}
    [data-testid="stMetric"] {border-left:3px solid #ef914f!important;background:#fff8ec!important;border-radius:2px 10px 10px 2px;padding:12px 18px!important}
    [data-testid="stMetricValue"] {font-weight:650}
    .location-strip strong {background:#f6d94b;padding:3px 7px;box-shadow:2px 2px 0 #eab570}
    .archive-tag {background:#fff1df;border-color:#f3d5b3;color:#8c5128!important}
    .summary-title {width:fit-content;background:linear-gradient(transparent 60%,#ffd6b6 60%,#ffd6b6 95%,transparent 95%)}
    .forecast-readout .place {display:inline-block;background:#f6d94b;padding:2px 6px}
    .forecast-readout .number {color:#ad501e}
    .process-list b {display:flex;align-items:center;justify-content:center;align-self:flex-start;min-width:32px;height:32px;background:#f6d94b;color:#24303d;font-weight:700;transform:rotate(-4deg);border-radius:2px}
    .process-list>div:nth-child(2) b {background:#ffc28e;transform:rotate(3deg)}
    .process-list>div:nth-child(3) b {background:#dbe8fa;transform:rotate(-2deg)}
    .process-list strong {width:fit-content;background:linear-gradient(transparent 65%,#fff0ad 65%)}
    .connection-row b {background:#fff0d6;color:#945018;padding:2px 7px;border-radius:3px}
    .research-note {background:#fff5e6;border-left:4px solid #ed8c43;color:#795337}
    .research-note b {width:fit-content;background:#ffd599;color:#57391f;padding:2px 5px;margin-bottom:6px}
    [data-testid="stDownloadButton"] button {background:#f6d94b;border:1px solid #e0c242;color:#24303d;font-weight:600;box-shadow:2px 2px 0 #24303d}
    [data-testid="stDownloadButton"] button:hover {background:#ffc28e;border-color:#d78a48;color:#24303d}
    .st-key-map-panel [role="radiogroup"] label:has(input:checked) {background:#ffecd8;border-color:#efb67c;color:#914519}
    .location-strip {display:flex;align-items:center;justify-content:space-between;padding:6px 2px 14px;gap:12px}
    .location-strip strong {font-size:15px;font-weight:600}
    .location-strip span {color:var(--muted);font-size:12px;margin-left:10px}
    .archive-tag {font-size:11px!important;border:1px solid var(--line);padding:5px 9px;border-radius:5px;white-space:nowrap}
    .st-key-map-stage {position:relative;border:1px solid #dfe4e9;border-radius:14px;overflow:hidden;background:#f1f3f4}
    .st-key-map-stage>div {gap:0!important}
    .st-key-map-panel {position:absolute!important;top:18px;left:18px;width:310px!important;z-index:5;
       background:#fff;border:1px solid #e4e7ec;border-radius:12px;padding:20px;
       box-shadow:0 4px 24px #26354718}
    .st-key-map-panel [data-testid="stVerticalBlock"] {gap:12px}
    .panel-heading {font-size:20px;font-weight:650;letter-spacing:-.6px;margin-bottom:0}
    .panel-hint {font-size:12px;color:var(--muted);line-height:1.5;margin-top:4px;margin-bottom:2px}
    [data-baseweb="select"]>div {background:#f6f7f9!important;border-color:#edf0f3!important;border-radius:8px!important}
    [data-testid="stWidgetLabel"] p {font-size:11px!important;color:#687583}
    .st-key-map-panel [role="radiogroup"] {gap:5px;justify-content:space-between}
    .st-key-map-panel [role="radiogroup"] label {border:1px solid var(--line);border-radius:7px;padding:6px 12px;margin:0!important;flex:1;justify-content:center}
    .st-key-map-panel [role="radiogroup"] label:has(input:checked) {background:#edf3ff;border-color:#c8d7fd;color:var(--blue)}
    .st-key-map-panel [role="radiogroup"] label>div>div:first-child:not([data-testid="stMarkdownContainer"]) {display:none!important}
    .st-key-map-panel [role="radiogroup"] label p {font-size:12px!important}
    .forecast-readout {margin-top:4px;padding-top:16px;border-top:1px solid var(--line)}
    .forecast-readout .place {font-size:13px;font-weight:600}
    .forecast-readout .number {font-size:48px;letter-spacing:-2px;line-height:1.2;font-weight:600;margin:6px 0;color:var(--ink)}
    .forecast-readout .number small {font-size:13px;font-weight:400;letter-spacing:0;color:var(--muted);margin-left:5px}
    .forecast-readout .when {font-size:11px;color:var(--muted);line-height:1.5}
    .comparison-inline {font-size:11px;color:#5d6b79;padding-top:7px}
    .st-key-map-panel [data-testid="stCaptionContainer"] {font-size:10px;color:var(--muted)}
    .map-legend {display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:10px 4px;font-size:11px;color:#6f7b86}
    .map-legend i {display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
    .map-legend .map-tip {margin-left:auto;color:#9ba3aa}
    .summary-title {font-size:18px;font-weight:600;letter-spacing:-.4px;margin:15px 0 3px}
    .summary-subtitle {font-size:12px;color:var(--muted);margin-bottom:18px}
    [data-testid="stMetric"] {padding:4px 18px;border-left:1px solid var(--line);background:transparent}
    [data-testid="stMetricValue"] {font-size:23px;font-weight:500;color:var(--ink)}
    [data-testid="stMetricLabel"] {color:var(--muted);font-size:12px}
    [data-testid="stDeckGlJsonChart"] {border-radius:0;overflow:hidden}
    button {border-radius:7px!important}
    hr {border-color:var(--line);margin:16px 0}
    .page-heading {padding:16px 0 22px}
    .page-heading h1 {font-size:29px!important;font-weight:600!important;margin:0;padding-bottom:7px}
    .page-heading p {font-size:14px;color:#7b8690;margin:0;max-width:780px}
    h3 {font-weight:550!important;margin-top:20px!important}
    .connection-row {display:flex;justify-content:space-between;padding:13px 0;border-bottom:1px solid var(--line);font-size:13px;gap:8px}
    .connection-row b {font-weight:500;font-variant-numeric:tabular-nums;color:#285ee8}
    .process-list>div {display:flex;gap:18px;padding:16px 0;border-bottom:1px solid var(--line)}
    .process-list b {font-size:12px;color:#a4afbb;font-weight:400;padding-top:2px}
    .process-list span {color:#798592;font-size:12px;line-height:1.7}
    .process-list strong {display:block;color:#233243;font-weight:500;font-size:14px}
    .research-note {border-left:2px solid #d5a65b;background:#fcfaf5;padding:16px 20px;color:#7b6d51;font-size:13px;line-height:1.7;margin:16px 0}
    .research-note b {display:block;color:#504633;font-weight:600}
    @media(max-width:1050px) {
      .wordmark small {display:none}.st-key-topbar [role="radiogroup"] label{padding:8px}
    }
    @media(max-width:760px) {
      .jakarta-banner {height:122px;border-radius:9px}
      .st-key-photo-header .jakarta-banner {height:260px}
      .st-key-photo-header .jakarta-banner-caption {justify-content:flex-start;padding-top:28px}
      .st-key-photo-header .st-key-topbar {padding:8px 12px}
      .st-key-photo-header .wordmark {text-align:center}
      .jakarta-banner-caption strong {font-size:19px}

      .block-container {padding:3rem 12px 20px}
      .st-key-topbar [data-testid="stHorizontalBlock"] {flex-wrap:wrap}
      .st-key-topbar [data-testid="stColumn"] {width:100%!important;flex:1 1 100%!important;min-width:0!important}
      .st-key-topbar [role="radiogroup"] {justify-content:center!important;gap:0}
      .st-key-topbar [data-testid="stColumn"]:last-child {display:none}
      .st-key-topbar [role="radiogroup"] label {padding:8px 10px}
      .location-strip {align-items:flex-start}.location-strip span:not(.archive-tag){display:block;margin:4px 0 0}
      .st-key-map-panel {position:relative!important;top:auto;left:auto;width:100%!important;border:0;border-radius:0;
         padding:18px;box-shadow:none;border-bottom:1px solid var(--line)}
      .st-key-map-stage {overflow:hidden}
      .map-legend {gap:10px}.map-legend .map-tip {margin-left:0;width:100%}
      .st-key-map-panel [data-testid="stHorizontalBlock"] {flex-wrap:nowrap}
      .st-key-map-panel [data-testid="stColumn"] {min-width:0!important;width:auto!important}
    }
    </style>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def jakarta_banner_image():
    image = Path(__file__).parent / "assets" / "jakarta_hero.jpeg"
    return base64.b64encode(image.read_bytes()).decode("ascii")


def render_header():
    st.markdown(
        '<header class="jakarta-banner">'
        f'<img src="data:image/jpeg;base64,{jakarta_banner_image()}" '
        'alt="Panorama gedung dan jalan Jakarta saat matahari terbenam">'
        '<div class="jakarta-banner-caption"><span>JAKARTA &amp; SEKITARNYA</span>'
        '<strong class="hero-title">Kota bergerak. <mark class="ink-highlight">Baca arahnya.</mark></strong></div></header>',
        unsafe_allow_html=True,
    )
