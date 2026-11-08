"""Read-only audit and thesis figures from frozen preprocessing artifacts."""

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/thesis/bab3"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 12,
        "savefig.facecolor": "white",
    }
)
frames = {
    s: pd.read_csv(
        ROOT / f"data/final_64/forecasting_{s}_preprocessed.csv", parse_dates=["obs_time_utc"]
    )
    for s in ["train", "val", "test"]
}
df = pd.concat(frames.values()).sort_values(["obs_time_utc", "corridor_file"])
nodes = pd.read_csv(ROOT / "data/final_64/final_64_corridors.csv").sort_values("node_index")
names = nodes.corridor_file.tolist()
times = pd.DatetimeIndex(sorted(df.obs_time_utc.unique()))
obs = (
    df.pivot(index="obs_time_utc", columns="corridor_file", values="is_observed")
    .reindex(index=times, columns=names)
    .astype(bool)
)
rawdir = Path(
    os.environ.get("TOMTOM_RAW_DATA_DIR", ROOT / "data_update_dosen" / "full_data_update")
)
if not rawdir.is_dir():
    raise FileNotFoundError(
        "Set TOMTOM_RAW_DATA_DIR to the folder containing the raw corridor CSV files."
    )
for name in names:
    raw = pd.read_csv(rawdir / name)
    raw.obs_time_utc = pd.to_datetime(raw.obs_time_utc, utc=True, errors="coerce")
    raw = (
        raw.dropna(subset=["obs_time_utc"])
        .sort_values("obs_time_utc")
        .drop_duplicates("obs_time_utc", keep="last")
        .set_index("obs_time_utc")
        .reindex(times)
    )
    assert np.array_equal(raw.current_speed.notna(), obs[name]), name
missing = ~obs.all(axis=1)
isolated = missing & (~missing.shift(1, fill_value=True)) & (~missing.shift(-1, fill_value=True))
for name in names:
    g = df[df.corridor_file == name].set_index("obs_time_utc").reindex(times)
    for col in ["current_speed", "congestion_ratio"]:
        raw = pd.read_csv(rawdir / name)
        raw.obs_time_utc = pd.to_datetime(raw.obs_time_utc, utc=True, errors="coerce")
        raw = (
            raw.dropna(subset=["obs_time_utc"])
            .sort_values("obs_time_utc")
            .drop_duplicates("obs_time_utc", keep="last")
            .set_index("obs_time_utc")
            .reindex(times)
        )
        expected = raw[col].copy()
        fill = isolated & expected.isna()
        expected.loc[fill] = raw[col].shift(1).loc[fill]
        assert np.allclose(expected, g[col], equal_nan=True), (name, col)
valid = (
    df.groupby("obs_time_utc")[["current_speed", "congestion_ratio"]]
    .apply(lambda x: x.notna().all().all())
    .reindex(times)
)
assert np.array_equal(valid, df.groupby("obs_time_utc").segment_id.first().reindex(times).notna())
train = frames["train"].dropna(subset=["segment_id"]).current_speed.to_numpy()
sc = np.load(ROOT / "data/final_64/stgnn_scaler.npz")
assert np.allclose([train.mean(), train.std()], [sc["mean"][0], sc["scale"][0]])
report = {
    "raw_mask_matches": True,
    "imputation_matches": True,
    "validity_matches": True,
    "scaler_matches_train": True,
    "mean": float(sc["mean"][0]),
    "std": float(sc["scale"][0]),
    "scaler_points": len(train),
    "grid_hours": len(times),
    "valid_hours": int(valid.sum()),
    "original_points": int(obs.to_numpy().sum()),
    "missing_points": int((~obs).to_numpy().sum()),
    "isolated_gap_hours": int(isolated.sum()),
    "splits": {},
}
for s, d in frames.items():
    z = np.load(ROOT / f"data/final_64/stgnn_{s}.npz")
    report["splits"][s] = {
        "start": str(d.obs_time_utc.min()),
        "end": str(d.obs_time_utc.max()),
        "grid_hours": d.obs_time_utc.nunique(),
        "valid_hours": d[d.segment_id.notna()].obs_time_utc.nunique(),
        "samples": len(z["X"]),
        "valid_targets_by_horizon": z["mask"].sum(axis=(0, 2)).tolist(),
    }


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# Full-resolution hourly mask; short zoom makes the alternating gaps legible.
fig, axs = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
cmap = ListedColormap(["#eeeeee", "#254d64"])
for ax, sel, title in [
    (axs[0], np.arange(len(times)), "Seluruh grid: 12 Agustus–11 September 2026"),
    (axs[1], np.arange(48), "Pembesaran: 48 jam pertama"),
]:
    ax.imshow(
        obs.to_numpy()[sel].T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1
    )
    ticks = np.linspace(0, len(sel) - 1, 6 if len(sel) > 48 else 4, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([times[sel[i]].strftime("%d/%m\n%H:%M") for i in ticks], fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Waktu (UTC)")
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(range(64))
axs[0].set_yticklabels(
    [f"{i}: " + n.removeprefix("tt_").removesuffix(".csv") for i, n in enumerate(names)], fontsize=8
)
axs[0].set_ylabel("Indeks dan identitas koridor")
fig.legend(
    handles=[
        Patch(facecolor="#254d64", label="Kecepatan terobservasi"),
        Patch(facecolor="#eeeeee", edgecolor="#aaaaaa", label="Kecepatan tidak tersedia"),
    ],
    loc="lower center",
    ncol=2,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.035, 1, 1))
save(fig, "bab3_03_missing_heatmap")
# Deterministic example: first long gap, +/- 12 hours, corridor Cawang.
g = df[df.corridor_file == "tt_cawang.csv"].set_index("obs_time_utc").reindex(times)
center = times[~valid][0]
g = g.loc[center - pd.Timedelta(hours=12) : center + pd.Timedelta(hours=14)]
fig, ax = plt.subplots(figsize=(11, 4.8))
for sid, part in g[g.segment_id.notna()].groupby("segment_id"):
    ax.plot(part.index, part.current_speed, color="#254d64", lw=1.3)
    original = part.is_observed.astype(bool)
    imputed = ~original & part.was_imputed.astype(bool)
    ax.scatter(part.index[original], part.current_speed[original], color="#254d64", s=30, zorder=4)
    ax.scatter(
        part.index[imputed],
        part.current_speed[imputed],
        facecolor="white",
        edgecolor="#b16b13",
        marker="s",
        s=55,
        zorder=5,
    )
for t in g.index[g.segment_id.isna()]:
    ax.axvspan(
        t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=30), color="#dadada", alpha=0.8
    )
ax.scatter([], [], color="#254d64", label="Observasi asli")
ax.scatter(
    [], [], facecolor="white", edgecolor="#b16b13", marker="s", label="Imputasi dari jam sebelumnya"
)
handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor="#dadada"))
labels.append("Waktu tidak valid / batas segmen")
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.23),
    ncol=3,
    frameon=False,
    fontsize=11,
)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
ax.set(
    xlabel="Waktu (UTC)",
    ylabel="Kecepatan (km/jam)",
    title="Koridor Cawang: contoh di sekitar gap panjang pertama",
)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
save(fig, "bab3_04_imputation_segments")
report["example"] = {
    "corridor": "tt_cawang.csv",
    "start": str(g.index.min()),
    "end": str(g.index.max()),
}
# Split and shared scaler, following actual order (imputation and segmentation precede split).
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
ax.set(xlim=(0, 12), ylim=(0, 6))


def box(x, y, w, h, txt, col="#254d64"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="white", ec=col, lw=1.4)
    )
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)


def arrow(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#555555"})


box(0.5, 5.15, 11, 0.65, "Grid 729 jam → penanganan gap dan segmentasi → pembagian kronologis")
for j, (s, label) in enumerate(
    [("train", "Pelatihan"), ("val", "Validasi"), ("test", "Pengujian")]
):
    r = report["splits"][s]
    x = 0.3 + j * 4
    box(
        x,
        3.25,
        3.4,
        1.35,
        f"{label}: {r['grid_hours']} posisi grid\n"
        + pd.Timestamp(r["start"]).strftime("%d/%m/%Y %H:%M")
        + " –\n"
        + pd.Timestamp(r["end"]).strftime("%d/%m/%Y %H:%M")
        + f" UTC\n{r['valid_hours']} waktu valid",
    )
    arrow((x + 1.7, 5.15), (x + 1.7, 4.6))
box(
    0.3,
    1.45,
    3.4,
    1.1,
    "Fit scaler hanya pada train valid\n491 × 64 nilai (termasuk imputasi)\nμ = 30,5066; σ = 11,3311 km/jam",
)
arrow((2, 3.25), (2, 2.55))
box(
    4.3,
    1.45,
    7.4,
    1.1,
    "Statistik yang sama untuk train, validation, dan test\nz = (v − μ) / σ; tidak melakukan fit ulang",
)
arrow((3.7, 2), (4.3, 2))
arrow((6, 3.25), (6, 2.55))
arrow((10, 3.25), (10, 2.55))
box(4.3, 0.15, 7.4, 0.7, "Window dibentuk terpisah per split dan tidak melintasi segmen")
arrow((8, 1.45), (8, 0.85))
save(fig, "bab3_05_split_normalization")
(OUT / "preprocessing_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
