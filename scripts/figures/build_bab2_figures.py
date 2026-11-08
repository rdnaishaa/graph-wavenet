"""Rebuild thesis Chapter 2 figures from existing data; no model training."""

import ast
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/thesis/bab2"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)
blue = "#215a8a"
teal = "#008577"
orange = "#b86b12"
grey = "#637181"
manifest = {}


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, color=blue, size=11):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="white", ec=color, lw=1.6)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color)


def arrow(ax, a, b, color=grey):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})


# Source metadata and exact subset order from the existing notebook (parse, never execute).
nodes = pd.read_csv(ROOT / "data/final_64/final_64_corridors.csv").sort_values("node_index")
meta = pd.read_csv(ROOT / "corridors.csv")
meta["corridor_file"] = meta.station_id + ".csv"
nodes = nodes.merge(meta, on="corridor_file", validate="one_to_one")
z = np.load(ROOT / "outputs/final_64/fixed_graph/fixed_graph_64.npz")
D = z["distance_matrix"]
A = z["directed_knn_adjacency"]
nb = json.loads((ROOT / "notebooks/scalability_hybrid_vs_gclstm.ipynb").read_text())
source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
order = ast.literal_eval(re.search(r"NODE_ORDER\s*=\s*(\[[\s\S]*?\])", source).group(1))
assert len(set(order)) == 64
# 1 Real window: observed targets at all requested horizons; imputed input explicitly marked.
df = pd.read_csv(
    ROOT / "data/final_64/forecasting_train_preprocessed.csv", parse_dates=["obs_time_utc"]
)
g = df[df.corridor_file == "tt_cawang.csv"].sort_values("obs_time_utc").reset_index(drop=True)
candidates = []
for i in range(11, len(g) - 6):
    w = g.iloc[i - 11 : i + 7]
    if (
        w.segment_id.nunique() == 1
        and (w.obs_time_utc.diff().dropna() == pd.Timedelta(hours=1)).all()
        and g.iloc[[i + 1, i + 3, i + 6]].is_observed.all()
    ):
        candidates.append((w.current_speed.std(), i))
_, i = max(candidates)
w = g.iloc[i - 11 : i + 7].reset_index(drop=True)
x = np.arange(-11, 7)
y = w.current_speed.to_numpy()
obs = w.is_observed.to_numpy(bool)
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.axvspan(-11.5, 0.4, color=blue, alpha=0.07)
ax.plot(x[:12], y[:12], color=blue, lw=1.4)
ax.scatter(x[:12][obs[:12]], y[:12][obs[:12]], color=blue, label="Input: observasi asli", zorder=4)
ax.scatter(
    x[:12][~obs[:12]],
    y[:12][~obs[:12]],
    facecolor="white",
    edgecolor=orange,
    marker="s",
    label="Input: hasil imputasi",
    zorder=4,
)
for h in [1, 3, 6]:
    ax.scatter(h, y[11 + h], s=80, marker="D", color=teal, zorder=4)
    ax.annotate(f"t+{h}", (h, y[11 + h]), xytext=(0, 10), textcoords="offset points", ha="center")
ax.scatter([], [], s=65, marker="D", color=teal, label="Target aktual (bukan prediksi)")
ax.axvline(0.4, color=grey, ls="--")
ax.set_xticks([-11, -8, -5, -2, 0, 1, 3, 6])
ax.set_xticklabels(["t−11", "t−8", "t−5", "t−2", "t", "t+1", "t+3", "t+6"])
ax.set(xlabel="Waktu relatif terhadap observasi terakhir (jam)", ylabel="Kecepatan (km/jam)")
ax.set_title(
    "Contoh window koridor Cawang | t = "
    + str(w.iloc[11].obs_time_utc.tz_convert("Asia/Jakarta").strftime("%d-%m-%Y %H:%M WIB")),
    fontsize=11,
    pad=14,
)
ax.grid(axis="y", alpha=0.18)
ax.margins(y=0.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, fontsize=9, frameon=False)
save(fig, "bab2_01_window")
manifest["window"] = {
    "corridor": "tt_cawang.csv",
    "t_utc": str(w.iloc[11].obs_time_utc),
    "target_values": {str(h): float(y[11 + h]) for h in [1, 3, 6]},
    "input_imputed": int((~obs[:12]).sum()),
}
# 2 Graph example: exact induced subgraph of saved 64-node adjacency.
sel = order[:8]
small = nodes.iloc[sel]
fig, ax = plt.subplots(figsize=(9, 5.8))
for ii in sel:
    for jj in sel:
        if A[ii, jj] > 0:
            p = nodes.iloc[ii]
            q = nodes.iloc[jj]
            ax.annotate(
                "",
                (q.lon, q.lat),
                (p.lon, p.lat),
                arrowprops={
                    "arrowstyle": "->",
                    "color": grey,
                    "alpha": 0.35,
                    "lw": 1,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
            )
ax.scatter(small.lon, small.lat, c=blue, s=90, zorder=3)
for idx, r in small.iterrows():
    ax.annotate(
        str(r.node_index), (r.lon, r.lat), xytext=(6, 6), textcoords="offset points", fontsize=10
    )
ax.set(
    xlabel="Bujur (derajat)",
    ylabel="Lintang (derajat)",
    title="Delapan node contoh dari graf geografis 64 koridor",
)
ax.grid(alpha=0.15)
labels = [
    "%d: %s" % (r.node_index, r.station_id.removeprefix("tt_").replace("_", " "))
    for _, r in small.iterrows()
]
fig.text(0.99, 0.5, "ID node pada dataset\n\n" + "\n".join(labels), va="center", fontsize=10)
save(fig, "bab2_02_graf")
# 3 Conceptual architecture, connections follow current notebook.
fig, ax = plt.subplots(figsize=(11, 6))
ax.set(xlim=(0, 11), ylim=(0, 6))
ax.axis("off")
box(ax, 0.2, 2.5, 1.5, 1, "Input\n12 × N × F")
box(ax, 2.1, 2.5, 1.5, 1, "Padding kiri\n+ proyeksi 1×1")
box(ax, 4, 2.5, 2, 1, "Temporal berdilatasi\n+ gerbang")
box(ax, 6.5, 2.5, 1.8, 1, "Difusi graf\n+ dropout")
box(ax, 8.8, 2.5, 1.9, 1, "Residual\n+ normalisasi")
for a, b in [((1.7, 3), (2.1, 3)), ((3.6, 3), (4, 3)), ((6, 3), (6.5, 3)), ((8.3, 3), (8.8, 3))]:
    arrow(ax, a, b)
ax.plot([3.8, 3.8, 9.75], [3, 4.4, 4.4], c=grey, lw=1.2)
arrow(ax, (9.75, 4.4), (9.75, 3.5))
ax.text(6.7, 4.5, "Jalur residual (panjang waktu disesuaikan)", ha="center", fontsize=10)
box(ax, 4, 0.4, 2, 1, "Proyeksi skip 1×1\n+ jumlah antar-lapisan", teal, 10)
box(ax, 6.5, 0.4, 1.8, 1, "ReLU + proyeksi\nkeluaran", teal, 10)
box(ax, 8.8, 0.4, 1.9, 1, "Prediksi langsung\n3 × N", teal, 10)
arrow(ax, (5, 2.5), (5, 1.4), teal)
arrow(ax, (6, 0.9), (6.5, 0.9), teal)
arrow(ax, (8.3, 0.9), (8.8, 0.9), teal)
ax.text(7.3, 2, "Lapisan spasial-temporal diulang", ha="center", fontsize=10)
ax.text(7.4, 5.5, "Skema aliran data Graph WaveNet", ha="center", fontsize=14, fontweight="bold")
save(fig, "bab2_03_arsitektur")
# 4 Receptive field faithful to implementation.
fig, axs = plt.subplots(2, 1, figsize=(10, 5.6), gridspec_kw={"height_ratios": [1.3, 1]})
dil = np.tile([1, 2], 4)
rf = 1 + np.r_[0, np.cumsum(dil)]
axs[0].plot(range(9), rf, "o-", color=blue)
axs[0].set_xticks(range(9))
axs[0].set_xticklabels(["Input"] + [f"L{k}\nd={d}" for k, d in enumerate(dil, 1)])
axs[0].set(ylabel="Receptive field (langkah)", ylim=(0, 15))
axs[0].grid(axis="y", alpha=0.2)
axs[0].set_title("Kernel 2; dilasi 1 dan 2 diulang pada empat blok", fontsize=12)
ax = axs[1]
ax.axis("off")
ax.set(xlim=(-0.5, 13), ylim=(0, 2))
for j in range(13):
    ax.add_patch(
        Rectangle(
            (j, 0.8), 0.9, 0.7, fc="#f3ece2" if j == 0 else "#e5eff5", ec=orange if j == 0 else blue
        )
    )
    ax.text(
        j + 0.45,
        1.15,
        "0" if j == 0 else ("t" if j == 12 else f"t−{12 - j}"),
        ha="center",
        fontsize=9,
    )
ax.text(0.45, 0.45, "Padding", ha="center", fontsize=9, color=orange)
ax.text(
    6.8, 0.3, "12 langkah waktu masukan; padding bukan observasi tambahan", ha="center", fontsize=11
)
fig.tight_layout()
save(fig, "bab2_04_dilasi")
# 5 Embedding dimensions match stored orientation in notebook.
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.axis("off")
ax.set(xlim=(0, 11), ylim=(0, 3.4))
items = [
    (0.1, 1.5, "E₁\nN × d"),
    (2.3, 1.5, "E₂\nd × N"),
    (4.5, 1.5, "ReLU\nN × N"),
    (6.7, 1.5, "Softmax\nper baris"),
    (8.9, 1.5, "A adaptif\nN × N"),
]
for x0, w0, t in items:
    box(ax, x0, 1.2, w0, 1, t, blue, 11)
ax.text(1.95, 1.7, "×", ha="center", fontsize=20)
for a, b in [(3.8, 4.5), (6, 6.7), (8.2, 8.9)]:
    arrow(ax, (a, 1.7), (b, 1.7))
ax.text(
    5.5,
    0.45,
    "Embedding dipelajari saat pelatihan; bobot graf tetap untuk checkpoint yang sama.",
    ha="center",
    fontsize=11,
)
save(fig, "bab2_05_adaptif")
# 6 Supports, no invented learned edges.
fig, axs = plt.subplots(1, 3, figsize=(11, 4))
for ax, title, labels in zip(
    axs,
    ["Fixed Graph", "Adaptive Graph", "Hybrid Graph"],
    [["P maju", "P balik"], ["A adaptif"], ["P maju", "P balik", "A adaptif"]],
):
    ax.axis("off")
    ax.set(xlim=(0, 3), ylim=(0, 4))
    ax.text(1.5, 3.7, title, ha="center", fontsize=13, fontweight="bold")
    for j, label in enumerate(labels):
        box(ax, 0.5, 2.8 - j * 0.75, 2, 0.5, label, teal if "adaptif" in label else blue, 11)
    ax.text(1.5, 0.35, f"{len(labels)} support terpisah", ha="center", fontsize=11)
fig.tight_layout()
save(fig, "bab2_06_konfigurasi")
# 7 Actual split extent; dates are UTC and postprocessing discontinuities explained in caption.
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.axis("off")
ax.set(xlim=(0, 11), ylim=(0, 3.4))
for j, (split, label, col) in enumerate(
    [("train", "Pelatihan", blue), ("val", "Validasi", teal), ("test", "Pengujian", orange)]
):
    d = pd.read_csv(
        ROOT / f"data/final_64/forecasting_{split}_preprocessed.csv", usecols=["obs_time_utc"]
    )
    t = pd.to_datetime(d.obs_time_utc, utc=True)
    start, end = t.min(), t.max()
    manifest[split] = {"start_utc": str(start), "end_utc": str(end)}
    box(
        ax,
        0.15 + j * 3.65,
        1.05,
        3.25,
        1.35,
        label
        + "\n"
        + start.strftime("%d %b %Y %H:%M")
        + "\ns.d. "
        + end.strftime("%d %b %Y %H:%M"),
        col,
        10,
    )
    if j < 2:
        arrow(ax, (3.4 + j * 3.65, 1.7), (3.8 + j * 3.65, 1.7))
ax.text(
    5.5,
    0.4,
    "Urutan waktu →  |  Zona waktu UTC  |  Lebar kotak tidak menyatakan durasi",
    ha="center",
    fontsize=11,
)
save(fig, "bab2_07_split")
# 8 Nested node subsets matching exact saved notebook NODE_ORDER.
fig, axs = plt.subplots(2, 2, figsize=(9, 7.4), sharex=True, sharey=True)
for ax, n in zip(axs.flat, [8, 16, 32, 64]):
    s = nodes.iloc[order[:n]]
    ax.scatter(nodes.lon, nodes.lat, s=17, c="#d7dde2", label="Di luar subset")
    ax.scatter(s.lon, s.lat, s=24, c=blue, label="Dalam subset")
    c = nodes.iloc[order[0]]
    ax.scatter(c.lon, c.lat, marker="*", s=115, c=orange, label="Pusat: Kuningan")
    ax.set_title(f"{n} koridor")
    ax.grid(alpha=0.15)
for ax in axs[1, :]:
    ax.set_xlabel("Bujur (derajat)")
for ax in axs[:, 0]:
    ax.set_ylabel("Lintang (derajat)")
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "bab2_08_subset")
manifest["node_order"] = order
manifest["sources"] = [
    "data/final_64/forecasting_train_preprocessed.csv",
    "data/final_64/final_64_corridors.csv",
    "corridors.csv",
    "outputs/final_64/fixed_graph/fixed_graph_64.npz",
    "notebooks/graphwavenet_final64_audited.ipynb",
    "notebooks/scalability_hybrid_vs_gclstm.ipynb",
]
(OUT / "provenance.json").write_text(json.dumps(manifest, indent=2))
print("Created 8 figures (PNG and PDF):", OUT)
