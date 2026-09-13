"""Draw the matchup chart used by the current Strategy-track report.

Reads build/postgap_mu.json, a local aggregate not distributed in this repo.
Writes reports/writeup_fig4_matchups.png (Figure 3 in the current report).
Named builds are shown separately, with Wilson 95% intervals. Groups with
fewer than 15 games and games without recovered decklists are omitted from
the chart; these omissions do not change the report's aggregate statistics.

    python analysis/_writeup_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports")
BLUE, RED, MUTED, GRID, INK = "#2a78d6", "#e34948", "#898781", "#e1e0d9", "#0b0b0b"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

mu = json.load(open(os.path.join(ROOT, "build", "postgap_mu.json"), encoding="utf-8"))
EN = {
    "dragapult_jamming": "Dragapult / Jamming Tower",
    "dragapult_watchtower": "Dragapult / TR Watchtower",
    "dragapult_ruins": "Dragapult / Risky Ruins",
    "dragapult_blaziken": "Dragapult + Blaziken ex",
    "dragapult_dusknoir": "Dragapult + Dusknoir",
    "kanga_primecatcher": "M-Kangaskhan box / Prime Catcher",
    "kanga_secretbox": "M-Kangaskhan box / Secret Box",
    "kanga_ogerpon": "M-Kangaskhan + Teal Mask Ogerpon",
    "ogerpon_hydrapple": "Ogerpon-Hydrapple (mirror)",
    "alakazam": "Alakazam", "mega_lopunny": "M-Lopunny",
    "kangaskhan_crustle": "M-Kangaskhan + Crustle", "mega_lucario": "M-Lucario",
    "ogerpon": "Ogerpon (mono)", "grookey_dipplin": "Grookey-Dipplin",
    # the three lists classify_deck could not place; named after what they run
    "7ae3fa73": "Espeon ex + Sylveon (Eevee wall)",
    "7a47958a": "Latias ex + Meowth ex",
    "cf84cb8b": "Cornerstone Mask Ogerpon ex",
}
# rows that carry no decklist at all are not a matchup and are dropped
DROP = {"unknown"}
by_arch = {}
for v in mu["variants"]:
    by_arch.setdefault(v["arch"], []).append(v)
by_fp = {}
for r in mu["lists"]:
    by_fp.setdefault(r["arch"], []).append(r)
rows = []
for a in mu["arch"]:
    if a["arch"] in DROP:
        continue
    if a["arch"] in by_arch:                      # split into named variants
        for v in by_arch[a["arch"]]:
            rows.append((v["key"], v["g"], v["wr"], v["lo"], v["hi"]))
    elif a["arch"] == "other":                    # split into its actual lists
        for r in by_fp.get("other", []):
            rows.append((r["fp"], r["g"], r["wr"], r["lo"], r["hi"]))
    else:
        rows.append((a["arch"], a["g"], a["wr"], a["lo"], a["hi"]))
rows = sorted([r for r in rows if r[1] >= 15], key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(10, 6.2))
for i, (k, g, wr, lo, hi) in enumerate(rows):
    col = BLUE if wr >= 50 else RED
    # One interval per row; the colored dot marks the observed win rate.
    ax.plot([lo, hi], [i, i], color=MUTED, lw=1.4,
            solid_capstyle="butt", zorder=3)
    ax.vlines([lo, hi], i - .065, i + .065, color=MUTED, lw=1.2, zorder=3)
    ax.plot(wr, i, "o", color=col, ms=7, mec="white", mew=1.5, zorder=4)
    ax.text(101, i, f"{wr:.1f}%", va="center", fontsize=9, color=col, fontweight="bold")
    ax.text(-1, i, f"{EN.get(k, k)}  ({g})", va="center", ha="right", fontsize=9, color=INK)
ax.axvline(50, color="#c3c2b7", lw=1.4)
ax.set_xlim(0, 100)
ax.set_ylim(-.8, len(rows) - .2)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, lw=.8)
ax.spines["left"].set_visible(False)
ax.set_title("Elo-matched win rate by opponent build (1,800 games, Wilson 95% CI)",
             loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "writeup_fig4_matchups.png"), dpi=170,
            bbox_inches="tight", pad_inches=.15)
plt.close(fig)
print("wrote reports/writeup_fig4_matchups.png")
