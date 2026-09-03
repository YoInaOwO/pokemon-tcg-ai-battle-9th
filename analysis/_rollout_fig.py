"""Rollout-size comparison for the writeup (reports/writeup_fig2_rollout.png).

Three runs of the same recipe -- same learner, lr, clip, epochs and per-GPU
minibatch -- differing only in how many decisions are collected per PPO update.

  131,072    the earlier recipe; the grey reference series in
             reports/ppo_hydra_15038_vt_trend.png (run on an earlier pool),
             read off that chart because its log is not on this machine
  524,288    logs/ppo_hydra_15038_vt.log, also read off the same chart (the
             copy of that log kept here is truncated)
  8,388,608  logs/run_hydra_15038_r8m.log -- the shipped model, full series

    python build/_rollout_fig.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "writeup_fig2_rollout.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": "#52514e",
                     "ytick.color": "#52514e"})

r8 = json.load(open(os.path.join(ROOT, "build", "r8m_curve.json")))

# (cumulative decisions in millions, share-weighted pool win rate)
vt524 = [(0.5, .527), (1, .535), (1.5, .545), (2, .555), (2.5, .560), (3, .575),
         (3.5, .585), (4, .588), (4.5, .605), (5, .600), (5.5, .608), (6, .620),
         (6.5, .612), (7, .622), (7.5, .628), (8, .625), (8.5, .632), (9, .628),
         (9.5, .635), (10, .628), (10.5, .640), (11, .655), (11.5, .645),
         (12, .650), (13, .643), (13.5, .658), (14, .652), (15, .665),
         (15.5, .660), (16, .672), (17, .688), (17.5, .680), (18, .690),
         (19, .685), (20, .692), (21, .690), (22, .693), (23, .688), (24, .700),
         (25, .695), (26, .703), (27, .700), (28, .708), (29, .712), (30, .705),
         (31, .713), (32, .708), (33, .715), (34, .710), (35, .718), (36, .712),
         (37, .728), (38, .715), (39, .722), (40.4, .720)]
vt131 = [(0.5, .470), (1, .462), (1.5, .505), (2, .495), (2.5, .520), (3, .535),
         (3.5, .545), (4, .552), (4.5, .560), (5, .565), (6, .573), (7, .578),
         (8, .582), (9, .585), (10, .590), (11, .585), (12, .595), (13, .600),
         (14, .598), (15, .605), (16, .600), (17, .610), (18, .605), (19, .615),
         (20, .612), (21, .620), (22, .618), (23, .628), (24, .622), (25, .630),
         (26, .625), (27, .632), (28, .628), (29, .638), (30, .630), (31, .640),
         (32, .635), (33, .628), (34, .645), (35, .658)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.7),
                             gridspec_kw={"width_ratios": [1.1, 1]})

# ---- left: matched-budget window, first ~42M decisions
a1.plot([d for d, _ in vt131], [w for _, w in vt131], "-", color=AQUA, lw=1.4,
        alpha=.95, label="rollout 131k    (~1.8k games / update)")
a1.plot([d for d, _ in vt524], [w for _, w in vt524], "-", color=ORANGE, lw=1.7,
        label="rollout 524k    (~7.3k games / update)")
sel = [r for r in r8 if r["dec"] <= 45e6]
a1.plot([r["dec"] / 1e6 for r in sel], [r["wr"] for r in sel], "o-", color=BLUE,
        lw=2, ms=5, label="rollout 8.4M    (~120k games / update)")
a1.axhline(.5, color="#c3c2b7", lw=1)
a1.set_xlim(0, 42)
a1.set_ylim(.45, .87)
a1.set_xlabel("cumulative decisions (M)")
a1.set_ylabel("share-weighted win rate vs pool")
a1.set_title("131k → 524k lifts the whole curve;  8.4M starts slower",
             loc="left", fontsize=11, color=INK)
a1.grid(color=GRID, lw=.8)
a1.legend(loc="lower right", frameon=False, fontsize=8.8)

# ---- right: the full 8.4M run
a2.plot([r["dec"] / 1e6 for r in r8], [r["wr"] for r in r8], "o-", color=BLUE,
        lw=1.6, ms=3.5)
for y, col, txt in ((.720, ORANGE, "524k ceiling  0.72"),
                    (.658, AQUA, "131k ceiling  0.66")):
    a2.axhline(y, color=col, lw=1.2, ls=(0, (5, 3)))
    a2.text(486, y + .008, txt, color=col, fontsize=8.6, ha="right")
best = max(r8, key=lambda r: r["wr"])
a2.scatter([best["dec"] / 1e6], [best["wr"]], s=70, facecolor="none",
           edgecolor=BLUE, lw=1.8, zorder=6)
a2.annotate("submitted checkpoint  %.2f" % best["wr"],
            (best["dec"] / 1e6, best["wr"]), textcoords="offset points",
            xytext=(-8, 14), ha="right", fontsize=8.6, color=BLUE)
a2.axhline(.5, color="#c3c2b7", lw=1)
a2.set_xlim(0, 490)
a2.set_ylim(.45, .87)
a2.set_xlabel("cumulative decisions (M)")
a2.set_title("8.4M keeps climbing past both ceilings; best at 0.83", loc="left",
             fontsize=11, color=INK)
a2.grid(color=GRID, lw=.8)
a2.text(20, .475, "the 131k series ran on an earlier opponent pool",
        fontsize=8, color=MUTED)

fig.tight_layout()
fig.savefig(OUT, dpi=170, facecolor="white")
print("wrote", OUT)
