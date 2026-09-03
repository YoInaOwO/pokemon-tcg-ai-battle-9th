"""Figure 3: the 8.4M PPO run's health.

Three panels. The pool-win-rate curve is not repeated here (figure 2 has it),
and the per-seat split is left out: wr_pool is a share-weighted mean over
300-game rolling windows, so it is far noisier than a whole-update average and
the two do not tell the same story late in the run.

    python build/_fig3_training.py
"""
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "run_hydra_15038_r8m.log")
OUT = os.path.join(ROOT, "reports", "writeup_fig3_training.png")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#22201c", "#6b6862", "#e3e1d9"

rows = []
for line in open(LOG, encoding="utf-8", errors="ignore"):
    line = line.strip()
    if line.startswith("{'upd'"):
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            pass
upd = [r["upd"] for r in rows]
best_wr, best_upd = rows[-1]["best"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cbc9c1",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.42, wspace=0.28)
fig.suptitle("PPO run: rollout 8.4M, 57 updates, 56.3 h on 8 RTX 4090s",
             fontsize=12.5, color=INK, y=0.98)

# ---- per-archetype win rate (spans both rows) ----------------------------
ARCHS = [("bc:kangaskhan_box", "kangaskhan_box", "#d62728"),
         ("bc:mega_lopunny", "mega_lopunny", "#e377c2"),
         ("bc:marnie_grimmsnarl", "marnie_grimmsnarl", "#9467bd"),
         ("bc:dragapult", "dragapult", "#2ca02c"),
         ("bc:grookey_dipplin", "grookey_dipplin", "#8c564b"),
         ("bc:ogerpon_hydrapple", "ogerpon_hydrapple", "#7f7f7f"),
         ("bc:alakazam", "alakazam", ORANGE),
         ("mirror", "mirror", "#4c8fd6")]
a = fig.add_subplot(gs[:, 0])
for key, label, color in ARCHS:
    y = [(r.get("wr_arch") or {}).get(key) for r in rows]
    if all(v is None for v in y):
        continue
    a.plot(upd, y, "-", color=color, lw=1.5, label=label)
scr = [[v for k, v in (r.get("wr_arch") or {}).items() if k.startswith("script:")]
       for r in rows]
a.plot(upd, [sum(v) / len(v) for v in scr], "--", color=INK, lw=1.4,
       label="scripted agents (avg)")
a.axvline(best_upd, color=INK, lw=1, ls=(0, (4, 3)), alpha=0.6)
a.text(best_upd - 1.2, 0.975, "submitted checkpoint", fontsize=8.6, color=MUTED,
       ha="right", va="top")
a.set_ylim(0.15, 1.0)
a.set_xlim(0, 58.5)
a.set_xlabel("update")
a.set_ylabel("win rate this update")
a.set_title("win rate by opponent archetype", loc="left", fontsize=11, color=INK)
a.grid(color=GRID, lw=0.8)
a.legend(loc="lower right", frameon=False, fontsize=8.6, ncol=2)

# ---- PPO health ----------------------------------------------------------
a = fig.add_subplot(gs[0, 1])
a.plot(upd, [r["kl"] for r in rows], "-", color="#d62728", lw=1.5, label="KL")
on = [(u, r["kl_bc"]) for u, r in zip(upd, rows) if r["kl_bc"] > 0]
a.plot([u for u, _ in on], [v for _, v in on], "-", color="#e377c2", lw=1.5,
       label="KL to the BC anchor (while active)")
a.axvline(on[-1][0] + 0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
a.text(on[-1][0] + 1.5, 0.087, "anchor annealed off", fontsize=8.4, color=MUTED)
a.plot(upd, [r["clipfrac"] for r in rows], "-", color="#9467bd", lw=1.5,
       label="clip fraction")
a.set_ylim(0, 0.098)
a.set_xlim(0, 58.5)
a.set_xlabel("update")
a.set_ylabel("KL / clip fraction")
a.set_title("PPO health", loc="left", fontsize=11, color=INK)
a.grid(color=GRID, lw=0.8)
b = a.twinx()
b.plot(upd, [r["ent"] for r in rows], "-", color=AQUA, lw=1.5, label="entropy")
b.plot(upd, [r["ev"] for r in rows], "-", color=BLUE, lw=1.5,
       label="explained variance")
b.set_ylabel("entropy / explained variance")
b.tick_params(colors=MUTED)
h1, l1 = a.get_legend_handles_labels()
h2, l2 = b.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8.2)

# ---- throughput ----------------------------------------------------------
a = fig.add_subplot(gs[1, 1])
a.bar(upd, [r["collect_s"] / 60 for r in rows], color=BLUE, width=0.8,
      label="collect (min)")
a.bar(upd, [r["train_s"] / 60 for r in rows],
      bottom=[r["collect_s"] / 60 for r in rows], color=ORANGE, width=0.8,
      label="train (min)")
a.set_xlabel("update")
a.set_ylabel("minutes per update")
a.set_xlim(0, 58.5)
a.set_ylim(0, 96)
a.set_title("throughput: collection dominates", loc="left", fontsize=11, color=INK)
a.grid(color=GRID, lw=0.8, axis="y")
b = a.twinx()
b.plot(upd, [r["dec_per_s"] for r in rows], "-", color=INK, lw=1.2,
       label="decisions / s")
b.set_ylabel("decisions / s")
b.set_ylim(0, 4000)
b.tick_params(colors=MUTED)
h1, l1 = a.get_legend_handles_labels()
h2, l2 = b.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8.2, ncol=3)

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
