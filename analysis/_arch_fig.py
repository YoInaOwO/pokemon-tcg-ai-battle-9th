"""Architecture schematic for the writeup (reports/writeup_fig1_arch.png).

Laid out so that no connector crosses another and every connector is drawn on
right angles: the state path along the top, the option path directly below it
(fed by two short vertical drops), the policy output below that, and the three
heads in their own column on the right, fed by one bus down the clear gutter
to the right of the option path.

Type sizes are set against the box widths -- the binding constraints are the
longest body line in the observation box and the longest bold title, so those
strings are kept short enough that 12pt body / 15pt titles still fit.

    python build/_arch_fig.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "writeup_fig1_arch.png")

INK, MUTED, FAINT = "#0b0b0b", "#4a4945", "#8e8c85"
BLUE, BLUE_BG = "#2a78d6", "#eff5fd"
ORANGE, ORANGE_BG = "#d4571f", "#fdf3ec"
VIOLET, VIOLET_BG = "#4a3aa7", "#f3f1fb"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(15.2, 10.0))
ax.set_xlim(0, 152)
ax.set_ylim(0, 100)
ax.axis("off")

X = {"a": (2, 33), "b": (39, 23), "c": (66, 27), "d": (97, 23), "h": (124, 27)}
TOP, MID, LOW = 72, 42, 16          # row baselines
RH = 24                             # standard row height
BUS = 121.0                         # clear gutter right of the option lane
TFS, BFS = 15, 12                   # title / body point size
LFS = 13                            # lane labels and edge annotations


def box(col, y, title, body, accent=MUTED, bg="white", h=RH, tfs=TFS, bfs=BFS):
    x, w = X[col]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=bg, ec=accent, lw=2, zorder=3))
    ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=tfs,
            fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + (h - 7.0) / 2, body, ha="center", va="center",
            fontsize=bfs, color=MUTED, linespacing=1.45, zorder=4)


def tip(pt, d, color):
    x, y = pt
    s, L = 1.3, 3.3
    tri = {"r": [(x, y), (x - L, y + s), (x - L, y - s)],
           "l": [(x, y), (x + L, y + s), (x + L, y - s)],
           "d": [(x, y), (x - s, y + L), (x + s, y + L)],
           "u": [(x, y), (x - s, y - L), (x + s, y - L)]}[d]
    ax.add_patch(Polygon(tri, closed=True, fc=color, ec=color, zorder=5))


def route(pts, color=MUTED, d="r", lw=2.0, ls="-"):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=ls,
            solid_joinstyle="miter", solid_capstyle="butt", zorder=2)
    tip(pts[-1], d, color)


def right(col):
    return X[col][0] + X[col][1]


def mid(col):
    return X[col][0] + X[col][1] / 2


# ----------------------------------------------------------------- state lane
ax.text(2, 98, "S T A T E   P A T H", fontsize=LFS, color=BLUE, fontweight="bold")
box("a", TOP, "OBSERVATION → 64 tokens",
    "global 40 · select context 68\n"
    "stadium · board ×18 · hand ×30\n"
    "looking ×8 · discard ×2\n"
    "log summary · game memory\n"
    "bag of my own unseen deck",
    BLUE, BLUE_BG)
box("b", TOP, "CardRepr",
    "id embedding\n+ projected static\ncard features\n\nid-dropout in training",
    BLUE, BLUE_BG)
box("c", TOP, "Transformer encoder",
    "5 layers, pre-LN\nd = 384, 4 heads\ntype + position emb\n7.8M parameters",
    BLUE, BLUE_BG)
box("d", TOP, "pooled state  h", "masked mean\nover tokens", BLUE, BLUE_BG)
for f, t in (("a", "b"), ("b", "c"), ("c", "d")):
    route([(right(f), TOP + RH / 2), (X[t][0], TOP + RH / 2)], BLUE)

# ---------------------------------------------------------------- option lane
ax.text(2, 68, "O P T I O N   P A T H      (K legal options per decision)",
        fontsize=LFS, color=ORANGE, fontweight="bold")
box("a", MID, "engine lookahead probe",
    "execute every option once\nin the engine sandbox and\n"
    "read its consequences:\ndamage, KO, prizes, deltas,\n"
    "turn end, macro E[·], legality",
    ORANGE, ORANGE_BG)
box("b", MID, "option encoding",
    "60 base columns\n+ 28 lookahead columns\n+ card, target and\nattack embeddings",
    ORANGE, ORANGE_BG)
box("c", MID, "cross-attention",
    "every option reads the\nstate tokens — the live\nHP and Energy of the\nbench slot it targets",
    ORANGE, ORANGE_BG)
box("d", MID, "score MLP", "logit = MLP(\n[opt, h, opt ⊙ h])", ORANGE, ORANGE_BG)
for f, t in (("a", "b"), ("b", "c"), ("c", "d")):
    route([(right(f), MID + RH / 2), (X[t][0], MID + RH / 2)], ORANGE)

route([(mid("c"), TOP), (mid("c"), MID + RH)], BLUE, d="d")
ax.text(mid("c") + 2.6, (TOP + MID + RH) / 2, "state tokens", fontsize=LFS - 1,
        color=BLUE, va="center")
route([(mid("d"), TOP), (mid("d"), MID + RH)], BLUE, d="d")
ax.text(mid("d") + 2.6, (TOP + MID + RH) / 2, "h", fontsize=LFS - 1, color=BLUE,
        va="center")

box("d", LOW, "policy logits", "single-select: argmax\nmulti-select: top-k",
    ORANGE, ORANGE_BG, h=20)
route([(mid("d"), MID), (mid("d"), LOW + 20)], ORANGE, d="d")

# ----------------------------------------------------------------- head column
ax.text(124, 98, "H E A D S", fontsize=LFS, color=MUTED, fontweight="bold")
box("h", TOP, "count head  k", "[h, mean option repr]\n→ 24 classes,\nthen top-k")
box("h", MID, "blind value  V(s)",
    "the critic that plays:\nsees only what the\npolicy sees")
box("h", LOW, "oracle value", "h + a snapshot of the\nopponent's hand;\ndrives GAE in PPO",
    VIOLET, VIOLET_BG, h=20)

route([(right("d"), TOP + RH / 2), (X["h"][0], TOP + RH / 2)], MUTED)
ax.plot([BUS, BUS], [TOP + RH / 2, LOW + 10], color=MUTED, lw=2.0, zorder=2)
ax.plot([BUS], [TOP + RH / 2], marker="o", ms=6, color=MUTED, zorder=4)
route([(BUS, MID + RH / 2), (X["h"][0], MID + RH / 2)], MUTED)
route([(BUS, LOW + 10), (X["h"][0], LOW + 10)], VIOLET)

route([(mid("h"), 8), (mid("h"), LOW)], VIOLET, d="u", ls=(0, (4, 2.5)))
ax.text(mid("h"), 6.4, "side channel, training only —\nnever enters the encoder",
        fontsize=LFS - 1, color=VIOLET, ha="center", va="top", linespacing=1.45)

ax.text(2, 10, "Behavioural cloning additionally trains an auxiliary head\n"
        "(prizes taken by each side within 2 and 4 turns) on the pooled state.\n"
        "It and the oracle value head are dropped at inference; the shipped\n"
        "agent runs the policy path in pure NumPy.",
        fontsize=LFS - 1, color=FAINT, va="top", linespacing=1.5)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=170, facecolor="white")
print("wrote", OUT)
