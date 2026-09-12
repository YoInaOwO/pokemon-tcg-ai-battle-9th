"""Recreate model.png as clean, editable vector artwork with Matplotlib.

Run from any directory:
    python analysis/draw_model.py
    python analysis/draw_model.py --output reports/model_redrawn --dpi 240

Coordinates use a 1600 x 1000 design canvas. No source-image pixels are used.
The labels describe v6 with train_bc.py's default 384 / 5 / 6 configuration,
not a configuration inferred from the final submission checkpoint.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1600, 1000
INK = "#092354"
BODY = "#183F83"
TEAL = "#009BAB"
BLUE = "#0877F9"
PURPLE = "#770DFA"
GRAY = "#8792B0"
MUTED = "#7783A5"
PALE = {
    TEAL: ("#F0FAFB", "#E1F4F6"),
    BLUE: ("#F1F7FF", "#E4EFFF"),
    PURPLE: ("#F7F2FF", "#EEE6FC"),
    GRAY: ("#F7F8FA", "#EDF0F4"),
}


def create_figure():
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ("Arial", "DejaVu Sans") if name in available)
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "svg.fonttype": "none",  # SVG text remains editable/searchable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    text_bounds = []

    def label(x, y, value, size=19, color=BODY, weight="normal", align="left",
              va="center", bounds=None, linespacing=1.23):
        artist = ax.text(x, y, value, fontsize=size * .72, color=color,
                         fontweight=weight, ha=align, va=va,
                         linespacing=linespacing, zorder=6)
        if bounds:
            text_bounds.append((artist, bounds))
        return artist

    def panel(x, y, w, h, color, title, body=(), title_size=21,
              body_size=18, center=False, header=47, gap=27):
        fill, band = PALE[color]
        outer = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0,rounding_size=11",
                              linewidth=1.25, edgecolor=color, facecolor=fill,
                              zorder=4)
        ax.add_patch(outer)
        if header:
            stripe = Rectangle((x + .8, y + .8), w - 1.6, header,
                               facecolor=band, edgecolor="none", zorder=4.1)
            stripe.set_clip_path(outer)
            ax.add_patch(stripe)
        tx = x + w / 2 if center else x + 18
        ha = "center" if center else "left"
        label(tx, y + 27, title, title_size, INK, "bold", ha,
              bounds=(x + 8, y + 5, x + w - 8, y + h - 5))
        line_y = y + header + 19
        for line in body:
            label(tx, line_y, line, body_size, align=ha, va="top",
                  bounds=(x + 9, y + 5, x + w - 9, y + h - 6))
            line_y += gap * (line.count("\n") + 1)

    def compact(x, y, w, h, color, title, body, size=20, body_size=18):
        fill, _ = PALE[color]
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0,rounding_size=10",
                                   fc=fill, ec=color, lw=1.25, zorder=4))
        title_lines = title.count("\n") + 1
        label(x + w / 2, y + 16, title, size, INK, "bold", "center", va="top",
              bounds=(x + 6, y + 4, x + w - 6, y + h - 4))
        label(x + w / 2, y + 19 + title_lines * size * 1.14,
              body, body_size, align="center", va="top",
              bounds=(x + 6, y + 4, x + w - 6, y + h - 4))

    def route(points, color, arrow=True, dashed=False, width=2.5, radius=12):
        vertices = [points[0]]
        codes = [MplPath.MOVETO]
        for previous, corner, following in zip(points, points[1:], points[2:]):
            d1 = ((corner[0] - previous[0]) ** 2 + (corner[1] - previous[1]) ** 2) ** .5
            d2 = ((following[0] - corner[0]) ** 2 + (following[1] - corner[1]) ** 2) ** .5
            r = min(radius, d1 / 2, d2 / 2)
            before = tuple(corner[i] + (previous[i] - corner[i]) * r / d1 for i in (0, 1))
            after = tuple(corner[i] + (following[i] - corner[i]) * r / d2 for i in (0, 1))
            vertices.extend((before, corner, after))
            codes.extend((MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3))
        vertices.append(points[-1])
        codes.append(MplPath.LINETO)
        ax.add_patch(PathPatch(MplPath(vertices, codes), fill=False, ec=color,
                               lw=width, capstyle="round", joinstyle="round",
                               linestyle=(0, (4, 3)) if dashed else "solid", zorder=2))
        if arrow:
            end, prev = points[-1], points[-2]
            length = ((end[0] - prev[0]) ** 2 + (end[1] - prev[1]) ** 2) ** .5
            dx, dy = (end[0] - prev[0]) / length, (end[1] - prev[1]) / length
            al, aw = (13, 5.5) if dashed else (14, 6)
            base = (end[0] - al * dx, end[1] - al * dy)
            ax.add_patch(Polygon([end, (base[0] - aw * dy, base[1] + aw * dx),
                                 (base[0] + aw * dy, base[1] - aw * dx)],
                                fc=color, ec="none", zorder=3))

    # Heading and configuration badge.
    label(800, 43, "Pokémon TCG Policy Network", 49, INK, "bold", "center")
    ax.add_patch(FancyBboxPatch((495, 112), 610, 44,
                               boxstyle="round,pad=0,rounding_size=21",
                               ec="none", fc="#EDF1F6"))
    label(800, 134, "64 state tokens   |   d = 384   |   5 layers   |   6 heads",
          21, "#3D609E", "bold", "center")

    # State lane.
    label(20, 185, "State processing (game state)", 27, TEAL, "bold")
    panel(20, 210, 238, 179, TEAL, "Observed State", [
        "Board, hand & card zones", "Selection context & global\nfeatures",
        "Public history & opponent-\ndeck belief"], body_size=17.5, gap=23)
    panel(280, 210, 252, 179, TEAL, "State Token Embedding", [
        "Card / attack IDs + static\nattributes",
        "Feature projections + type &\nposition embeddings"], title_size=19, body_size=17.5, gap=24)
    panel(556, 210, 246, 179, TEAL, "Transformer Encoder", [
        "5 × pre-norm encoder blocks", "Self-attention + feed-forward"],
        body_size=16.5, gap=31)
    state_y = 210 + 179 / 2
    compact(824, state_y - 96 / 2, 142, 96, TEAL, "Encoded state\ntokens X", "(64 × 384)", size=19)
    compact(998, state_y - 86 / 2, 128, 86, TEAL, "Global state\nh = X[0]", "(384)", size=19, body_size=17)
    for start, end in [(258, 280), (532, 556), (802, 824), (966, 998)]:
        route([(start, state_y), (end, state_y)], TEAL)

    # Candidate-action lane and its queries / keys / values.
    label(20, 451, "Option processing (candidate actions)", 27, BLUE, "bold")
    panel(20, 477, 260, 159, BLUE, "Candidate Options", [
        "Action, card, target & attack", "Numeric features + lookahead\nprobes"],
        body_size=16.5, gap=29)
    panel(306, 477, 263, 159, BLUE, "Option Embedding", [
        "Shared card / attack\nrepresentations", "Concatenate + linear\nprojection"],
        body_size=18, gap=23)
    option_y = 477 + 159 / 2
    compact(676, option_y - 101 / 2, 222, 101, BLUE, "Cross-Attention", "Residual + LayerNorm", size=22)
    compact(924, option_y - 91 / 2, 130, 91, BLUE, "Contextual\noptions oᵢ", "(N × 384)", size=19, body_size=17)
    route([(280, option_y), (306, option_y)], BLUE)
    route([(569, option_y), (676, option_y)], BLUE)
    label(622, option_y - 20, "Queries", 18, BLUE, "bold", "center")
    # Connect the bottom/top edge midpoints, with the bend between lanes.
    state_x, attention_x = 824 + 142 / 2, 676 + 222 / 2
    route([(state_x, state_y + 96 / 2), (state_x, 426),
           (attention_x, 426), (attention_x, option_y - 101 / 2)], BLUE)
    label(state_x + 16, 402, "Keys / Values", 19, BLUE, "bold")
    route([(898, option_y), (924, option_y)], BLUE)

    # Inference heads: state and option buses retain distinct colors.
    label(1176, 356, "Action heads (inference)", 26, PURPLE, "bold")
    panel(1194, 389, 228, 121, PURPLE, "Policy Head", [
        "[oᵢ, h, oᵢ ⊙ h] → MLP", "Masked option scores"],
        center=True, body_size=18, gap=28)
    panel(1194, 531, 228, 119, PURPLE, "Count Head", [
        "[h, masked mean(o)] → Linear", "Selection count: 0–23"],
        center=True, body_size=15.5, gap=28)
    compact(1464, 457, 120, 127, PURPLE, "Action\nSelection", "Choose k, then\nselect options", size=22, body_size=15.5)
    policy_y, count_y, selection_y = 389 + 121 / 2, 531 + 119 / 2, 457 + 127 / 2
    # Paired inputs sit equally above/below each head's center.
    route([(1126, state_y), (1145, state_y),
           (1145, policy_y - 15), (1194, policy_y - 15)], TEAL)
    route([(1145, policy_y - 29), (1145, count_y - 15), (1194, count_y - 15)], TEAL)
    route([(1054, option_y), (1103, option_y)], BLUE, arrow=False)
    route([(1103, option_y), (1103, policy_y + 15), (1194, policy_y + 15)], BLUE)
    route([(1103, option_y), (1103, count_y + 15), (1194, count_y + 15)], BLUE)
    route([(1422, policy_y), (1450, policy_y), (1450, selection_y)], PURPLE, arrow=False)
    route([(1422, count_y), (1450, count_y), (1450, selection_y)], PURPLE, arrow=False)
    route([(1450, selection_y), (1464, selection_y)], PURPLE)

    # Secondary training/evaluation branches, including the private side input.
    ax.add_patch(FancyBboxPatch((331, 699), 1091, 231,
                               boxstyle="round,pad=0,rounding_size=13",
                               fc="white", ec=GRAY, lw=1.3,
                               linestyle=(0, (5, 3)), zorder=1))
    label(350, 725, "Training / evaluation branches", 22, MUTED, "bold")
    label(669, 725, "· omitted from NumPy policy inference", 19, MUTED)
    compact(361, 822, 243, 70, GRAY, "Hidden opponent hand",
            "(card bag + numeric features)", size=19, body_size=16.5)
    panel(738, 788, 184, 104, GRAY, "Blind Value", ["h → Linear → V(s)"],
          center=True, title_size=20, body_size=17)
    panel(937, 788, 235, 104, GRAY, "Oracle Critic", [
        "[h, oracle embedding] → MLP", "Opponent hand · training only"],
        center=True, title_size=20, body_size=16, gap=25, header=39)
    panel(1187, 788, 214, 107, GRAY, "Auxiliary Head", [
        "h → MLP", "Future prizes · optional"],
        center=True, title_size=20, body_size=17, gap=25, header=39)
    global_x, oracle_x, hidden_x = 998 + 128 / 2, 937 + 235 / 2, 361 + 243 / 2
    route([(global_x, state_y + 86 / 2), (global_x, 752)],
          GRAY, arrow=False, dashed=True, width=1.4)
    for head_x in (830, 1294):
        route([(global_x, 752), (head_x, 752), (head_x, 788)],
              GRAY, dashed=True, width=1.4)
    route([(oracle_x, 752), (oracle_x, 788)], GRAY, dashed=True, width=1.4)
    route([(hidden_x, 892), (hidden_x, 914), (oracle_x, 914), (oracle_x, 892)],
          GRAY, dashed=True, width=1.4)

    label(20, 956, "Card / attack representation = learned ID embedding + projected static features.",
          17, MUTED)

    # Check actual font extents, including all multiline text inside boxes.
    fig.canvas.draw()
    inverse = ax.transData.inverted()
    errors = []
    for artist, (left, top, right, bottom) in text_bounds:
        bb = artist.get_window_extent(fig.canvas.get_renderer()).transformed(inverse)
        if bb.x0 < left - 1 or bb.x1 > right + 1 or bb.y0 < top - 1 or bb.y1 > bottom + 1:
            errors.append(artist.get_text())
    if errors:
        raise ValueError("Text exceeds panel bounds: " + "; ".join(errors))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "model_redrawn")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = create_figure()
    for extension in ("png", "svg", "pdf"):
        output = args.output.with_suffix("." + extension)
        fig.savefig(output, dpi=args.dpi, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
