"""Plot wr_pool + archetype trends from a selfplay_ppo stdout log."""

from __future__ import annotations

import argparse
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ema(vals: list[float | None], span: int = 15) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    prev = None
    alpha = 2.0 / (span + 1)
    for i, v in enumerate(vals):
        if v is None:
            continue
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{'upd'"):
                continue
            try:
                rows.append(ast.literal_eval(line))
            except (SyntaxError, ValueError):
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "ppo_wr_trend.png"))
    ap.add_argument("--title", default="PPO win-rate trend")
    args = ap.parse_args()

    rows = load_rows(args.log)
    if not rows:
        raise SystemExit(f"no update rows in {args.log}")

    upd = [r["upd"] for r in rows]
    pool = [r.get("wr_pool") for r in rows]
    best_wr, best_upd = max((r["best"][0], r["best"][1]) for r in rows)
    print(f"updates: {len(rows)} (last {upd[-1]}), "
          f"best wr_pool={best_wr:.4f} @ upd {best_upd}, "
          f"latest={pool[-1]:.4f}, elapsed={rows[-1].get('elapsed_min', 0):.0f} min")

    ARCHS = [
        ("bc:marnie_grimmsnarl", "grimmsnarl (26.3%)", "#1f77b4"),
        ("bc:alakazam", "alakazam (13.8%)", "#d62728"),
        ("bc:mega_lucario", "lucario (3.2%)", "#9467bd"),
        ("bc:kangaskhan_crustle", "crustle (5.8%)", "#ff7f0e"),
        ("bc:dragapult", "dragapult (7.0%)", "#8c564b"),
        ("bc:kangaskhan_box", "kanga_box (3.5%)", "#e377c2"),
        ("bc:mega_lopunny", "lopunny-field", "#2ca02c"),
        ("mirror", "mirror", "#7f7f7f"),
    ]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    fig.suptitle(args.title, fontsize=13, y=0.98)

    ax0.plot(upd, pool, color="#bbbbbb", linewidth=0.8, alpha=0.6, label="wr_pool raw")
    ax0.plot(upd, ema(pool, 15), color="#1f77b4", linewidth=2.0, label="wr_pool EMA(15)")
    if best_upd > 0:
        ax0.axvline(best_upd, color="#2ca02c", linestyle="--", alpha=0.5, linewidth=1)
        ax0.scatter([best_upd], [best_wr], color="#2ca02c", s=40, zorder=5,
                    label=f"best {best_wr:.3f} @ {best_upd}")
    ax0.set_ylabel("arena-weighted WR")
    ax0.set_ylim(0.35, 0.78)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right", fontsize=9)

    for key, label, color in ARCHS:
        raw = [(r.get("wr_arch") or {}).get(key) for r in rows]
        if all(v is None for v in raw):
            continue
        ax1.plot(upd, ema(raw, 20), color=color, linewidth=1.6, label=label)
    ax1.set_xlabel("update")
    ax1.set_ylabel("per-upd arch WR (EMA 20)")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
