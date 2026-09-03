"""Round-robin league between arena decks piloted by the archetype BC anchors.

Participants (default): from one meta day, the top --top-wr decks by arena
win rate (>= --min-n decided games) plus the top --top-use decks by usage,
deduplicated. Each deck is piloted by its archetype's BC anchor. Every pair
plays --games-per games (seats alternate via paired seeds). Two rankings:
  * total WR      -- unweighted mean over all games;
  * arena-wtd WR  -- per-opponent WR weighted by that opponent's arena share.

    python -m ptcg_rl.bc_league --day 0808 --games-per 100 --workers 20
    # 5-day high-WR tournament (top 10 by WR, n>=10, no usage filler):
    python -m ptcg_rl.bc_league --last-days 5 --top-wr 10 --top-use 0 \
        --min-n 10 --games-per 100 --workers 20
    python -m ptcg_rl.bc_league --opp-config configs/opponents_env.json  # legacy
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import glob
import itertools
import json
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .eval_local import run_match  # noqa: E402


def _accumulate_meta(paths: list[str]) -> tuple[dict, dict, int]:
    """fp -> [slots, points, decided], fp -> deck, total_slots."""
    import orjson
    from replay_loader import deck_fingerprint

    stats: dict[str, list[float]] = {}
    decks: dict[str, list[int]] = {}
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                rw = r.get("rewards") or [None, None]
                for side in (0, 1):
                    deck = r.get(f"deck{side}")
                    if not deck:
                        continue
                    fp = deck_fingerprint(deck)
                    st = stats.setdefault(fp, [0, 0.0, 0])
                    st[0] += 1
                    decks.setdefault(fp, deck)
                    if rw[0] is not None and rw[1] is not None:
                        st[1] += 1.0 if rw[side] > rw[1 - side] else \
                            (0.5 if rw[side] == rw[1 - side] else 0.0)
                        st[2] += 1
    total_slots = sum(st[0] for st in stats.values())
    return stats, decks, total_slots


def meta_participants(day: str | None, top_wr: int, top_use: int,
                      min_n: int, last_days: int = 1) -> list[dict]:
    """Top decks from one day (or last N days), each piloted by its BC anchor.

    When last_days > 1, pool the most recent N meta jsonl files and rank by
    aggregated arena WR / usage. Walks the WR ranking until ``top_wr`` decks
    with a usable BC anchor are collected (skips unmapped arches like
    ``other`` / ``mega_froslass``), then appends up to ``top_use`` usage picks.
    """
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from make_env_pool import ARCH_CKPT
    from replay_loader import classify_deck

    paths = sorted(glob.glob(os.path.join(ROOT, "data", "meta", "*.jsonl")))
    if not paths:
        raise SystemExit("no data/meta/*.jsonl found")
    if day:
        want = [p for p in paths if day in os.path.basename(p)]
        if not want:
            raise SystemExit(f"day {day} not found in data/meta/")
        paths = want
    elif last_days > 1:
        paths = paths[-last_days:]
    else:
        paths = [paths[-1]]
    print(f"meta source: {', '.join(os.path.basename(p) for p in paths)}")

    stats, decks, total_slots = _accumulate_meta(paths)
    eligible = [fp for fp in stats if stats[fp][2] >= min_n]
    by_wr = sorted(eligible, key=lambda fp: -stats[fp][1] / stats[fp][2])
    by_use = sorted(stats, key=lambda fp: -stats[fp][0])[:top_use] if top_use else []

    out_dir = os.path.join(ROOT, "data", "decks_league")
    os.makedirs(out_dir, exist_ok=True)

    def _try_add(fp: str, parts: list[dict], seen: set[str]) -> bool:
        if fp in seen:
            return False
        arch = classify_deck(decks[fp])
        ckpt = os.path.join(ROOT, ARCH_CKPT.get(arch, ""))
        if arch not in ARCH_CKPT or not os.path.exists(ckpt):
            print(f"[skip] {arch}__{fp}: no BC anchor for archetype '{arch}'")
            return False
        deck_csv = os.path.join(out_dir, f"{arch}_{fp}.csv")
        with open(deck_csv, "w", newline="\n") as f:
            f.write("\n".join(str(x) for x in sorted(decks[fp])) + "\n")
        st = stats[fp]
        parts.append({"name": f"{arch}__{fp}", "ckpt": os.path.abspath(ckpt),
                      "deck": os.path.abspath(deck_csv),
                      "share": st[0] / max(1, total_slots),
                      "arena_wr": st[1] / max(1, st[2]), "n": st[2]})
        seen.add(fp)
        return True

    parts: list[dict] = []
    seen: set[str] = set()
    for fp in by_wr:
        if len(parts) >= top_wr:
            break
        _try_add(fp, parts, seen)
    for fp in by_use:
        _try_add(fp, parts, seen)
    return parts


def load_participants(config_path: str) -> list[dict]:
    """One participant per unique ckpt; keeps that ckpt's highest-weight deck."""
    with open(config_path) as f:
        entries = json.load(f)
    best: dict[str, dict] = {}
    for e in entries:
        if e.get("kind") != "bc":
            continue
        ckpt = os.path.join(ROOT, e["ckpt"]) if not os.path.isabs(e["ckpt"]) else e["ckpt"]
        deck = os.path.join(ROOT, e["deck"]) if not os.path.isabs(e["deck"]) else e["deck"]
        if not (os.path.exists(ckpt) and os.path.exists(deck)):
            print(f"[skip] {e.get('name', '?')}: missing ckpt/deck")
            continue
        key = os.path.abspath(ckpt)
        if key not in best or e.get("weight", 0) > best[key]["weight"]:
            best[key] = {"name": e["name"].split("__")[0], "ckpt": key,
                         "deck": os.path.abspath(deck), "weight": e.get("weight", 0)}
    return list(best.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None,
                    help="meta day for participant selection (default: latest)")
    ap.add_argument("--last-days", type=int, default=1,
                    help="aggregate the most recent N meta days (ignored if "
                         "--day is set); use 5 for a 5-day WR tournament")
    ap.add_argument("--top-wr", type=int, default=10,
                    help="decks with the highest arena win rate")
    ap.add_argument("--top-use", type=int, default=10,
                    help="decks with the highest arena usage (0 = WR-only)")
    ap.add_argument("--min-n", type=int, default=30,
                    help="min decided arena games for the win-rate list")
    ap.add_argument("--opp-config", default=None,
                    help="legacy mode: participants from an opponents config "
                         "instead of the meta day")
    ap.add_argument("--games-per", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards_v3.npz"))
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    args = ap.parse_args()

    if args.opp_config:
        parts = load_participants(args.opp_config)
        for p in parts:  # legacy configs: pool weight stands in for arena share
            p.setdefault("share", p.get("weight", 0.0))
    else:
        parts = meta_participants(args.day, args.top_wr, args.top_use,
                                  args.min_n, last_days=args.last_days)
    if len(parts) < 2:
        raise SystemExit(f"need >=2 participants, got {len(parts)}")
    names = [p["name"] for p in parts]
    share = {p["name"]: float(p.get("share", 0.0)) for p in parts}
    print(f"{len(parts)} participants:")
    for p in parts:
        extra = ""
        if "arena_wr" in p:
            extra = (f" share={p['share']:6.2%} arena_wr={p['arena_wr']:.3f}"
                     f" (n={p['n']})")
        print(f"  {p['name']:<34}{extra}")
    n_pairs = len(parts) * (len(parts) - 1) // 2
    print(f"{n_pairs} pairs x {args.games_per} games\n")

    prior = os.path.abspath(args.prior) if os.path.exists(args.prior) else None
    # score[name] = [wins(+0.5 draw), decided games]; wr[a][b] = a's WR vs b
    score = {n: [0.0, 0] for n in names}
    wr: dict[str, dict[str, float]] = {n: {} for n in names}
    t0 = time.time()
    for i, (a, b) in enumerate(itertools.combinations(parts, 2), 1):
        cfg = {"ckpt": a["ckpt"], "cards": os.path.abspath(args.cards),
               "deck": a["deck"], "opp_deck": b["deck"],
               "opponent": "ckpt:" + b["ckpt"], "temperature": 0.0,
               "mcts": None, "sub_dir": None, "device": args.device,
               "prior": prior}
        r = run_match(cfg, args.games_per, args.workers, seed_base=args.seed_base)
        dec = r["w"] + r["l"] + r["d"]
        score[a["name"]][0] += r["w"] + 0.5 * r["d"]
        score[a["name"]][1] += dec
        score[b["name"]][0] += r["l"] + 0.5 * r["d"]
        score[b["name"]][1] += dec
        wr[a["name"]][b["name"]] = r["wr"]
        wr[b["name"]][a["name"]] = 1.0 - r["wr"]
        ab_note = f"  ABORTS={r['abort']}" if r["abort"] else ""
        print(f"[{i:2d}/{n_pairs}] {a['name']:<20} {r['wr']:.3f} vs {b['name']:<20}"
              f" ({r['games_per_s']:.1f} g/s){ab_note}", flush=True)

    order = sorted(names, key=lambda n: -(score[n][0] / max(1, score[n][1])))
    print(f"\n=== ranking (total WR over {len(parts) - 1} pairings) ===")
    for rank, n in enumerate(order, 1):
        pts, dec = score[n]
        print(f"{rank:2d}. {n:<34} {pts / max(1, dec):.3f}  ({pts:.1f}/{dec})")

    # arena-share-weighted: each opponent counts by its real-field frequency
    wtd = {}
    for n in names:
        num = den = 0.0
        for m in names:
            if m == n or m not in wr[n]:
                continue
            num += share[m] * wr[n][m]
            den += share[m]
        wtd[n] = num / den if den > 0 else 0.0
    order_w = sorted(names, key=lambda n: -wtd[n])
    print("\n=== ranking (arena-share-weighted WR) ===")
    for rank, n in enumerate(order_w, 1):
        print(f"{rank:2d}. {n:<34} {wtd[n]:.3f}  (own share={share[n]:.2%})")

    def _short(n: str) -> str:
        if "__" in n:
            a, fp = n.split("__", 1)
            return f"{a[:8]}.{fp[:4]}"
        return n[:13]

    short = [_short(n) for n in order]
    colw = max(len(s) for s in short) + 1
    print("\n=== pairwise matrix (row's WR vs column) ===")
    print(" " * 15 + "".join(s.rjust(colw) for s in short))
    for n in order:
        cells = "".join(
            (f"{wr[n][m]:.2f}".rjust(colw) if m != n else "--".rjust(colw))
            for m in order)
        print(f"{_short(n):<14} {cells}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
