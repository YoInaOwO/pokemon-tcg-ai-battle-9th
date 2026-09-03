"""Build an arena-replica opponent pool config from the latest day's meta data.

Takes the top-N exact decks (fingerprints) by slot share, maps each to its
archetype BC model, exports the deck lists, and writes an opponents config:
<mirror_w> self-play + (1 - mirror_w) split by normalized arena share.

    python tools/make_env_pool.py                     # latest day, top 20
    python tools/make_env_pool.py --day 20250808 --top 20 --mirror 0.10
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orjson

from replay_loader import classify_deck, deck_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# archetype -> BC anchor checkpoint (train these before running PPO).
# v6 anchors: won or tied every archetype in the v5-vs-v6 acceptance league
# (logs/league_v5v6.log, 2026-08-11); v5 paths kept for reference below.
ARCH_CKPT = {
    a: f"runs/bc_{a}_v6/model.pt"
    for a in ("marnie_grimmsnarl", "alakazam", "mega_lopunny", "mega_lucario",
              "dragapult", "kangaskhan_crustle", "ogerpon", "kangaskhan_box",
              "grookey_dipplin", "ogerpon_hydrapple", "cynthia", "ns_zoroark")
}
ARCH_CKPT_V5 = {
    "marnie_grimmsnarl": "runs/bc_grimm/model.pt",
    "alakazam": "runs/bc_alakazam/model.pt",
    "mega_lopunny": "runs/bc_lopunny/model.pt",
    "mega_lucario": "runs/bc_lucario/model.pt",
    "dragapult": "runs/bc_dragapult/model.pt",
    "kangaskhan_crustle": "runs/bc_crustle/model.pt",
    "ogerpon": "runs/bc_ogerpon/model.pt",
    "kangaskhan_box": "runs/bc_kanga_box/model.pt",
    "grookey_dipplin": "runs/bc_grookey/model.pt",
    "ogerpon_hydrapple": "runs/bc_hydrapple/model.pt",
    "cynthia": "runs/bc_cynthia/model.pt",
    "ns_zoroark": "runs/bc_zoroark/model.pt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None, help="YYYYMMDD; default = latest jsonl")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--mirror", type=float, default=0.05)
    ap.add_argument("--random", type=float, default=0.10,
                    help="prior-deck opponent share: decks sampled from the "
                         "replay prior, piloted by the matching archetype BC "
                         "anchor (uniform random actions when unmapped)")
    ap.add_argument("--league", type=float, default=0.0,
                    help="share reserved for own past PPO snapshots "
                         "(kind=league); 0.4 matched the ab_hydra78 armB setup")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "opponents_env.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(ROOT, "data", "meta", "*.jsonl")))
    if not paths:
        sys.exit("no data/meta/*.jsonl found")
    path = paths[-1]
    if args.day:
        want = [p for p in paths if args.day in os.path.basename(p)]
        if not want:
            sys.exit(f"day {args.day} not found in data/meta/")
        path = want[0]
    print(f"source: {path}")

    slots: collections.Counter[str] = collections.Counter()
    decks: dict[str, list[int]] = {}
    with open(path, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            for side in (0, 1):
                deck = r.get(f"deck{side}")
                if not deck:
                    continue
                fp = deck_fingerprint(deck)
                slots[fp] += 1
                decks.setdefault(fp, deck)

    total_slots = sum(slots.values())
    picked: list[tuple[str, str, int]] = []  # (fp, arch, n)
    for fp, n in slots.most_common():
        if len(picked) >= args.top:
            break
        arch = classify_deck(decks[fp])
        if arch not in ARCH_CKPT:
            print(f"[warn] skip {arch}__{fp} ({n} slots, {n / total_slots:.1%}): "
                  f"no BC model mapping for archetype '{arch}'")
            continue
        picked.append((fp, arch, n))

    picked_slots = sum(n for _, _, n in picked)
    print(f"picked {len(picked)} decks covering {picked_slots / total_slots:.1%} of slots")

    entries: list[dict] = [{"kind": "mirror", "weight": round(args.mirror, 4)}]
    if args.random > 0:
        entries.append({"kind": "random", "weight": round(args.random, 4)})
    if args.league > 0:
        entries.append({"kind": "league", "weight": round(args.league, 4)})
    bc_mass = 1.0 - args.mirror - args.random - args.league
    for fp, arch, n in picked:
        deck_csv = os.path.join("data", "decks", f"{arch}_{fp}.csv")
        abs_csv = os.path.join(ROOT, deck_csv)
        os.makedirs(os.path.dirname(abs_csv), exist_ok=True)
        with open(abs_csv, "w", newline="\n") as f:
            f.write("\n".join(str(x) for x in sorted(decks[fp])) + "\n")
        w = bc_mass * n / picked_slots
        entries.append({"kind": "bc", "name": f"{arch}__{fp}",
                        "ckpt": ARCH_CKPT[arch], "deck": deck_csv.replace(os.sep, "/"),
                        "weight": round(w, 4)})
        missing = "" if os.path.exists(os.path.join(ROOT, ARCH_CKPT[arch])) else "  [ckpt MISSING]"
        print(f"  {arch}__{fp:<12} share={n / total_slots:6.2%}  w={w:.4f}{missing}")

    with open(args.out, "w", newline="\n") as f:
        f.write(json.dumps(entries, indent=2) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
