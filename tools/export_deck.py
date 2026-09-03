"""Export the exact 60-card list of a deck fingerprint to a deck.csv.

    python tools/export_deck.py --fp 220eddd2 --out data/decks/mega_lopunny_220eddd2.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orjson

from replay_loader import classify_deck, deck_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_deck(fp: str) -> list[int] | None:
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "meta", "*.jsonl")), reverse=True):
        with open(path, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                for side in (0, 1):
                    deck = r.get(f"deck{side}")
                    if deck and deck_fingerprint(deck) == fp:
                        return deck
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    deck = find_deck(args.fp)
    if deck is None:
        sys.exit(f"fingerprint {args.fp} not found in data/meta/*.jsonl")
    label = f"{classify_deck(deck)}_{args.fp}"
    out = args.out or os.path.join(ROOT, "data", "decks", f"{label}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as f:
        f.write("\n".join(str(x) for x in sorted(deck)) + "\n")
    print(f"written {out} ({label})")


if __name__ == "__main__":
    main()
