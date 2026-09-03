"""Dump all card / attack metadata (including rules text) to JSON.

    python tools/dump_card_texts.py --out data/card_texts.json

The JSON feeds the LLM annotation pass that produces
data/card_skill_schema.json (semantic feature columns for cards_v3.npz).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "card_texts.json"))
    args = ap.parse_args()

    from cg.api import all_attack, all_card_data

    cards = []
    for c in all_card_data():
        cards.append({
            "id": c.cardId,
            "name": c.name,
            "cardType": int(c.cardType),
            "energyType": None if c.energyType is None else int(c.energyType),
            "hp": c.hp,
            "evolvesFrom": c.evolvesFrom,
            "basic": bool(c.basic), "stage1": bool(c.stage1),
            "stage2": bool(c.stage2), "ex": bool(c.ex),
            "megaEx": bool(c.megaEx), "tera": bool(c.tera),
            "aceSpec": bool(c.aceSpec),
            "skills": [{"name": s.name, "text": s.text} for s in c.skills],
            "attacks": list(c.attacks),
        })

    attacks = []
    for a in all_attack():
        attacks.append({
            "id": a.attackId,
            "name": getattr(a, "name", None),
            "damage": a.damage,
            "energies": [int(e) for e in a.energies],
            "text": a.text,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cards": cards, "attacks": attacks}, f,
                  ensure_ascii=False, indent=1)
    n_sk = sum(1 for c in cards if c["skills"])
    n_tx = sum(1 for a in attacks if (a["text"] or "").strip())
    print(f"written {args.out}: {len(cards)} cards ({n_sk} with skills), "
          f"{len(attacks)} attacks ({n_tx} with text)")


if __name__ == "__main__":
    main()
