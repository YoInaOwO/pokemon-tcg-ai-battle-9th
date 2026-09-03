"""Static card / attack feature tables.

Build once (requires the cg DLL):
    python -m ptcg_rl.cards --out data/cards.npz

The resulting npz ships with training code AND the submission bundle so the
feature space is bit-identical everywhere. Arrays:
    card_feats   (N_CARDS, CARD_F)  float32   static per-card features
    card_attacks (N_CARDS, 2)       int16     attack ids (0 = none)
    attack_feats (N_ATTACKS, ATK_F) float32   static per-attack features
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import sys

import numpy as np

N_CARDS = 1268    # card ids 1..1267, 0 = none/unknown
N_ATTACKS = 1557  # attack ids 1..1556, 0 = none
N_ENERGY = 12     # EnergyType 0..11
ATK_F = 35        # 15 legacy + 20 effect-text flags (v2 table)
CARD_F = 57 + 2 * ATK_F
# v3 table (--schema): +24 semantic attack columns (LLM/hand-audited,
# data/card_skill_schema.json) and +32 skill/trainer-text columns per card.
# Layout: [57 base][32 skill][atk0 59][atk1 59] = 207.
ATK2_F = 24
SK_F = 32
ATK_F_V3 = ATK_F + ATK2_F
CARD_F_V3 = 57 + SK_F + 2 * ATK_F_V3

# Semantic effect flags parsed from the attack text: unseen cards with a
# familiar effect profile land near seen ones instead of relying purely on
# the id embedding (which is random noise for cards absent from training).
_EFFECT_FLAGS = [
    ("draw", r"\bdraw\b"),
    ("search_deck", r"search your deck"),
    ("disc_opp_energy", r"discard[^.]{0,60}energy[^.]{0,40}opponent"),
    ("disc_own_energy", r"discard[^.]{0,60}energy[^.]{0,40}this pok"),
    ("heal", r"\bheal\b"),
    ("bench_dmg", r"damage to[^.]{0,40}benched pok"),
    ("spread", r"each of your opponent"),
    ("self_dmg", r"damage to itself"),
    ("asleep", r"asleep"),
    ("poison", r"poisoned"),
    ("paralyze", r"paralyzed"),
    ("burn", r"burned"),
    ("confuse", r"confused"),
    ("gust", r"switch[^.]{0,50}opponent|opponent[^.]{0,10}switches"),
    ("prevent", r"prevent all"),
    ("coin", r"\bflip\b"),
    ("scaling", r"more damage|damage for each"),
    ("energy_accel", r"attach[^.]{0,60}energy"),
    ("mill", r"top[^.]{0,25}card[^.]{0,25}deck"),
    ("hand_disrupt", r"opponent[^.]{0,30}hand"),
]
assert 15 + len(_EFFECT_FLAGS) == ATK_F
_FLAG_RE = [re.compile(p) for _, p in _EFFECT_FLAGS]


def _norm_text(t: str) -> str:
    return re.sub(r"[\u2018\u2019\u201c\u201d]", "'", t or "").lower()


def _attack_vec(a) -> np.ndarray:
    v = np.zeros(ATK_F, dtype=np.float32)
    v[0] = a.damage / 300.0
    v[1] = len(a.energies) / 5.0
    for e in a.energies:
        if 0 <= int(e) < N_ENERGY:
            v[2 + int(e)] += 1.0 / 3.0
    text = _norm_text(a.text)
    v[14] = 1.0 if text.strip() else 0.0
    for j, rx in enumerate(_FLAG_RE):
        if rx.search(text):
            v[15 + j] = 1.0
    return v


def build(out_path: str, schema_path: str | None = None) -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "sample_submission", "sample_submission"))
    from cg.api import all_attack, all_card_data

    cards = all_card_data()
    attacks = all_attack()
    assert max(c.cardId for c in cards) < N_CARDS
    assert max(a.attackId for a in attacks) < N_ATTACKS

    sk_map: dict[str, list[float]] = {}
    atk2_map: dict[str, list[float]] = {}
    if schema_path:  # v3: semantic columns from the annotated schema
        import json
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        assert len(schema["sk_cols"]) == SK_F, len(schema["sk_cols"])
        assert len(schema["atk2_cols"]) == ATK2_F, len(schema["atk2_cols"])
        sk_map = schema["cards"]
        atk2_map = schema["attacks"]
    atk_f = ATK_F_V3 if schema_path else ATK_F
    card_f = CARD_F_V3 if schema_path else CARD_F

    atk_feats = np.zeros((N_ATTACKS, atk_f), dtype=np.float32)
    for a in attacks:
        atk_feats[a.attackId, :ATK_F] = _attack_vec(a)
        if schema_path:
            extra = atk2_map.get(str(a.attackId))
            if extra:
                atk_feats[a.attackId, ATK_F:] = np.asarray(extra, dtype=np.float32)

    card_feats = np.zeros((N_CARDS, card_f), dtype=np.float32)
    card_attacks = np.zeros((N_CARDS, 2), dtype=np.int16)
    for c in cards:
        v = np.zeros(card_f, dtype=np.float32)
        o = 0
        v[o + int(c.cardType)] = 1.0; o += 7                     # cardType
        if c.energyType is not None:
            v[o + int(c.energyType)] = 1.0
        o += N_ENERGY                                            # energyType
        v[o] = 1.0 if c.weakness is None else 0.0
        if c.weakness is not None:
            v[o + 1 + int(c.weakness)] = 1.0
        o += 13                                                  # weakness
        v[o] = 1.0 if c.resistance is None else 0.0
        if c.resistance is not None:
            v[o + 1 + int(c.resistance)] = 1.0
        o += 13                                                  # resistance
        v[o] = c.hp / 340.0; v[o + 1] = c.retreatCost / 5.0; o += 2
        for i, flag in enumerate([c.basic, c.stage1, c.stage2, c.ex, c.megaEx, c.tera, c.aceSpec]):
            v[o + i] = 1.0 if flag else 0.0
        o += 7
        v[o] = len(c.skills) / 2.0; v[o + 1] = len(c.attacks) / 2.0; o += 2
        prize_value = 3 if c.megaEx else (2 if c.ex else 1)
        v[o] = prize_value / 3.0; o += 1
        assert o == 57
        if schema_path:                                          # skill semantics
            sk = sk_map.get(str(c.cardId))
            if sk:
                v[o:o + SK_F] = np.asarray(sk, dtype=np.float32)
            o += SK_F
        for i, aid in enumerate(c.attacks[:2]):
            v[o + i * atk_f: o + (i + 1) * atk_f] = atk_feats[aid]
            card_attacks[c.cardId, i] = aid
        o += 2 * atk_f
        assert o == card_f
        card_feats[c.cardId] = v

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, card_feats=card_feats, card_attacks=card_attacks,
                        attack_feats=atk_feats)
    print(f"written {out_path}  cards={card_feats.shape} attacks={atk_feats.shape}")


def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in ("card_feats", "card_attacks", "attack_feats")}


def load_matching(path: str, card_f: int, atk_f: int) -> tuple[dict[str, np.ndarray], str]:
    """Load the card table whose feature widths match a checkpoint.

    Old checkpoints were trained on the 87/15-column table, new ones on the
    thicker one; both kinds coexist (league anchors vs the live policy). If
    the given file is missing or mismatches, probe versioned siblings
    (cards*.npz in the same directory) before failing.
    -> (tables, actual path used)."""
    tab = None
    if os.path.exists(path):
        tab = load(path)
        if tab["card_feats"].shape[1] == card_f and tab["attack_feats"].shape[1] == atk_f:
            return tab, path
    for cand in sorted(_glob.glob(os.path.join(os.path.dirname(path) or ".",
                                               "cards*.npz"))):
        if os.path.abspath(cand) == os.path.abspath(path):
            continue
        t = load(cand)
        if t["card_feats"].shape[1] == card_f and t["attack_feats"].shape[1] == atk_f:
            return t, cand
    raise ValueError(f"no card table matching widths {card_f}/{atk_f} next to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cards.npz")
    ap.add_argument("--schema", default=None,
                    help="card_skill_schema.json -> build the v3 table")
    args = ap.parse_args()
    build(args.out, schema_path=args.schema)
