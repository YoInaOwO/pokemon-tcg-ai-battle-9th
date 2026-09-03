"""Semantic annotation of card skills / trainer text / attack text.

    python tools/annotate_schema.py            # -> data/card_skill_schema.json

Reads data/card_texts.json (tools/dump_card_texts.py), extracts SK_COLS
semantic columns per card and ATK2_COLS per attack with rule-based parsing,
then applies hand-audited corrections from data/card_schema_overrides.json
(reviewed for every card appearing in arena meta decks). The output feeds
cards_v3.npz (ptcg_rl/cards.py --schema).

Column values are floats in [0, ~1.5]; multi-hot flags are 0/1, magnitudes
are normalized counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------- card columns

SK_COLS = [
    "draw",            # 0  draw cards
    "search_deck",     # 1  searches own deck
    "fetch_pokemon",   # 2  fetches/moves pokemon cards
    "fetch_energy",    # 3  fetches/moves energy cards
    "fetch_trainer",   # 4  fetches trainer cards
    "to_hand",         # 5  puts cards into hand
    "to_bench",        # 6  puts pokemon directly onto bench
    "energy_accel",    # 7  attaches energy outside the manual attach
    "from_discard",    # 8  recovers from a discard pile
    "heal",            # 9
    "damage_boost",    # 10 own attacks do more damage
    "damage_reduce",   # 11 takes less damage / damage reduction
    "gust",            # 12 forces a switch of the OPPONENT's active
    "self_switch",     # 13 switches own active
    "status",          # 14 inflicts a special condition
    "disrupt_hand",    # 15 messes with opponent's hand
    "disrupt_energy",  # 16 removes opponent's energy
    "mill",            # 17 deck -> discard
    "shuffle_draw",    # 18 hand refresh (shuffle + draw)
    "protect",         # 19 prevents damage/effects
    "retreat_help",    # 20 reduces/zeroes retreat cost
    "evo_accel",       # 21 accelerates evolution (Rare Candy etc.)
    "is_ability",      # 22 pokemon Ability (vs trainer/energy rules text)
    "once_per_turn",   # 23
    "passive",         # 24 continuous static effect
    "on_play",         # 25 triggers when played from hand
    "cost_discard",    # 26 requires discarding as a cost
    "coin_flip",       # 27
    "team_wide",       # 28 affects each/all of OUR pokemon
    "opp_wide",        # 29 affects each/all of THEIR pokemon
    "n_cards",         # 30 cards drawn/fetched magnitude (/7)
    "n_hp",            # 31 heal / reduce / boost magnitude (/120)
]

# ------------------------------------------------------------- attack columns

ATK2_COLS = [
    "dmg_times",        # 0  "... damage for each / times ..."
    "scale_energy",     # 1  scales with energy in play
    "scale_bench",      # 2  scales with bench size
    "scale_damage",     # 3  scales with damage counters / hp lost
    "scale_prize",      # 4  scales with prize count
    "scale_cards",      # 5  scales with cards (hand/discard/deck)
    "cond_extra",       # 6  conditional bonus damage ("if ..., ... more")
    "snipe",            # 7  targeted damage to a chosen pokemon
    "spread",           # 8  damage to each/all opposing pokemon
    "self_recoil",      # 9  damages itself / own side
    "cost_self_energy", # 10 discards own energy as cost/effect
    "lock_next",        # 11 can't attack / can't use next turn
    "protect_next",     # 12 prevents damage next turn
    "heal_self",        # 13
    "energy_move",      # 14 attaches/moves energy
    "deck_interact",    # 15 searches / manipulates deck
    "hand_disrupt",     # 16 attacks opponent's hand
    "extra_dmg",        # 17 flat bonus damage magnitude (/200)
    "mult_dmg",         # 18 per-unit scaling magnitude (/100)
    "status_any",       # 19 inflicts any special condition
    "self_switch",      # 20 switches/bounces itself out of the active spot
    "pierce",           # 21 damage ignores effects / weakness-resistance
    "gust",             # 22 forces the opponent's active to the bench
    "draw",             # 23 draws cards as an attack effect
]

_NUM = r"(\d+)"


def _norm(t: str) -> str:
    t = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", t or "")
    t = t.replace("\ufffd", "e")  # mojibake in Pok<?>mon
    return re.sub(r"\s+", " ", t).lower()


def _any(text: str, *pats: str) -> float:
    return 1.0 if any(re.search(p, text) for p in pats) else 0.0


def _max_num(text: str, *pats: str) -> int:
    best = 0
    for p in pats:
        for m in re.finditer(p, text):
            for g in m.groups():
                if g and g.isdigit():
                    best = max(best, int(g))
    return best


def card_vector(card: dict) -> list[float]:
    """SK_COLS floats from a card's combined skills text."""
    texts = [_norm(s.get("text") or "") for s in card.get("skills") or []]
    t = " || ".join(x for x in texts if x)
    v = [0.0] * len(SK_COLS)
    if not t:
        return v
    c = {n: i for i, n in enumerate(SK_COLS)}
    is_pokemon = card["cardType"] == 0
    v[c["draw"]] = _any(t, r"\bdraw ", r"\bdraws ")
    v[c["search_deck"]] = _any(t, r"search (your|their) deck")
    v[c["fetch_pokemon"]] = _any(t, r"(search|look).{0,80}pok(e|.)mon",
                                 r"put.{0,60}pok(e|.)mon.{0,40}(hand|bench)")
    v[c["fetch_energy"]] = _any(t, r"(search|look).{0,80}energy",
                                r"put.{0,50}energy.{0,30}hand")
    v[c["fetch_trainer"]] = _any(t, r"(search|look).{0,80}(trainer|item|supporter|stadium|tool)")
    v[c["to_hand"]] = _any(t, r"(put|take).{0,60}(into|to) (your|their) hand")
    v[c["to_bench"]] = _any(t, r"put.{0,60}onto (your|their) bench")
    # attach(?!ed): "as long as this card is attached" is a location clause,
    # not acceleration; energy may appear on either side of the verb
    # ("attach an energy" / "energy cards and attach them")
    v[c["energy_accel"]] = _any(t, r"attach(?!ed).{0,80}energy",
                                r"energy.{0,60}\battach(?!ed)")
    v[c["from_discard"]] = _any(t, r"from your discard pile")
    v[c["heal"]] = _any(t, r"\bheal\b")
    v[c["damage_boost"]] = _any(t, r"do(es)? \d+ more damage",
                                r"\+\s?\d+ damage", r"more damage")
    v[c["damage_reduce"]] = _any(t, r"\d+ less damage", r"damage is reduced",
                                 r"takes? less damage")
    v[c["gust"]] = _any(t, r"switch.{0,70}opponent'?s (active|benched)",
                        r"opponent switches", r"switch it with (the|their) active")
    v[c["self_switch"]] = _any(t, r"switch (this pok|your active)")
    v[c["status"]] = _any(t, r"(asleep|burned|confused|paralyzed|poisoned)")
    v[c["disrupt_hand"]] = _any(t, r"opponent.{0,50}(discards?|shuffles?|reveals?).{0,40}hand",
                                r"opponent'?s hand")
    v[c["disrupt_energy"]] = _any(t, r"discard.{0,60}energy.{0,50}(from your opponent|opponent'?s)")
    v[c["mill"]] = _any(t, r"discard the top \d* ?cards? of.{0,30}deck")
    v[c["shuffle_draw"]] = _any(t, r"shuffles? (their|your|his or her) hand",
                                r"shuffle your hand into your deck")
    v[c["protect"]] = _any(t, r"prevent all", r"has no effect", r"can'?t be affected",
                           r"protected from")
    v[c["retreat_help"]] = _any(t, r"retreat cost.{0,40}(less|is \{?0|nothing)",
                                r"has no retreat cost", r"free retreat")
    v[c["evo_accel"]] = _any(t, r"evolv.{0,80}(skip|as (though|if)|first turn|turn (it|they) (was|were) played|this turn)")
    v[c["is_ability"]] = 1.0 if is_pokemon else 0.0
    v[c["once_per_turn"]] = _any(t, r"once during your turn")
    v[c["passive"]] = 1.0 if (is_pokemon and _any(t, r"as long as", r"all of your",
                                                  r"whenever", r"can'?t be")
                              and not v[c["once_per_turn"]]) else 0.0
    v[c["on_play"]] = _any(t, r"when you play this pok")
    v[c["cost_discard"]] = _any(t, r"(you must )?discard.{0,50}(from your hand )?in order to use",
                                r"discard \d+ (other )?cards? from your hand")
    v[c["coin_flip"]] = _any(t, r"\bflip\b")
    v[c["team_wide"]] = _any(t, r"(each|all) of your pok")
    v[c["opp_wide"]] = _any(t, r"each of your opponent'?s")
    v[c["n_cards"]] = min(_max_num(
        t, r"draw " + _NUM + r" cards?", r"draw cards until you have " + _NUM,
        r"draw up to " + _NUM, r"for up to " + _NUM,
        r"search your deck for " + _NUM), 7) / 7.0
    v[c["n_hp"]] = min(_max_num(
        t, r"heal " + _NUM, _NUM + r" less damage", _NUM + r" more damage",
        _NUM + r" damage counter"), 120) / 120.0
    return v


def attack_vector(atk: dict) -> list[float]:
    t = _norm(atk.get("text") or "")
    v = [0.0] * len(ATK2_COLS)
    if not t:
        return v
    c = {n: i for i, n in enumerate(ATK2_COLS)}
    v[c["dmg_times"]] = _any(t, r"damage (for each|times)")
    v[c["scale_energy"]] = _any(t, r"for each.{0,50}energy")
    v[c["scale_bench"]] = _any(t, r"for each.{0,40}bench")
    v[c["scale_damage"]] = _any(t, r"for each damage counter", r"equal to the damage")
    v[c["scale_prize"]] = _any(t, r"for each prize")
    v[c["scale_cards"]] = _any(t, r"for each card")
    v[c["cond_extra"]] = _any(t, r"if .{0,100}(this attack does|more damage)")
    v[c["snipe"]] = _any(t, r"damage to 1 of your opponent'?s",
                         r"choose.{0,50}opponent'?s.{0,40}damage",
                         r"damage counters? on your opponent'?s benched")
    v[c["spread"]] = _any(t, r"damage to each")
    v[c["self_recoil"]] = _any(t, r"damage to itself", r"also does .{0,20}damage to (this|itself)")
    v[c["cost_self_energy"]] = _any(t, r"discard.{0,40}energy from this pok")
    v[c["lock_next"]] = _any(t, r"can'?t (attack|use).{0,60}next turn",
                             r"during your next turn.{0,40}can'?t (attack|use)")
    v[c["protect_next"]] = _any(t, r"prevent all (damage|effects?).{0,80}next turn",
                                r"during your opponent'?s next turn.{0,80}less damage",
                                r"during your opponent'?s next turn.{0,50}prevent all")
    v[c["heal_self"]] = _any(t, r"heal.{0,40}from (this|1 of your|your) pok")
    v[c["energy_move"]] = _any(t, r"(attach|move).{0,60}energy")
    v[c["deck_interact"]] = _any(t, r"search your deck", r"top .{0,25}of your deck",
                                 r"from your discard pile")
    v[c["hand_disrupt"]] = _any(t, r"opponent.{0,50}hand")
    v[c["extra_dmg"]] = min(_max_num(t, _NUM + r" more damage"), 200) / 200.0
    v[c["mult_dmg"]] = min(_max_num(t, _NUM + r" damage (for each|times)",
                                    r"does " + _NUM + r" damage for each"), 100) / 100.0
    v[c["status_any"]] = _any(t, r"(asleep|burned|confused|paralyzed|poisoned)")
    v[c["self_switch"]] = _any(t, r"switch this pok",
                               r"put this pok.{0,50}into your hand")
    v[c["pierce"]] = _any(t, r"isn'?t affected by")
    v[c["gust"]] = _any(t, r"switch out your opponent'?s active",
                        r"opponent'?s active pok.{0,40}to the bench")
    v[c["draw"]] = _any(t, r"\bdraw (a card|\d+ cards?)", r"draw cards until")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=os.path.join(ROOT, "data", "card_texts.json"))
    ap.add_argument("--overrides", default=os.path.join(ROOT, "data", "card_schema_overrides.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "card_skill_schema.json"))
    ap.add_argument("--meta-ids", default=os.path.join(ROOT, "build", "_meta_card_ids.json"))
    args = ap.parse_args()

    with open(args.texts, encoding="utf-8") as f:
        data = json.load(f)

    cards = {str(cd["id"]): card_vector(cd) for cd in data["cards"]}
    attacks = {str(a["id"]): attack_vector(a) for a in data["attacks"]}

    n_over = 0
    if os.path.exists(args.overrides):
        with open(args.overrides, encoding="utf-8") as f:
            over = json.load(f)
        sk_idx = {n: i for i, n in enumerate(SK_COLS)}
        atk_idx = {n: i for i, n in enumerate(ATK2_COLS)}
        for cid, patch in (over.get("cards") or {}).items():
            for col, val in patch.items():
                cards[cid][sk_idx[col]] = float(val)
            n_over += 1
        for aid, patch in (over.get("attacks") or {}).items():
            for col, val in patch.items():
                attacks[aid][atk_idx[col]] = float(val)
            n_over += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"sk_cols": SK_COLS, "atk2_cols": ATK2_COLS,
                   "cards": cards, "attacks": attacks}, f)

    nz_c = sum(1 for v in cards.values() if any(v))
    nz_a = sum(1 for v in attacks.values() if any(v))
    print(f"written {args.out}: {len(cards)} cards ({nz_c} annotated), "
          f"{len(attacks)} attacks ({nz_a} annotated), {n_over} overrides")
    if os.path.exists(args.meta_ids):
        with open(args.meta_ids) as f:
            meta = [str(i) for i in json.load(f)]
        with_skill = [cd for cd in data["cards"]
                      if str(cd["id"]) in meta and cd.get("skills")]
        missed = [cd["id"] for cd in with_skill if not any(cards[str(cd["id"])][:22])]
        print(f"meta cards with skills: {len(with_skill)}; "
              f"no effect column extracted for: {missed}")


if __name__ == "__main__":
    main()
