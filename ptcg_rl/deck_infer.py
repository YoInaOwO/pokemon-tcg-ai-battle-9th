"""Opponent hidden-information inference for determinized search.

Offline: build a deck prior from the replay metadata (recency-weighted play
counts of exact 60-card lists):
    python -m ptcg_rl.deck_infer --build --out data/deck_prior.json

Runtime: given the current observation, sample plausible completions of the
opponent's hidden zones (deck / hand / prizes / facedown active) consistent
with everything visible, plus a partition of our own unseen cards.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

import numpy as np

from .features import ARCH_F, ARCH_UNK, N_ARCH

SOFT_MIN_COVER = 0.8  # a prior deck must explain >=80% of the seen multiset
SOFT_POWER = 6.0      # coverage^power weighting: near-exact matches dominate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Canonical archetype order for the posterior feature (must match
# tools/replay_loader.ARCHETYPE_RULES priority order; "other" last).
ARCH_NAMES = [
    "marnie_grimmsnarl", "cynthia", "ns_zoroark", "alakazam", "mega_lopunny",
    "mega_lucario", "dragapult", "kangaskhan_crustle", "kangaskhan_box",
    "mega_venusaur", "ogerpon_hydrapple", "grookey_dipplin", "ogerpon",
    "mega_froslass", "other",
]
assert len(ARCH_NAMES) == N_ARCH


def basics_from_cards(card_feats) -> set[int]:
    """Basic-Pokemon card ids from the static card feature table."""
    return set(np.nonzero((card_feats[:, 0] > 0.5) & (card_feats[:, 47] > 0.5))[0].tolist())

try:
    import orjson as _json

    def _loads(b):
        return _json.loads(b)
except ImportError:  # pragma: no cover
    def _loads(b):
        return json.loads(b)


# ------------------------------------------------------------------ offline

def build_prior(meta_dir: str, out_path: str, tau_days: float = 5.0, top: int = 400,
                only_days: str | None = None) -> None:
    """only_days: comma list like '0714,0715' to build a train-only prior
    (avoid leaking validation-day meta into arch features)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from replay_loader import classify_deck, deck_fingerprint

    days = sorted(glob.glob(os.path.join(meta_dir, "*.jsonl")))
    if only_days:
        want = set(only_days.split(","))
        days = [p for p in days
                if os.path.splitext(os.path.basename(p))[0] in want]
    if not days:
        sys.exit("no meta jsonl found; run tools/etl_all.py first")
    weights: dict[str, float] = {}
    decks: dict[str, list[int]] = {}
    for di, path in enumerate(days):
        age = len(days) - 1 - di
        w_day = 2.718281828 ** (-age / tau_days)
        with open(path, "rb") as f:
            for line in f:
                r = _loads(line)
                for side in (0, 1):
                    deck = r.get(f"deck{side}")
                    if not deck:
                        continue
                    fp = deck_fingerprint(deck)
                    weights[fp] = weights.get(fp, 0.0) + w_day
                    if fp not in decks:
                        decks[fp] = sorted(deck)
    ranked = sorted(weights.items(), key=lambda x: -x[1])[:top]
    total = sum(w for _, w in ranked)
    payload = [{"cards": decks[fp], "w": round(w / total, 6),
                "arch": classify_deck(decks[fp])} for fp, w in ranked]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    n_arch = len(set(p["arch"] for p in payload))
    print(f"written {out_path}: {len(payload)} decks, {n_arch} archetypes, "
          f"weight coverage of top10 = {sum(p['w'] for p in payload[:10]):.2f}")


# ------------------------------------------------------------------ runtime

def observed_multiset(cur: dict, player: int) -> Counter:
    """Multiset of the given player's cards visible outside deck/hand/prize.

    Includes the `looking` zone: cards there sit outside deck/hand counts
    (verified empirically: zone totals conserve to 60 with looking counted
    separately), so they belong to the visible part of the 60."""
    seen: Counter = Counter()

    def add_card(c):
        if c and c.get("id"):
            seen[int(c["id"])] += 1

    pl = cur["players"][player]
    for pk in (pl.get("active") or []) + (pl.get("bench") or []):
        if not pk:
            continue  # facedown active: unknown, handled separately
        add_card(pk)
        for c in (pk.get("energyCards") or []) + (pk.get("tools") or []) + (pk.get("preEvolution") or []):
            add_card(c)
    for c in pl.get("discard") or []:
        add_card(c)
    for c in pl.get("prize") or []:
        add_card(c)  # revealed prizes only (None entries skipped)
    for c in cur.get("looking") or []:
        if c and c.get("playerIndex") == player:
            add_card(c)
    st = cur.get("stadium") or []
    if st and st[0] and st[0].get("playerIndex") == player:
        add_card(st[0])
    return seen


class DeckPrior:
    def __init__(self, path: str, basic_pokemon_ids: set[int]):
        with open(path) as f:
            entries = json.load(f)
        self.decks = [Counter(e["cards"]) for e in entries]
        self.deck_lists = [e["cards"] for e in entries]
        self.weights = [e["w"] for e in entries]
        self.arch_idx = [ARCH_NAMES.index(e.get("arch", "other"))
                         if e.get("arch", "other") in ARCH_NAMES else N_ARCH - 1
                         for e in entries]
        self.basics = basic_pokemon_ids
        # global archetype distribution (posterior fallback when nothing matches)
        self.global_arch = np.zeros(N_ARCH, dtype=np.float32)
        for ai, w in zip(self.arch_idx, self.weights):
            self.global_arch[ai] += w
        self.global_arch /= max(float(self.global_arch.sum()), 1e-9)
        # global card frequency for fallback completion
        freq: Counter = Counter()
        for c, w in zip(self.decks, self.weights):
            for cid, n in c.items():
                freq[cid] += n * w
        self.common = [cid for cid, _ in freq.most_common(60)]
        self.common_basics = [c for c in self.common if c in basic_pokemon_ids] or [
            next(iter(basic_pokemon_ids))]

    def match_idx(self, seen: Counter) -> list[int]:
        """Indices of prior decks consistent with the observed multiset."""
        items = seen.items()
        return [i for i, cnt in enumerate(self.decks)
                if all(cnt.get(c, 0) >= k for c, k in items)]

    def soft_match(self, seen: Counter) -> list[tuple[int, float]]:
        """[(deck idx, coverage)] where coverage is the fraction of the seen
        multiset the deck contains. Exact matches have coverage 1.0; a 1-2
        card tech swap on a 15-card-seen board still scores ~0.9 instead of
        falling off the exact-subset cliff. Decks below SOFT_MIN_COVER are
        dropped."""
        total = sum(seen.values())
        if total == 0:
            return [(i, 1.0) for i in range(len(self.decks))]
        items = list(seen.items())
        out = []
        for i, cnt in enumerate(self.decks):
            short = 0
            for c, k in items:
                d = k - cnt.get(c, 0)
                if d > 0:
                    short += d
            cov = 1.0 - short / total
            if cov >= SOFT_MIN_COVER:
                out.append((i, cov))
        return out

    def candidates(self, seen: Counter, max_n: int = 24) -> list[tuple[list[int], float]]:
        """Decks consistent with observed cards, with renormalized weights."""
        idxs = self.match_idx(seen)[:max_n]
        return [(self.deck_lists[i], self.weights[i]) for i in idxs]

    def arch_feature(self, obs: dict, extra_ids=None) -> np.ndarray:
        """Opponent-model feature vector (ARCH_F): archetype posterior from
        cards seen so far + match statistics. Cheap (one scan of the prior).

        extra_ids: additional revealed opponent cards (e.g. cards known to sit
        in their hand from the game-log memory) merged into the multiset."""
        cur = obs["current"]
        op = 1 - int(cur["yourIndex"])
        seen = observed_multiset(cur, op)
        if extra_ids:
            for c in extra_ids:
                if c > 0:
                    seen[int(c)] += 1
        v = np.zeros(ARCH_F, dtype=np.float32)
        cands = self.soft_match(seen)
        if cands:
            ws = [self.weights[i] * (cov ** SOFT_POWER) for i, cov in cands]
            tot = sum(ws)
            for (i, _), w in zip(cands, ws):
                v[self.arch_idx[i]] += w
            v[:N_ARCH] /= max(tot, 1e-9)
            v[N_ARCH] = min(len(cands) / 32.0, 1.0)         # candidate breadth
            v[N_ARCH + 1] = max(ws) / max(tot, 1e-9)        # posterior concentration
            v[N_ARCH + 2] = min(tot, 1.0)                   # meta coverage of matches
        else:
            # nothing in the prior explains this deck: say so explicitly
            # instead of smearing the global meta distribution over it
            v[ARCH_UNK] = 1.0
        v[N_ARCH + 3] = min(sum(seen.values()) / 60.0, 1.0)  # information revealed
        v[N_ARCH + 4] = float(cur["players"][op].get("deckCount") or 0) / 60.0
        return v

    def sample_unseen(self, seen: Counter, n_unseen: int, need_basic: bool,
                      rng: random.Random) -> list[int]:
        """Sample the opponent's unseen cards (multiset of size n_unseen)."""
        cands = self.candidates(seen)
        if cands:
            cards, _ = rng.choices(cands, weights=[w for _, w in cands], k=1)[0]
            pool = Counter(cards)
            pool.subtract(seen)
            unseen = [c for c, k in pool.items() for _ in range(k) if k > 0]
        else:
            unseen = []
        # fix size mismatches (wrong guess / >60 effects): trim or pad
        rng.shuffle(unseen)
        if len(unseen) > n_unseen:
            unseen = unseen[:n_unseen]
        while len(unseen) < n_unseen:
            unseen.append(rng.choice(self.common))
        if need_basic and not any(c in self.basics for c in unseen) and unseen:
            unseen[rng.randrange(len(unseen))] = rng.choice(self.common_basics)
        return unseen


def sample_determinization(obs: dict, my_deck_list: list[int], prior: DeckPrior,
                           rng: random.Random,
                           known_hand: list[int] | None = None) -> dict | None:
    """-> kwargs for search begin: your_deck, your_prize, opponent_deck,
    opponent_prize, opponent_hand, opponent_active. None if inconsistent.

    known_hand: cards known to be in the opponent's hand (log memory); they
    are pinned into the sampled hand instead of floating anywhere."""
    cur = obs["current"]
    me = int(cur["yourIndex"])
    op = 1 - me
    pme, pop = cur["players"][me], cur["players"][op]

    # deck-view selections (`select.deck`) list exact deck contents while the
    # cards stay counted in deckCount (verified: len == deckCount in all
    # replays). Split by owner; ownerless entries default to the acting side.
    deck_view = [c for c in ((obs.get("select") or {}).get("deck") or [])
                 if c and c.get("id")]
    my_deck_known = [int(c["id"]) for c in deck_view
                     if c.get("playerIndex") in (me, None)]
    op_deck_known = [int(c["id"]) for c in deck_view
                     if c.get("playerIndex") == op]

    # ---- my unseen: known 60 list minus visible zones minus my hand
    # (observed_multiset already counts my revealed prizes and looking cards
    #  -> my_unseen is exactly deck + unrevealed prize slots; when the deck
    #  contents are revealed too, it is exactly the hidden prize slots)
    mine = Counter(my_deck_list)
    mine.subtract(observed_multiset(cur, me))
    for c in pme.get("hand") or []:
        if c and c.get("id"):
            mine[int(c["id"])] -= 1
    for cid in my_deck_known:
        mine[cid] -= 1
    my_unseen = [c for c, k in mine.items() for _ in range(max(0, k))]
    rng.shuffle(my_unseen)
    my_prize_all = pme.get("prize") or []
    my_prize_revealed = [int(c["id"]) for c in my_prize_all if c and c.get("id")]
    n_deck = int(pme.get("deckCount") or 0)
    n_prize_hidden = len(my_prize_all) - len(my_prize_revealed)
    n_deck_unknown = 0 if my_deck_known else n_deck
    need = n_deck_unknown + n_prize_hidden
    if len(my_unseen) > need:
        my_unseen = my_unseen[:need]
    while len(my_unseen) < need:
        my_unseen.append(rng.choice(my_deck_list))
    if my_deck_known:
        # verified in engine source (Search.h): when the select has a deck
        # view (state.selectDeck), SearchBegin skips config.myDeck entirely
        # and keeps the real deck, so this value is ignored; otherwise the
        # passed order IS the draw order, and a shuffled sample is exactly
        # the right belief over hidden orderings.
        your_deck = list(my_deck_known)
        rng.shuffle(your_deck)
    else:
        your_deck = my_unseen[:n_deck_unknown]
    # prize slots are positional in SearchBegin (the engine wipes both prize
    # lists and refills index-by-index): revealed prizes must sit at their
    # true slot, hidden slots take the sampled cards
    hid = iter(my_unseen[n_deck_unknown:need])
    your_prize = [int(c["id"]) if c and c.get("id") else next(hid)
                  for c in my_prize_all]

    # ---- opponent unseen
    seen_op = observed_multiset(cur, op)
    op_active = pop.get("active") or []
    facedown = bool(op_active) and op_active[0] is None
    op_prize_all = pop.get("prize") or []
    op_prize_revealed = [int(c["id"]) for c in op_prize_all if c and c.get("id")]
    n_deck_op = int(pop.get("deckCount") or 0)
    n_hand_op = int(pop.get("handCount") or 0)
    n_prize_op_hidden = len(op_prize_all) - len(op_prize_revealed)
    # cards known to sit in their hand / revealed in their deck are "seen"
    # for deck matching and are pinned to their zones below
    pinned = [int(c) for c in (known_hand or []) if c > 0][:n_hand_op]
    for c in pinned:
        seen_op[c] += 1
    pinned_deck = op_deck_known[:n_deck_op]
    for c in pinned_deck:
        seen_op[c] += 1
    n_deck_op_unknown = n_deck_op - len(pinned_deck)
    n_unseen = (n_deck_op_unknown + n_hand_op - len(pinned) + n_prize_op_hidden
                + (1 if facedown else 0))
    unseen = prior.sample_unseen(seen_op, n_unseen, need_basic=True, rng=rng)

    opponent_active: list[int] = []
    if facedown:
        bi = next((i for i, c in enumerate(unseen) if c in prior.basics), None)
        if bi is None:
            return None
        opponent_active = [unseen.pop(bi)]
    opponent_deck = pinned_deck + unseen[:n_deck_op_unknown]
    hid = iter(unseen[n_deck_op_unknown:n_deck_op_unknown + n_prize_op_hidden])
    opponent_prize = [int(c["id"]) if c and c.get("id") else next(hid)
                      for c in op_prize_all]
    opponent_hand = pinned + unseen[n_deck_op_unknown + n_prize_op_hidden:]
    # setup phase: engine requires >=1 basic pokemon in opponent deck
    if n_deck_op > 0 and not any(c in prior.basics for c in opponent_deck):
        swap = next((i for i, c in enumerate(opponent_hand) if c in prior.basics), None)
        if swap is not None:
            opponent_deck[0], opponent_hand[swap] = opponent_hand[swap], opponent_deck[0]
        else:
            opponent_deck[0] = rng.choice(prior.common_basics)
    return {
        "your_deck": your_deck, "your_prize": your_prize,
        "opponent_deck": opponent_deck, "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand, "opponent_active": opponent_active,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--meta-dir", default=os.path.join(ROOT, "data", "meta"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--tau-days", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--days", default=None,
                    help="comma list of days to include (train-only prior)")
    args = ap.parse_args()
    if args.build:
        build_prior(args.meta_dir, args.out, args.tau_days, args.top, args.days)


if __name__ == "__main__":
    main()
