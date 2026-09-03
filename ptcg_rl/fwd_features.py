"""Engine lookahead ("forward") option features.

For every MAIN-phase single-pick decision, execute each candidate option once
inside an engine search sandbox and read out its public consequences as 24
numeric columns appended to that option's feature vector (features.OPT_FWD).
This turns "what does this card do" from an id-embedding memorization problem
into an explicit input: the scoring head directly sees "option 3: 180 damage,
KO, I take 2 prizes, turn ends".

Determinization is deliberately crude and fixed-seed:
  * our hidden zones (deck + facedown prizes) = our known 60-list minus all
    visible zones, shuffled with a fixed seed;
  * opponent hidden zones = filler cards (lowest-id basic energy, plus a
    lowest-id basic pokemon where the engine demands one).
The extracted fields depend only on public information (damage, KO, prize
deltas, hand/deck count deltas, turn handover, instant win/loss), so the
filler never leaks into them -- verified by the multi-seed invariance check
in tools/verify_fwd.py. Never call SearchEnd here (it would wipe unrelated
search state on a shared agent pointer); every opened node is released
individually.

Layout (indices into the fwd block, base = OPT_F - OPT_FWD):
  single-step (post = state right after executing the option):
     0 valid            probe succeeded for this option
     1 dmg              damage dealt to opp active (/300)
     2 ko               we took >=1 prize (opp pokemon KO'd)
     3 my_prizes        prizes we took this step (/3)
     4 opp_prizes       prizes they took this step (/3)
     5 hand_delta       my hand count change (clip /10)
     6 deck_delta       my deck count change (clip /10)
     7 self_dmg         damage our active took (/300)
     8 my_ko            they took >=1 prize (our pokemon KO'd)
     9 landed_coin      post state is a coin-flip decision (chance-gated)
    10 win              instant win
    11 lose             instant loss
  macro expansion (forced chains auto-stepped, coins split 0.5/0.5, up to
  MAX_LEAVES leaves / MAX_NODES engine steps; aggregated over leaf probs):
    12 macro_valid
    13 E[dmg]   14 E[ko]   15 E[my_prizes]   16 E[opp_prizes]
    17 E[win]   18 E[lose] 19 min dmg        20 max dmg
    21 E[turn_end]        22 E[hand_delta]   23 n_leaves (/MAX_LEAVES)
  legality delta (v6, n_cols >= 28): what this option unlocks -- "Rare Candy
  makes Charizard's attack legal" becomes an explicit input instead of a
  memorized combo:
    24 still_me           post state is still our decision
    25 post_n_options     options at the post decision (/40)
    26 post_n_attacks     legal ATTACK options at the post decision (/8)
    27 attack_delta       (post - root legal attacks) / 4, clipped to [-1,1]
  col 25 stays zero when the post decision is a deck search or when cards
  were drawn (those counts reflect sandbox shuffle order, not learnable
  legality); cols 26-27 are board-determined and always safe.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from .deck_infer import observed_multiset
from .features import OPT_FWD, OPT_FWD_V6

FWD_MAX_OPTIONS = 32       # v5 probe cap (rest stay zero)
FWD_MAX_OPTIONS_V6 = 64    # v6: measured decision max is 64 options
MAX_LEAVES = 8         # macro expansion caps
MAX_NODES = 24
_FWD_SEED = 20260817   # fixed: train/inference parity requires one seed
_CTX_COIN = 46         # SelectContext.COIN_HEAD (manual_coin decision node)


def filler_ids(card_feats: np.ndarray) -> tuple[int, int]:
    """(lowest basic-energy id, lowest basic-pokemon id) from the card table."""
    energy = np.nonzero(card_feats[:, 5] > 0.5)[0]           # CardType BASIC_ENERGY
    basic = np.nonzero((card_feats[:, 0] > 0.5) & (card_feats[:, 47] > 0.5))[0]
    if len(energy) == 0 or len(basic) == 0:
        raise ValueError("card table lacks basic energy / basic pokemon")
    return int(energy[0]), int(basic[0])


def _snap(obs: dict, me: int) -> dict:
    """Public-state metrics snapshot from the root player's perspective."""
    cur = obs.get("current") or {}
    pls = cur.get("players") or [{}, {}]

    def active(p: int) -> tuple[int, float]:
        a = pls[p].get("active") or []
        pk = a[0] if a else None
        if not pk:
            return 0, 0.0
        return int(pk.get("id") or 0), float(pk.get("hp") or 0)

    op = 1 - me
    my_id, my_hp = active(me)
    op_id, op_hp = active(op)
    res = cur.get("result")
    return {
        "result": int(res) if res is not None else -1,
        "my_prize": len(pls[me].get("prize") or []),
        "op_prize": len(pls[op].get("prize") or []),
        "my_hand": int(pls[me].get("handCount") or 0),
        "my_deck": int(pls[me].get("deckCount") or 0),
        "my_act_id": my_id, "my_act_hp": my_hp,
        "op_act_id": op_id, "op_act_hp": op_hp,
    }


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _delta(pre: dict, post: dict, me: int) -> dict:
    """Consequence fields between two snapshots (root player perspective)."""
    myp = max(0, pre["my_prize"] - post["my_prize"])    # prizes I took
    oppp = max(0, pre["op_prize"] - post["op_prize"])
    dmg = 0.0
    if post["op_act_id"] == pre["op_act_id"] and pre["op_act_id"] > 0:
        dmg = max(0.0, pre["op_act_hp"] - post["op_act_hp"])
    elif myp > 0:
        dmg = pre["op_act_hp"]  # KO'd and replaced: at least the remaining hp
    self_dmg = 0.0
    if post["my_act_id"] == pre["my_act_id"] and pre["my_act_id"] > 0:
        self_dmg = max(0.0, pre["my_act_hp"] - post["my_act_hp"])
    elif oppp > 0:
        self_dmg = pre["my_act_hp"]
    res = post["result"]
    return {
        "dmg": min(dmg, 300.0) / 300.0,
        "ko": 1.0 if myp > 0 else 0.0,
        "myp": min(myp, 3) / 3.0,
        "oppp": min(oppp, 3) / 3.0,
        "hand_d": _clip1((post["my_hand"] - pre["my_hand"]) / 10.0),
        "deck_d": _clip1((post["my_deck"] - pre["my_deck"]) / 10.0),
        "self_dmg": min(self_dmg, 300.0) / 300.0,
        "my_ko": 1.0 if oppp > 0 else 0.0,
        "win": 1.0 if res == me else 0.0,
        "lose": 1.0 if res == (1 - me) else 0.0,
    }


def _is_coin(sel: dict | None) -> bool:
    return bool(sel) and int(sel.get("context") or 0) == _CTX_COIN


class ForwardProbe:
    """Per-process probe: one SearchIO arena, one deck list at a time.

    Construction never raises on a missing engine; the probe just stays
    disabled (probe() returns None) so pure-numpy consumers degrade to
    zero columns instead of crashing.
    """

    def __init__(self, card_feats: np.ndarray, deck: list[int] | None = None,
                 n_cols: int = OPT_FWD, max_options: int = FWD_MAX_OPTIONS):
        self.filler_energy, self.filler_basic = filler_ids(np.asarray(card_feats))
        self.deck: list[int] | None = list(deck) if deck else None
        self.n_cols = n_cols            # 24 (v5) or 28 (v6 legality deltas)
        self.max_options = max_options  # 32 (v5) or 64 (v6)
        self.io = None
        self.disabled = False
        self.stats = {"probes": 0, "options": 0, "begin_fail": 0, "errors": 0}

    def set_deck(self, deck: list[int]) -> None:
        self.deck = list(deck)

    def _ensure_io(self) -> bool:
        if self.disabled:
            return False
        if self.io is None:
            try:
                from .search_io import SearchIO
                self.io = SearchIO()
            except Exception:  # noqa: BLE001  engine unavailable: stay numpy-only
                self.disabled = True
                return False
        return True

    # ------------------------------------------------------------- internals

    def _determinize(self, obs: dict) -> dict | None:
        """Fixed-seed filler determinization (see module docstring)."""
        cur = obs["current"]
        me = int(cur["yourIndex"])
        op = 1 - me
        pme, pop = cur["players"][me], cur["players"][op]
        rng = random.Random(_FWD_SEED)

        # ---- our side: real remaining cards, fixed-seed shuffle
        deck_view = [c for c in ((obs.get("select") or {}).get("deck") or [])
                     if c and c.get("id")]
        my_deck_known = [int(c["id"]) for c in deck_view
                         if c.get("playerIndex") in (me, None)]
        mine = Counter(self.deck)
        mine.subtract(observed_multiset(cur, me))
        for c in pme.get("hand") or []:
            if c and c.get("id"):
                mine[int(c["id"])] -= 1
        for cid in my_deck_known:
            mine[cid] -= 1
        my_unseen = [c for c, k in mine.items() for _ in range(max(0, k))]
        rng.shuffle(my_unseen)
        my_prize_all = pme.get("prize") or []
        n_prize_hidden = sum(1 for c in my_prize_all if not (c and c.get("id")))
        n_deck = int(pme.get("deckCount") or 0)
        n_deck_unknown = 0 if my_deck_known else n_deck
        need = n_deck_unknown + n_prize_hidden
        if len(my_unseen) > need:
            my_unseen = my_unseen[:need]
        while len(my_unseen) < need:
            my_unseen.append(rng.choice(self.deck))
        your_deck = my_unseen[:n_deck_unknown]
        hid = iter(my_unseen[n_deck_unknown:need])
        your_prize = [int(c["id"]) if c and c.get("id") else next(hid)
                      for c in my_prize_all]

        # ---- opponent side: pure filler (fields extracted are filler-blind)
        op_active = pop.get("active") or []
        facedown = bool(op_active) and op_active[0] is None
        n_deck_op = int(pop.get("deckCount") or 0)
        n_hand_op = int(pop.get("handCount") or 0)
        opponent_deck = [self.filler_energy] * n_deck_op
        if n_deck_op > 0:
            opponent_deck[0] = self.filler_basic  # engine wants >=1 basic in deck
        opponent_hand = [self.filler_energy] * n_hand_op
        opponent_prize = [int(c["id"]) if c and c.get("id") else self.filler_energy
                          for c in (pop.get("prize") or [])]
        return {
            "your_deck": your_deck, "your_prize": your_prize,
            "opponent_deck": opponent_deck, "opponent_prize": opponent_prize,
            "opponent_hand": opponent_hand,
            "opponent_active": [self.filler_basic] if facedown else [],
        }

    def _macro(self, root_state: dict, pre: dict, me: int,
               sids: list[int]) -> list[tuple[float, dict, float]]:
        """Expand forced chains / coin splits -> [(prob, leaf_snap, turn_end)]."""
        stack = [(1.0, root_state)]
        leaves: list[tuple[float, dict, float]] = []
        nodes = 0
        while stack and len(leaves) < MAX_LEAVES and nodes < MAX_NODES:
            p, state = stack.pop()
            obs = state["observation"]
            cur = obs.get("current") or {}
            sel = obs.get("select")
            res = cur.get("result")
            if sel is None or (res is not None and int(res) >= 0):
                leaves.append((p, _snap(obs, me), 1.0))     # terminal
                continue
            opts = sel.get("option") or []
            n = len(opts)
            mx = int(sel.get("maxCount") or 0)
            if _is_coin(sel) and n == 2:                    # chance node: split
                for a in (0, 1):
                    nst = self.io.step(state["searchId"], [a])
                    nodes += 1
                    if nst is None:
                        continue
                    sids.append(int(nst["searchId"]))
                    stack.append((p * 0.5, nst))
                continue
            if n == 0 or (n == 1 and mx <= 1):              # forced chain
                nst = self.io.step(state["searchId"], [0] if n == 1 else [])
                nodes += 1
                if nst is None:
                    leaves.append((p, _snap(obs, me), 0.0))
                    continue
                sids.append(int(nst["searchId"]))
                stack.append((p, nst))
                continue
            # a real decision: leaf; turn ended iff it is the opponent's
            yi = int(cur.get("yourIndex", me))
            leaves.append((p, _snap(obs, me), 1.0 if yi != me else 0.0))
        for p, state in stack:                              # cap hit: close out
            obs = state["observation"]
            yi = int((obs.get("current") or {}).get("yourIndex", me))
            leaves.append((p, _snap(obs, me), 1.0 if yi != me else 0.0))
        return leaves

    # ---------------------------------------------------------------- public

    def probe(self, obs: dict) -> np.ndarray | None:
        """-> (K, n_cols) float32, or None when not applicable/unavailable."""
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        K = len(opts)
        if (int(sel.get("type") or 0) != 0 or K < 2
                or int(sel.get("maxCount") or 0) != 1
                or not obs.get("search_begin_input")
                or not self.deck or not self._ensure_io()):
            return None
        try:
            det = self._determinize(obs)
        except Exception:  # noqa: BLE001  malformed obs: skip quietly
            self.stats["errors"] += 1
            return None
        me = int(obs["current"]["yourIndex"])
        root_atk = sum(1 for o in opts if int(o.get("type") or 0) == 13)

        def _my_board(o: dict) -> tuple:
            p = ((o.get("current") or {}).get("players") or [{}, {}])[me]
            ids = [int(pk.get("id") or 0)
                   for zone in (p.get("active") or [], p.get("bench") or [])
                   for pk in zone if pk]
            return tuple(sorted(ids))

        def _my_hand(o: dict) -> Counter:
            p = ((o.get("current") or {}).get("players") or [{}, {}])[me]
            return Counter(int(c["id"]) for c in (p.get("hand") or [])
                           if c and c.get("id"))
        out = np.zeros((K, self.n_cols), dtype=np.float32)
        sids: list[int] = []
        try:
            root = self.io.begin(obs, det, manual_coin=True)
            if root is None:
                self.stats["begin_fail"] += 1
                return None
            root_sid = int(root["searchId"])
            sids.append(root_sid)
            pre = _snap(root["observation"], me)
            pre_board = _my_board(root["observation"])
            pre_hand = _my_hand(root["observation"])
            self.stats["probes"] += 1
            for k in range(min(K, self.max_options)):
                st = self.io.step(root_sid, [k])
                if st is None:
                    continue
                sids.append(int(st["searchId"]))
                post_obs = st["observation"]
                d = _delta(pre, _snap(post_obs, me), me)
                row = out[k]
                row[0] = 1.0
                row[1] = d["dmg"]; row[2] = d["ko"]
                row[3] = d["myp"]; row[4] = d["oppp"]
                row[5] = d["hand_d"]; row[6] = d["deck_d"]
                row[7] = d["self_dmg"]; row[8] = d["my_ko"]
                row[9] = 1.0 if _is_coin(post_obs.get("select")) else 0.0
                row[10] = d["win"]; row[11] = d["lose"]
                if self.n_cols >= OPT_FWD_V6:
                    pcur = post_obs.get("current") or {}
                    ps = post_obs.get("select")
                    pres = pcur.get("result")
                    still_me = (ps is not None
                                and int(pcur.get("yourIndex", -1)) == me
                                and (pres is None or int(pres) < 0))
                    row[24] = 1.0 if still_me else 0.0
                    if still_me:
                        popts = ps.get("option") or []
                        p_atk = sum(1 for o in popts
                                    if int(o.get("type") or 0) == 13)
                        # attack legality is board-determined -> deterministic
                        row[26] = min(p_atk, 8) / 8.0
                        row[27] = _clip1((p_atk - root_atk) / 4.0)
                        # total option count is only deterministic when the
                        # option touched no hidden zone: no new card entered
                        # the hand (post hand is a sub-multiset of the root
                        # hand), deck count unchanged, no fetch onto the
                        # board, and not a deck-search select -- otherwise
                        # sandbox shuffle order would leak into it as noise
                        pme = ((post_obs.get("current") or {})
                               .get("players") or [{}, {}])[me]
                        post_deck = int(pme.get("deckCount") or 0)
                        if (post_deck == pre["my_deck"]
                                and not _my_hand(post_obs) - pre_hand
                                and _my_board(post_obs) == pre_board
                                and not ps.get("deck")):
                            row[25] = min(len(popts), 40) / 40.0
                self.stats["options"] += 1
                leaves = self._macro(st, pre, me, sids)
                tot = sum(p for p, _, _ in leaves)
                if tot <= 0:
                    continue
                ds = [(p / tot, _delta(pre, s, me), te) for p, s, te in leaves]
                row[12] = 1.0
                row[13] = sum(p * d2["dmg"] for p, d2, _ in ds)
                row[14] = sum(p * d2["ko"] for p, d2, _ in ds)
                row[15] = sum(p * d2["myp"] for p, d2, _ in ds)
                row[16] = sum(p * d2["oppp"] for p, d2, _ in ds)
                row[17] = sum(p * d2["win"] for p, d2, _ in ds)
                row[18] = sum(p * d2["lose"] for p, d2, _ in ds)
                row[19] = min(d2["dmg"] for _, d2, _ in ds)
                row[20] = max(d2["dmg"] for _, d2, _ in ds)
                row[21] = sum(p * te for p, _, te in ds)
                row[22] = sum(p * d2["hand_d"] for p, d2, _ in ds)
                row[23] = len(ds) / float(MAX_LEAVES)
        except Exception:  # noqa: BLE001  engine hiccup: keep rows probed so far
            self.stats["errors"] += 1
        finally:
            for sid in sids:
                try:
                    self.io.release(sid)
                except Exception:  # noqa: BLE001
                    pass
        return out
