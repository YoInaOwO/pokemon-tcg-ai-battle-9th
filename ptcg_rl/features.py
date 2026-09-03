"""Observation / option encoding shared by BC extraction, training, and the
live agent. numpy only — no torch, no DLL.

Perspective is always the acting player ("me" = players[current.yourIndex]).

encode_obs(obs_dict, mem=None) -> dict of arrays (feature version 4):
    glob       (GLOB_F,)    f32   global numerics
    ctx        (CTX_F,)     f32   select-level features
    ctx_ids    (2,)         i16   [effect card id, context card id]
    board      (12, BOARD_F)f32   my active, my bench x5, opp active, opp bench x5
    board_ids  (12, 4)      i16   [pokemon, first tool, energy card 0, energy card 1]
    hand_ids   (MAX_HAND,)  i16   my hand card ids (0-padded)
    look_ids   (MAX_LOOK,)  i16   looking cards (0 = facedown/none)
    look_feats (MAX_LOOK,2) f32   [present, is_mine]
    disc_ids   (2, MAX_DISC)i16   my / opp discard pile ids (0-padded)
    deck_ids   (MAX_DISC,)  i16   select.deck contents when searching own deck
    logs       (LOG_F,)     f32   summary of events since our last decision
    logs_ids   (6,)         i16   [my last attack, opp last attack, 4 cards revealed
                                   into opp hand]  (attack ids index the attack table)
    mem        (MEM_F,)     f32   whole-game history summary (needs LogMemory)
    mem_ids    (20,)        i16   [my last 4 attacks, opp last 4 attacks,
                                   up to 12 cards known to be in opp hand]
    stadium_id ()           i16
    opt_feats  (K, OPT_F)   f32   per-option features (K = len(select.option))
    opt_card   (K,)         i16   resolved primary card id
    opt_tgt    (K,)         i16   resolved target pokemon card id
    opt_atk    (K,)         i16   attack id

LogMemory accumulates both players' action history across a whole game from
the per-decision log windows (verified disjoint & complete per side). Call
mem.update(obs) exactly once per decision before encode_obs(obs, mem).

Oracle side-channel (training only, never enters the encoder):
    ora        (ORA_F,)     f32   oracle_feats(...)
    ora_ids    (ORA_IDS,)   i16   opponent hand snapshot at their last decision
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

FEAT_VERSION = 5  # v5: +24 engine-lookahead option cols, arch unknown bit
# v6 (encode_obs(..., v6=True)): 8 bench slots per side (Area Zero), deck-bag
# filled with own remaining hidden pool, 4 legality-delta probe columns,
# HP norms to 400, count head to 24 classes. v5 output stays bit-identical.
FEAT_V6 = 6
SEQ_PAD = 24          # padded expert/action sequence length (measured max 21)
ORDERED_CTX = (10, 34)  # TO_DECK_BOTTOM, SKILL_ORDER: selection order matters


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

GLOB_F = 40
N_ARCH = 15   # 14 archetype rules + "other" (order fixed in deck_infer.ARCH_NAMES)
ARCH_F = 21   # N_ARCH posterior + 6 extras; optional model input (config arch_f)
ARCH_UNK = N_ARCH + 5  # explicit "opponent deck unknown" bit (soft-match miss)
CTX_F = 68
BOARD_F = 32
OPT_FWD = 24  # engine-lookahead columns appended per option (fwd_features)
OPT_FWD_V6 = 28  # v6: +4 legality-delta columns (appended after the v5 24)
OPT_F = 60 + OPT_FWD
OPT_F_V6 = 60 + OPT_FWD_V6
LOG_F = 16
MEM_F = 10
N_MEM_ATK = 4    # attack-history length per side
N_MEM_REV = 12   # known-opponent-hand bag size
ORA_F = 4
ORA_IDS = 20
N_BOARD = 12
N_BENCH_V6 = 8   # Area Zero Underdepths: benchMax reaches 8 (seen 7 in replays)
N_BOARD_V6 = 2 * (1 + N_BENCH_V6)
MAX_HAND = 30  # measured max 29 across replays
MAX_LOOK = 8   # measured max 7
MAX_DISC = 60
MAX_COUNT = 22  # count-head classes 0..21 (measured max selection: 21)
MAX_COUNT_V6 = 24  # forced selects reach k=23 in replays
HP_DIV = 340.0
HP_DIV_V6 = 400.0  # printed HP reaches 380+

_N_SELECT_TYPE = 11
_N_CONTEXT = 52
_N_OPT_TYPE = 17
_N_AREA = 13  # 0 = none, 1..12 = AreaType


def _cid(card) -> int:
    """Card dict -> id (0 for facedown None)."""
    if not card:
        return 0
    return int(card.get("id") or 0)


class LogMemory:
    """Cumulative both-players action history for one side of one game.

    update(obs) must be called exactly once per decision of this side, in
    order. Tracks attack history, per-side play/attach counts, and the
    publicly revealed hand knowledge in BOTH directions (cards we know in
    their hand, cards they know in ours -- tutors reveal to both players;
    removed when seen played/attached/evolved/moved; cleared when the hand
    is shuffled away facedown). Only op_known feeds our own features;
    my_known exists for the swapped() view at opponent search nodes.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.my_atk: list[int] = []
        self.op_atk: list[int] = []
        self.counts = np.zeros(6, dtype=np.float32)  # atk/play/attach x me/op
        self.op_known: list[int] = []   # cards we know sit in THEIR hand
        self.my_known: list[int] = []   # cards they know sit in OUR hand

    def clone(self) -> "LogMemory":
        """Cheap deep copy (for in-search per-node memory)."""
        m = LogMemory.__new__(LogMemory)
        m.my_atk = list(self.my_atk)
        m.op_atk = list(self.op_atk)
        m.counts = self.counts.copy()
        m.op_known = list(self.op_known)
        m.my_known = list(self.my_known)
        return m

    def swapped(self) -> "LogMemory":
        """Public history seen from the other side. Their knowledge of our
        hand is the publicly revealed part we track in my_known."""
        m = LogMemory.__new__(LogMemory)
        m.my_atk = list(self.op_atk)
        m.op_atk = list(self.my_atk)
        m.counts = self.counts[[1, 0, 3, 2, 5, 4]].copy()
        m.op_known = list(self.my_known)
        m.my_known = list(self.op_known)
        return m

    def _known_remove(self, known: list[int], cid: int) -> None:
        try:
            known.remove(cid)
        except ValueError:
            pass

    def _op_remove(self, cid: int) -> None:
        self._known_remove(self.op_known, cid)

    def update(self, obs: dict, as_player: int | None = None) -> None:
        """as_player: fix the "me" perspective regardless of obs.yourIndex
        (search nodes alternate perspectives; memory must not flip sides)."""
        cur = obs.get("current")
        if not cur:
            return
        me = int(cur["yourIndex"]) if as_player is None else int(as_player)
        op = 1 - me
        for e in obs.get("logs") or []:
            t = e.get("type")
            p = e.get("playerIndex")
            known = (self.op_known if p == op
                     else self.my_known if p == me else None)
            if t == 15:
                aid = int(e.get("attackId") or 0)
                if p == me:
                    self.my_atk.append(aid)
                    self.counts[0] += 1
                elif p == op:
                    self.op_atk.append(aid)
                    self.counts[1] += 1
            elif t == 10:
                self.counts[2 if p == me else 3] += 1
                if known is not None:
                    self._known_remove(known, int(e.get("cardId") or 0))
            elif t == 11:
                self.counts[4 if p == me else 5] += 1
                if known is not None:
                    self._known_remove(known, int(e.get("cardId") or 0))
            elif t == 12 and known is not None:
                self._known_remove(known, int(e.get("cardId") or 0))
            elif t == 6 and known is not None:
                cid = int(e.get("cardId") or 0)
                if e.get("toArea") == 2 and cid > 0:
                    if len(known) < 2 * N_MEM_REV:
                        known.append(cid)
                elif e.get("fromArea") == 2:
                    self._known_remove(known, cid)
            elif t == 7 and known is not None and e.get("fromArea") == 2:
                known.clear()  # hand left facedown (shuffle-draw etc.)

    def arrays(self, op_hand_count: int) -> tuple[np.ndarray, np.ndarray]:
        m = np.zeros(MEM_F, dtype=np.float32)
        ids = np.zeros(2 * N_MEM_ATK + N_MEM_REV, dtype=np.int16)
        m[0] = min(len(self.my_atk), 12) / 12.0
        m[1] = min(len(self.op_atk), 12) / 12.0
        m[2] = min(self.counts[2], 20.0) / 20.0
        m[3] = min(self.counts[3], 20.0) / 20.0
        m[4] = min(self.counts[4], 12.0) / 12.0
        m[5] = min(self.counts[5], 12.0) / 12.0
        m[6] = min(len(self.op_known), N_MEM_REV) / N_MEM_REV
        m[7] = min(len(self.op_known) / op_hand_count, 1.0) if op_hand_count > 0 else 0.0
        for j, aid in enumerate(self.my_atk[-N_MEM_ATK:][::-1]):
            ids[j] = aid
        for j, aid in enumerate(self.op_atk[-N_MEM_ATK:][::-1]):
            ids[N_MEM_ATK + j] = aid
        for j, cid in enumerate(self.op_known[-N_MEM_REV:]):
            ids[2 * N_MEM_ATK + j] = cid
        return m, ids


def oracle_feats(snap_ids: list[int], hand_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Opponent hand snapshot (their last decision) -> training-only critic input."""
    o = np.zeros(ORA_F, dtype=np.float32)
    ids = np.zeros(ORA_IDS, dtype=np.int16)
    n = min(len(snap_ids), ORA_IDS)
    for j in range(n):
        ids[j] = snap_ids[j]
    o[0] = n / float(MAX_HAND)
    o[1] = min(hand_count, MAX_HAND) / float(MAX_HAND)
    o[2] = min(n / hand_count, 2.0) / 2.0 if hand_count > 0 else 0.0
    o[3] = 1.0 if n > 0 else 0.0
    return o, ids


def _pokemon_at(cur: dict, p: int, area: int, idx: int):
    if p is None or not (0 <= p <= 1) or idx is None or idx < 0:
        return None
    pl = cur["players"][p]
    arr = pl.get("active") if area == 4 else (pl.get("bench") if area == 5 else None)
    if arr is None or idx >= len(arr):
        return None
    return arr[idx]


def _card_id_at(cur: dict, sel: dict, p: int | None, area: int | None, idx: int | None) -> int:
    if area is None or idx is None or idx < 0:
        return 0
    a = int(area)
    if a in (4, 5):
        return _cid(_pokemon_at(cur, p, a, idx))
    if a == 1:
        arr = sel.get("deck") or []
    elif a == 7:
        arr = cur.get("stadium") or []
    elif a == 12:
        arr = cur.get("looking") or []
    elif p is not None and 0 <= p <= 1:
        pl = cur["players"][p]
        if a == 2:
            arr = pl.get("hand") or []
        elif a == 3:
            arr = pl.get("discard") or []
        elif a == 6:
            arr = pl.get("prize") or []
        else:
            return 0
    else:
        return 0
    return _cid(arr[idx]) if idx < len(arr) else 0


def _poke_token(pk, is_active: bool, is_mine: bool, conditions: list[float],
                hp_div: float = HP_DIV):
    """Pokemon dict (or None=facedown, or missing) ->
    (feats, [pokemon_id, tool_id, energy_card0, energy_card1])."""
    v = np.zeros(BOARD_F, dtype=np.float32)
    ids = [0, 0, 0, 0]
    v[0] = 1.0  # present
    v[23] = 1.0 if is_active else 0.0
    v[24] = 1.0 if is_mine else 0.0
    if pk is None:  # facedown: present but unknown
        return v, ids
    v[1] = 1.0  # known
    ids[0] = int(pk.get("id") or 0)
    hp = float(pk.get("hp") or 0)
    mx = float(pk.get("maxHp") or 0)
    v[2] = hp / hp_div
    v[3] = mx / hp_div
    v[4] = hp / mx if mx > 0 else 0.0
    v[5] = (mx - hp) / hp_div
    v[6] = 1.0 if pk.get("appearThisTurn") else 0.0
    energies = pk.get("energies") or []
    v[7] = len(energies) / 5.0
    for e in energies:
        e = int(e)
        if 0 <= e < 12:
            v[8 + e] += 1.0 / 3.0
    tools = pk.get("tools") or []
    v[20] = len(tools) / 2.0
    v[21] = 1.0 if tools else 0.0
    if tools:
        ids[1] = _cid(tools[0])
    v[22] = len(pk.get("preEvolution") or []) / 3.0
    # identities of attached energy cards (special energies carry effects)
    ecs = pk.get("energyCards") or []
    for j, ec in enumerate(ecs[:2]):
        ids[2 + j] = _cid(ec)
    if is_active:
        v[25:30] = conditions
    return v, ids


def own_remaining(obs: dict, own_deck: list[int]) -> list[int]:
    """Our hidden pool (deck + facedown prizes): the known 60-list minus every
    card of ours visible anywhere (board incl. attachments/pre-evolutions,
    discard, revealed prizes, stadium, looking, hand). Feeds the v6 deck-bag
    token so the net always knows what it can still draw into."""
    from .deck_infer import observed_multiset  # local: deck_infer imports us
    cur = obs["current"]
    me = int(cur["yourIndex"])
    rem = Counter(int(c) for c in own_deck)
    rem.subtract(observed_multiset(cur, me))
    for c in cur["players"][me].get("hand") or []:
        if c and c.get("id"):
            rem[int(c["id"])] -= 1
    return [cid for cid, k in rem.items() for _ in range(max(0, k))]


def encode_obs(obs: dict, mem: LogMemory | None = None,
               fwd: np.ndarray | None = None, v6: bool = False,
               own_remain: list[int] | None = None) -> dict[str, np.ndarray]:
    """fwd: optional (K, OPT_FWD[_V6]) engine-lookahead block
    (fwd_features.probe); written into the tail of opt_feats. None -> zeros
    (valid bit stays 0).

    v6=False reproduces the v5 encoding bit-identically. v6=True widens the
    board to 8 bench slots per side, renorms HP to /400, and fills the
    deck-bag token with own_remain (our hidden pool) outside deck searches.
    """
    cur = obs["current"]
    sel = obs["select"]
    me = int(cur["yourIndex"])
    op = 1 - me
    pme, pop = cur["players"][me], cur["players"][op]
    n_bench = N_BENCH_V6 if v6 else 5
    n_board = N_BOARD_V6 if v6 else N_BOARD
    hp_div = HP_DIV_V6 if v6 else HP_DIV
    opt_w = OPT_F_V6 if v6 else OPT_F

    # ---------------- globals
    g = np.zeros(GLOB_F, dtype=np.float32)
    first = cur.get("firstPlayer", -1)
    g[0] = min(int(cur.get("turn") or 0), 60) / 50.0
    g[1] = min(int(cur.get("turnActionCount") or 0), 30) / 20.0
    g[2] = 1.0 if first >= 0 else 0.0
    g[3] = 1.0 if first == me else 0.0
    g[4] = 1.0 if cur.get("supporterPlayed") else 0.0
    g[5] = 1.0 if cur.get("stadiumPlayed") else 0.0
    g[6] = 1.0 if cur.get("energyAttached") else 0.0
    g[7] = 1.0 if cur.get("retreated") else 0.0
    conds = ("poisoned", "burned", "asleep", "paralyzed", "confused")
    for base, pl in ((8, pme), (14, pop)):
        g[base + 0] = int(pl.get("deckCount") or 0) / 60.0
        g[base + 1] = int(pl.get("handCount") or 0) / float(MAX_HAND)
        g[base + 2] = len(pl.get("prize") or []) / 6.0
        g[base + 3] = min(len(pl.get("discard") or []), 60) / 60.0
        g[base + 4] = len(pl.get("bench") or []) / float(n_bench)
        g[base + 5] = int(pl.get("benchMax") or 5) / float(n_bench)
    my_conds = [1.0 if pme.get(c) else 0.0 for c in conds]
    op_conds = [1.0 if pop.get(c) else 0.0 for c in conds]
    g[20:25] = my_conds
    g[25:30] = op_conds
    my_act = pme.get("active") or []
    op_act = pop.get("active") or []
    g[30] = 1.0 if len(my_act) > 0 else 0.0
    g[31] = 1.0 if (my_act and my_act[0] is not None) else 0.0
    g[32] = 1.0 if len(op_act) > 0 else 0.0
    g[33] = 1.0 if (op_act and op_act[0] is not None) else 0.0
    g[34] = sum(1 for c in (pme.get("prize") or []) if c) / 6.0
    g[35] = sum(1 for c in (pop.get("prize") or []) if c) / 6.0
    g[36] = len(cur.get("looking") or []) / 8.0
    g[37] = min(len(sel.get("deck") or []), 60) / 60.0
    stadium = cur.get("stadium") or []
    g[38] = 1.0 if stadium else 0.0
    g[39] = 1.0 if (stadium and stadium[0].get("playerIndex") == me) else 0.0
    stadium_id = np.int16(_cid(stadium[0]) if stadium else 0)

    # ---------------- select context
    c = np.zeros(CTX_F, dtype=np.float32)
    st = int(sel.get("type") or 0)
    sc = int(sel.get("context") or 0)
    if st < _N_SELECT_TYPE:
        c[st] = 1.0
    if sc < _N_CONTEXT:
        c[_N_SELECT_TYPE + sc] = 1.0
    o = _N_SELECT_TYPE + _N_CONTEXT
    # log scale, no hard caps: forced picks reach k=21 and "choose 3" must
    # stay distinguishable from "must choose 21"
    c[o + 0] = np.log1p(max(0, int(sel.get("minCount") or 0))) / np.log1p(21.0) * 1.5
    c[o + 1] = np.log1p(max(0, int(sel.get("maxCount") or 0))) / np.log1p(21.0) * 1.5
    c[o + 2] = np.log1p(max(0, int(sel.get("remainDamageCounter") or 0))) / np.log1p(32.0) * 1.5
    c[o + 3] = np.log1p(max(0, int(sel.get("remainEnergyCost") or 0))) / np.log1p(8.0) * 1.5
    ctx_ids = np.array([_cid(sel.get("effect")), _cid(sel.get("contextCard"))], dtype=np.int16)

    # ---------------- board tokens
    board = np.zeros((n_board, BOARD_F), dtype=np.float32)
    board_ids = np.zeros((n_board, 4), dtype=np.int16)
    zero5 = [0.0] * 5

    def fill(slot: int, pk_arr, idx: int, is_active: bool, is_mine: bool, conditions):
        if idx >= len(pk_arr):
            return
        v, ids = _poke_token(pk_arr[idx], is_active, is_mine, conditions, hp_div)
        board[slot] = v
        board_ids[slot] = ids

    fill(0, my_act, 0, True, True, my_conds)
    my_bench = pme.get("bench") or []
    for i in range(n_bench):
        fill(1 + i, my_bench, i, False, True, zero5)
    fill(1 + n_bench, op_act, 0, True, False, op_conds)
    op_bench = pop.get("bench") or []
    for i in range(n_bench):
        fill(2 + n_bench + i, op_bench, i, False, False, zero5)

    # ---------------- card lists
    hand_ids = np.zeros(MAX_HAND, dtype=np.int16)
    for i, cd in enumerate((pme.get("hand") or [])[:MAX_HAND]):
        hand_ids[i] = _cid(cd)
    look_ids = np.zeros(MAX_LOOK, dtype=np.int16)
    look_feats = np.zeros((MAX_LOOK, 2), dtype=np.float32)
    for i, cd in enumerate((cur.get("looking") or [])[:MAX_LOOK]):
        look_ids[i] = _cid(cd)
        look_feats[i, 0] = 1.0  # present (facedown entries stay id 0)
        if cd:
            look_feats[i, 1] = 1.0 if cd.get("playerIndex") == me else 0.0
    disc_ids = np.zeros((2, MAX_DISC), dtype=np.int16)
    for row, pl in ((0, pme), (1, pop)):
        for i, cd in enumerate((pl.get("discard") or [])[:MAX_DISC]):
            disc_ids[row, i] = _cid(cd)
    deck_ids = np.zeros(MAX_DISC, dtype=np.int16)
    sel_deck = sel.get("deck") or []
    for i, cd in enumerate(sel_deck[:MAX_DISC]):
        deck_ids[i] = _cid(cd)
    if v6 and not sel_deck and own_remain:
        # outside deck searches the v6 deck-bag carries our hidden pool
        # (deck + facedown prizes): the net always knows its remaining outs
        for i, cid in enumerate(own_remain[:MAX_DISC]):
            deck_ids[i] = cid

    # ---------------- whole-game history (cross-decision memory)
    if mem is not None:
        mem_v, mem_ids = mem.arrays(int(pop.get("handCount") or 0))
    else:
        mem_v = np.zeros(MEM_F, dtype=np.float32)
        mem_ids = np.zeros(2 * N_MEM_ATK + N_MEM_REV, dtype=np.int16)

    # ---------------- log summary (events since our previous decision)
    lg = np.zeros(LOG_F, dtype=np.float32)
    logs_ids = np.zeros(6, dtype=np.int16)
    logs = obs.get("logs") or []
    revealed: list[int] = []
    heads = tails = op_drew = op_played = op_evolved = op_attached = 0
    op_moves_hidden = 0
    my_lost = op_lost = my_heal = op_heal = 0.0
    op_switched = False
    for e in logs:
        t = e.get("type")
        p = e.get("playerIndex")
        if t == 22:
            heads, tails = (heads + 1, tails) if e.get("head") else (heads, tails + 1)
        elif t == 5 and p == op:
            op_drew += 1
        elif t == 15:
            aid = int(e.get("attackId") or 0)
            if p == me:
                logs_ids[0] = aid
            elif p == op:
                logs_ids[1] = aid
        elif t == 16:
            val = float(e.get("value") or 0)
            if p == me:
                my_lost, my_heal = (my_lost - min(val, 0), my_heal + max(val, 0))
            elif p == op:
                op_lost, op_heal = (op_lost - min(val, 0), op_heal + max(val, 0))
        elif t == 10 and p == op:
            op_played += 1
        elif t == 12 and p == op:
            op_evolved += 1
        elif t == 11 and p == op:
            op_attached += 1
        elif t in (8, 9) and p == op:
            op_switched = True
        elif t == 7 and p == op:
            op_moves_hidden += 1
        elif t == 6 and p == op and e.get("toArea") == 2:
            cid = int(e.get("cardId") or 0)
            if cid > 0:
                revealed.append(cid)  # card known to be in opponent's hand
    for j, cid in enumerate(revealed[-4:]):
        logs_ids[2 + j] = cid
    lg[0] = min(len(logs), 30) / 30.0
    lg[1] = min(heads, 3) / 3.0
    lg[2] = min(tails, 3) / 3.0
    lg[3] = min(op_drew, 8) / 8.0
    lg[4] = 1.0 if logs_ids[0] else 0.0
    lg[5] = 1.0 if logs_ids[1] else 0.0
    lg[6] = min(op_played, 4) / 4.0
    lg[7] = min(op_evolved, 3) / 3.0
    lg[8] = min(op_attached, 3) / 3.0
    lg[9] = 1.0 if op_switched else 0.0
    lg[10] = min(my_lost, hp_div) / hp_div
    lg[11] = min(op_lost, hp_div) / hp_div
    lg[12] = min(my_heal, 200.0) / 200.0
    lg[13] = min(op_heal, 200.0) / 200.0
    lg[14] = min(len(revealed), 4) / 4.0
    lg[15] = min(op_moves_hidden, 8) / 8.0

    # ---------------- options
    options = sel.get("option") or []
    K = len(options)
    opt_feats = np.zeros((K, opt_w), dtype=np.float32)
    opt_card = np.zeros(K, dtype=np.int16)
    opt_tgt = np.zeros(K, dtype=np.int16)
    opt_atk = np.zeros(K, dtype=np.int16)
    my_active_pk = my_act[0] if (my_act and my_act[0] is not None) else None

    for k, opt in enumerate(options):
        v = opt_feats[k]
        t = int(opt.get("type") or 0)
        if t < _N_OPT_TYPE:
            v[t] = 1.0
        area = opt.get("area")
        in_area = opt.get("inPlayArea")
        if area is not None and 1 <= int(area) <= 12:
            v[17 + int(area)] = 1.0
        else:
            v[17] = 1.0
        if in_area is not None and 1 <= int(in_area) <= 12:
            v[30 + int(in_area)] = 1.0
        else:
            v[30] = 1.0
        pidx = opt.get("playerIndex")
        v[43] = 1.0 if pidx == me else 0.0
        v[44] = 1.0 if pidx == op else 0.0
        if opt.get("number") is not None:  # log scale: no hard truncation
            v[45] = np.log1p(max(0, int(opt["number"]))) / np.log1p(64.0) * 2.0
        if opt.get("count") is not None:
            v[46] = np.log1p(max(0, int(opt["count"]))) / np.log1p(16.0) * 1.6
        if opt.get("energyIndex") is not None:
            v[47] = min(int(opt["energyIndex"]), 8) / 6.0
        if opt.get("toolIndex") is not None:
            v[48] = min(int(opt["toolIndex"]), 4) / 2.0
        if opt.get("index") is not None:
            v[49] = min(int(opt["index"]), 2 * MAX_HAND) / float(MAX_HAND)
        sct = opt.get("specialConditionType")
        if sct is not None and 0 <= int(sct) < 5:
            v[50 + int(sct)] = 1.0

        idx = opt.get("index")
        owner = pidx if pidx is not None else me
        cid = 0
        target_pk = None
        if t == 3:  # CARD
            cid = _card_id_at(cur, sel, owner, area, idx)
            pk = _pokemon_at(cur, owner, int(area) if area else -1, idx if idx is not None else -1)
            if pk:
                mx = float(pk.get("maxHp") or 0)
                v[56] = (float(pk.get("hp") or 0) / mx) if mx > 0 else 0.0
                v[57] = len(pk.get("energies") or []) / 5.0
        elif t in (4, 5, 6):  # TOOL_CARD / ENERGY_CARD / ENERGY on an attached pokemon
            pk = _pokemon_at(cur, owner, int(area) if area else -1, idx if idx is not None else -1)
            if pk:
                if t == 4:
                    tools = pk.get("tools") or []
                    ti = opt.get("toolIndex")
                    if ti is not None and 0 <= ti < len(tools):
                        cid = _cid(tools[ti])
                else:
                    ecs = pk.get("energyCards") or []
                    ei = opt.get("energyIndex")
                    if ei is not None and 0 <= ei < len(ecs):
                        cid = _cid(ecs[ei])
                opt_tgt[k] = int(pk.get("id") or 0)
        elif t == 7:  # PLAY from hand
            hand = pme.get("hand") or []
            if idx is not None and 0 <= idx < len(hand):
                cid = _cid(hand[idx])
        elif t in (8, 9):  # ATTACH / EVOLVE
            cid = _card_id_at(cur, sel, me, area, idx)
            target_pk = _pokemon_at(cur, me, int(in_area) if in_area else -1,
                                    opt.get("inPlayIndex") if opt.get("inPlayIndex") is not None else -1)
        elif t in (10, 11):  # ABILITY / DISCARD in play
            cid = _card_id_at(cur, sel, owner, area, idx)
        elif t == 12:  # RETREAT
            cid = int(my_active_pk.get("id") or 0) if my_active_pk else 0
        elif t == 13:  # ATTACK
            aid = opt.get("attackId")
            if aid is not None:
                opt_atk[k] = int(aid)
            cid = int(my_active_pk.get("id") or 0) if my_active_pk else 0
        elif t == 15:  # SKILL order
            cid = int(opt.get("cardId") or 0)

        if target_pk is not None:
            opt_tgt[k] = int(target_pk.get("id") or 0)
            mx = float(target_pk.get("maxHp") or 0)
            v[58] = (float(target_pk.get("hp") or 0) / mx) if mx > 0 else 0.0
            v[55] = 1.0
        opt_card[k] = cid

    if fwd is not None and K > 0:
        n = min(K, fwd.shape[0])
        fw = min(fwd.shape[1], opt_w - 60)  # base cols are always 60
        opt_feats[:n, 60:60 + fw] = fwd[:n, :fw]

    return {
        "glob": g, "ctx": c, "ctx_ids": ctx_ids,
        "board": board, "board_ids": board_ids,
        "hand_ids": hand_ids, "look_ids": look_ids, "look_feats": look_feats,
        "disc_ids": disc_ids, "deck_ids": deck_ids,
        "logs": lg, "logs_ids": logs_ids, "mem": mem_v, "mem_ids": mem_ids,
        "stadium_id": stadium_id,
        "opt_feats": opt_feats, "opt_card": opt_card, "opt_tgt": opt_tgt, "opt_atk": opt_atk,
    }
