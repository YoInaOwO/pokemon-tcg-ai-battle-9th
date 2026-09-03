"""Engine-free unit tests for the RL pipeline invariants flagged in the
code review (ordered labels, zone conservation, value mapping, clamps).

    python tests/test_units.py        # takes a few seconds, CPU only

Engine-dependent checks (search smoke, package load) live in
make_submission's self-test and the remote runbook instead.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ptcg_rl.deck_infer import DeckPrior, sample_determinization  # noqa: E402
from ptcg_rl.eval_local import _wilson  # noqa: E402
from ptcg_rl.features import (ARCH_UNK, GLOB_F, MAX_COUNT, MAX_HAND,  # noqa: E402
                              N_ARCH, OPT_F, OPT_FWD, SEQ_PAD,
                              LogMemory, encode_obs)
from ptcg_rl.mcts import MCTS  # noqa: E402
from ptcg_rl.model import D_CARD, CardRepr  # noqa: E402
from ptcg_rl.selfplay_ppo import MAX_SEQ  # noqa: E402
from ptcg_rl.train_bc import ordered_nll  # noqa: E402

PASS = []


def check(name, fn):
    fn()
    PASS.append(name)
    print(f"  ok  {name}")


# ---------------------------------------------------------------- constants

def t_constants():
    # measured across all replays: max forced multi-pick k = 21, hand <= 29
    assert MAX_COUNT - 1 >= 21, "count head cannot express k=21"
    assert SEQ_PAD >= 21 and MAX_SEQ >= 21, "sequence buffers too small"
    assert MAX_HAND >= 29, "hand slots below measured max"


# ------------------------------------------------------------- ordered NLL

def t_ordered_labels():
    lg = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    kk = torch.tensor([3])
    nll_sorted, acc_sorted = ordered_nll(lg, torch.tensor([[0, 1, 2]]), mask, kk)
    nll_perm, _ = ordered_nll(lg, torch.tensor([[2, 0, 1]]), mask, kk)
    assert abs(float(nll_sorted) - float(nll_perm)) > 1e-4, \
        "[0,1,2] and [2,0,1] must produce different losses"
    assert float(nll_sorted) < float(nll_perm), \
        "order agreeing with logits must be cheaper"
    assert float(acc_sorted) == 1.0
    # padding: [1,-1,-1] with k=1 equals [1] alone
    a, _ = ordered_nll(lg, torch.tensor([[1, -1, -1]]), mask, torch.tensor([1]))
    b, _ = ordered_nll(lg[:, :], torch.tensor([[1]]), mask, torch.tensor([1]))
    assert abs(float(a) - float(b)) < 1e-6


# ------------------------------------------------------------ value mapping

def t_value_mapping():
    m = MCTS(model=None, value_mode="win_logit")
    assert abs(m._map_value(0.0)) < 1e-9
    assert m._map_value(20.0) > 0.99 and m._map_value(-20.0) < -0.99
    s = MCTS(model=None, value_mode="signed")
    assert s._map_value(2.0) == 1.0 and s._map_value(-3.0) == -1.0
    assert abs(s._map_value(0.5) - 0.5) < 1e-9


# ------------------------------------------------------------- UNK embedding

def t_unk_card_ids():
    feats = torch.randn(5, 8)
    rep = CardRepr(feats)
    out = rep(torch.tensor([0, 4, 5, 9999]))  # 5/9999 beyond the real table
    assert out.shape == (4, D_CARD) and torch.isfinite(out).all()
    assert torch.allclose(out[2], out[3]), "all unknown ids must share UNK row"


# --------------------------------------------------------------- LogMemory

def _mem_obs(your_index: int) -> dict:
    return {"current": {"yourIndex": your_index},
            "logs": [{"type": 15, "playerIndex": 0, "attackId": 42},
                     {"type": 15, "playerIndex": 1, "attackId": 7},
                     {"type": 10, "playerIndex": 1, "cardId": 5},
                     {"type": 6, "playerIndex": 1, "toArea": 2, "cardId": 9},
                     {"type": 6, "playerIndex": 0, "toArea": 2, "cardId": 3}]}


def t_log_memory():
    m = LogMemory()
    m.update(_mem_obs(0))
    assert m.my_atk == [42] and m.op_atk == [7] and m.op_known == [9]
    assert m.my_known == [3], "public reveals into OUR hand must be tracked"
    c = m.clone()
    c.my_atk.append(1)
    c.counts[0] += 5
    assert m.my_atk == [42] and m.counts[0] == 1, "clone must be independent"
    s = m.swapped()
    assert s.my_atk == [7] and s.op_atk == [42]
    assert s.op_known == [3], "swapped view keeps their knowledge of our hand"
    assert s.counts[0] == m.counts[1] and s.counts[3] == m.counts[2]
    # as_player pins the perspective even when obs flips yourIndex
    f = LogMemory()
    f.update(_mem_obs(1), as_player=0)
    assert f.my_atk == [42] and f.op_atk == [7]


# ----------------------------------------------- determinization conservation

def _det_obs() -> dict:
    def pk(cid):
        return {"id": cid, "energyCards": [], "tools": [], "preEvolution": []}
    return {
        "current": {
            "yourIndex": 0, "looking": [], "stadium": [],
            "players": [
                {"deckCount": 47, "handCount": 5,
                 "hand": [{"id": 1}] * 5,
                 "prize": [None, None, {"id": 2}, None, None, None],
                 "active": [pk(3)], "bench": [], "discard": [{"id": 4}]},
                {"deckCount": 49, "handCount": 4, "prize": [None] * 6,
                 "active": [pk(7)], "bench": [], "discard": []},
            ]},
        "select": {"type": 0, "option": [{"type": 0}]},
    }


def _tmp_prior() -> str:
    decks = [{"cards": sorted([7] * 20 + [8] * 20 + [9] * 20), "w": 0.7,
              "arch": "other"},
             {"cards": sorted([7] * 10 + [8] * 10 + [9] * 40), "w": 0.3,
              "arch": "other"}]
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(decks, f)
    f.close()
    return f.name


def t_determinization():
    my_deck = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10
    path = _tmp_prior()
    try:
        prior = DeckPrior(path, basic_pokemon_ids={1, 7})
        rng = random.Random(0)
        for _ in range(20):
            d = sample_determinization(_det_obs(), my_deck, prior, rng,
                                       known_hand=[8])
            assert d is not None
            # zone sizes conserve exactly
            assert len(d["your_deck"]) == 47 and len(d["your_prize"]) == 6
            assert len(d["opponent_deck"]) == 49
            assert len(d["opponent_prize"]) == 6
            assert len(d["opponent_hand"]) == 4
            assert d["opponent_active"] == []  # active is faceup
            # our hidden zones = our 60 minus visible (hand/active/discard)
            got = Counter(d["your_deck"]) + Counter(d["your_prize"])
            want = Counter(my_deck)
            for cid, k in [(1, 5), (3, 1), (4, 1)]:
                want[cid] -= k
            assert got == want, "our 60-card multiset must conserve"
            assert 8 in d["opponent_hand"], "known hand card must be pinned"
            # SearchBegin fills prize slots by index: revealed prize must sit
            # at its true slot, not be moved to the front
            assert d["your_prize"][2] == 2, "revealed prize slot must align"
        # facedown opponent active must be filled with a basic
        obs2 = _det_obs()
        obs2["current"]["players"][1]["active"] = [None]
        obs2["current"]["players"][1]["deckCount"] = 48
        d = sample_determinization(obs2, my_deck, prior, rng)
        assert d is not None and len(d["opponent_active"]) == 1
        assert d["opponent_active"][0] in {1, 7}
        # select.deck view pins our exact deck contents
        obs3 = _det_obs()
        my_unseen = Counter(my_deck)
        for cid, k in [(1, 5), (3, 1), (4, 1), (2, 1)]:
            my_unseen[cid] -= k
        flat = [c for c, k in sorted(my_unseen.items()) for _ in range(k)]
        view = flat[:47]  # 47 in deck, remaining 5 are the hidden prizes
        obs3["select"]["deck"] = [{"id": c, "playerIndex": 0} for c in view]
        d = sample_determinization(obs3, my_deck, prior, rng)
        assert Counter(d["your_deck"]) == Counter(view)
        assert Counter(d["your_prize"]) == Counter(flat[47:]) + Counter([2])
        assert d["your_prize"][2] == 2
    finally:
        os.unlink(path)


# ------------------------------------------------------------- random agent

def t_random_act():
    from ptcg_rl.opponents import random_act
    rng = random.Random(0)
    # maxCount larger than the option count must not raise
    obs = {"select": {"option": [0, 1, 2], "minCount": 0, "maxCount": 9}}
    for _ in range(10):
        act = random_act(obs, rng)
        assert len(act) == 3 and sorted(act) == [0, 1, 2]
    obs = {"select": {"option": [0, 1, 2, 3, 4], "minCount": 1, "maxCount": 2}}
    assert len(random_act(obs, rng)) == 2


# ------------------------------------------------------------ encoder ranges

def t_encoder():
    obs = _det_obs()
    obs["select"]["option"] = [{"type": 0, "number": 100, "count": 30}]
    enc = encode_obs(obs)
    assert enc["glob"].shape == (GLOB_F,)
    assert enc["hand_ids"].shape == (MAX_HAND,)
    of = enc["opt_feats"][0]
    assert np.isfinite(of).all()
    assert of[45] > 2.0, "number=100 must not be truncated to the old cap"
    assert of[46] > 1.6, "count=30 must not be truncated to the old cap"
    # ctx min/max count must stay distinguishable up to the measured k=21
    # (the old encoding clamped at 8/12, making k=13 and k=21 identical)
    a = _det_obs()
    a["select"].update({"minCount": 13, "maxCount": 13})
    b = _det_obs()
    b["select"].update({"minCount": 21, "maxCount": 21})
    assert not np.allclose(encode_obs(a)["ctx"], encode_obs(b)["ctx"]), \
        "forced k=13 and k=21 must encode differently"


# ----------------------------------------------------------- fwd feature slot

def t_fwd_columns():
    obs = _det_obs()
    obs["select"]["option"] = [{"type": 0}, {"type": 13, "attackId": 5}]
    fwd = np.zeros((2, OPT_FWD), dtype=np.float32)
    fwd[1, 0] = 1.0
    fwd[1, 1] = 0.5
    enc = encode_obs(obs, fwd=fwd)
    of = enc["opt_feats"]
    assert of.shape == (2, OPT_F)
    assert of[1, OPT_F - OPT_FWD] == 1.0 and of[1, OPT_F - OPT_FWD + 1] == 0.5
    assert of[0, OPT_F - OPT_FWD:].sum() == 0.0
    # no fwd -> the tail stays zero (the "probe unavailable" training state)
    enc2 = encode_obs(obs)
    assert enc2["opt_feats"][:, OPT_F - OPT_FWD:].sum() == 0.0


# ---------------------------------------------------- soft deck match + unknown

def t_soft_match():
    path = _tmp_prior()
    try:
        prior = DeckPrior(path, basic_pokemon_ids={1, 7})
        deck = prior.deck_lists[0]
        seen_exact = Counter(deck[:15])
        # 2-card tech swap: exact subset match fails, soft match must not
        seen_swap = Counter(deck[:13]) + Counter({901: 1, 902: 1})
        assert prior.match_idx(seen_swap) == [], "exact match should fail here"
        soft = prior.soft_match(seen_swap)
        assert soft, "soft match must survive a 2-card swap"
        assert prior.soft_match(seen_exact)[0][1] == 1.0
        # arch posterior: tech swap keeps the archetype, garbage -> unknown bit
        obs = _det_obs()
        obs["current"]["players"][1]["discard"] = [{"id": c} for c in
                                                   list(seen_swap.elements())]
        v = prior.arch_feature(obs)
        assert v[:N_ARCH].sum() > 0.99 and v[ARCH_UNK] == 0.0
        obs["current"]["players"][1]["discard"] = [{"id": 900 + i} for i in range(15)]
        v = prior.arch_feature(obs)
        assert v[:N_ARCH].sum() == 0.0 and v[ARCH_UNK] == 1.0
    finally:
        os.unlink(path)


# ----------------------------------------------------------------- id dropout

def t_id_dropout():
    feats = torch.randn(5, 8)
    rep = CardRepr(feats)
    ids = torch.tensor([1, 2, 3, 4])
    rep.train()
    rep.id_dropout = 1.0  # always drop: output must equal the static path only
    static_only = rep.proj(rep.static[ids])
    assert torch.allclose(rep(ids), static_only, atol=1e-6)
    rep.eval()  # inference: dropout must be inert
    assert not torch.allclose(rep(ids), static_only)


# -------------------------------------------------------------------- wilson

def t_wilson():
    lo, hi = _wilson(0.5, 100)
    assert 0.40 < lo < 0.42 and 0.58 < hi < 0.60
    lo, hi = _wilson(1.0, 10)
    assert hi <= 1.0 and lo > 0.6
    assert _wilson(0.5, 0) == (0.0, 1.0)


def main() -> None:
    for name, fn in [("constants cover measured maxima", t_constants),
                     ("ordered multi-select labels keep order", t_ordered_labels),
                     ("value_mode mapping to [-1,1]", t_value_mapping),
                     ("UNK embedding for out-of-table ids", t_unk_card_ids),
                     ("LogMemory update/clone/swapped/as_player", t_log_memory),
                     ("determinization zone conservation", t_determinization),
                     ("random opponent legality", t_random_act),
                     ("encoder log1p scaling / shapes", t_encoder),
                     ("fwd columns land in the opt_feats tail", t_fwd_columns),
                     ("soft deck match survives tech swaps", t_soft_match),
                     ("id-dropout keeps static attributes", t_id_dropout),
                     ("wilson interval sanity", t_wilson)]:
        check(name, fn)
    print(f"\nall {len(PASS)} unit tests passed")


if __name__ == "__main__":
    main()
