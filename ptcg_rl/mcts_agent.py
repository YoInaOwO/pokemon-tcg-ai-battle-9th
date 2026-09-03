"""Inference agent: determinized PUCT on top of the trained policy/value net.

Decision flow per observation:
  1. trivial fast paths (forced moves) -> no model call at all
  2. multi-select decisions -> policy greedy (search over sets not worth it)
  3. confident policy (top prior > skip_conf) -> play it directly
  4. otherwise PUCT across n_det determinizations within a time budget
Any failure at any stage falls back to the plain policy, then to random.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .agent import BCAgent
from .deck_infer import DeckPrior, basics_from_cards, sample_determinization
from .features import MEM_F, N_MEM_ATK, N_MEM_REV
from .mcts import MCTS


def _read_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()]


class MCTSAgent:
    def __init__(self, ckpt_path: str, cards_path: str, deck_path: str,
                 prior_path: str, threads: int = 1, n_det: int = 4,
                 fixed_budget: float = 1.0, max_budget: float = 2.5,
                 min_budget: float = 0.25, reserve_s: float = 60.0,
                 safety: float = 0.5, c_puct: float = 1.5,
                 max_sims: int = 400, skip_conf: float = 0.97,
                 seed: int = 0):
        self.deck_list = _read_deck_csv(deck_path)
        self.policy = BCAgent(ckpt_path, cards_path, device="cpu",
                              temperature=0.0, threads=threads,
                              prior_path=prior_path, deck=self.deck_list)
        self.prior = DeckPrior(prior_path,
                               basics_from_cards(self.policy.cards["card_feats"]))
        self._root_yi = 0
        self._root_mem: tuple[np.ndarray, np.ndarray] | None = None
        self.mcts = MCTS(self.policy.model, c_puct=c_puct, max_sims=max_sims,
                         extra_fn=self._extra, value_mode=self.policy.value_mode,
                         feat_v6=getattr(self.policy, "feat_v6", False),
                         deck_list=self.deck_list)
        self.skip_conf = skip_conf
        self.n_det = n_det
        self.fixed_budget = fixed_budget
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.reserve_s = reserve_s
        self.safety = safety
        self.rng = random.Random(seed)
        self.decisions = 0
        self.last_turn = -1
        self.stats = {"search": 0, "policy": 0, "fast": 0, "fallback": 0,
                      "zero_sim": 0}
        self.io = None
        self.io_broken = False

    # ------------------------------------------------------------- helpers

    def warmup(self) -> None:
        """Pay one-time costs (torch lazy init, search dll load, first
        transformer forward ~seconds) outside any real decision budget --
        ideally during the deck-submission step of the match."""
        try:
            self._get_io()
        except Exception:  # noqa: BLE001
            pass
        try:
            from .agent import batch_of_one
            from .features import ARCH_F, encode_obs
            fake = {"current": {"yourIndex": 0, "turn": 0, "result": -1,
                                "looking": [], "stadium": [], "logs": [],
                                "players": [{"deckCount": 53, "handCount": 7,
                                             "prize": [None] * 6, "active": [],
                                             "bench": [], "discard": []}
                                            for _ in range(2)]},
                    "select": {"type": 0, "context": 0, "minCount": 1,
                               "maxCount": 1,
                               "option": [{"type": 0}, {"type": 0}]}}
            enc = encode_obs(fake, v6=getattr(self.policy, "feat_v6", False))
            if getattr(self.policy.model, "arch_f", 0) > 0:
                enc["arch"] = np.zeros(int(self.policy.model.arch_f),
                                       dtype=np.float32)
            import torch
            with torch.no_grad():
                self.policy.model.forward(batch_of_one(enc, 2))
        except Exception:  # noqa: BLE001
            pass

    def _extra(self, obs: dict, mem=None) -> dict:
        """Extra encoder inputs for in-search evaluations.

        mem is the per-node LogMemory maintained by the search (root player's
        perspective, incrementally updated along the path). Our nodes use it
        directly; opponent-perspective nodes get the public half swapped to
        their side. Without a node memory (tree root / legacy), fall back to
        the frozen root snapshot for our nodes and zeros for theirs."""
        out: dict = {}
        yi = int((obs.get("current") or {}).get("yourIndex") or 0)
        ours = yi == self._root_yi
        if getattr(self.policy.model, "arch_f", 0) > 0:
            extra = self.policy.mem.op_known if ours else None
            af = self.prior.arch_feature(obs, extra_ids=extra)
            out["arch"] = af[:int(self.policy.model.arch_f)]
        opp_hc = int(((obs.get("current") or {}).get("players")
                      or [{}, {}])[1 - yi].get("handCount") or 0)
        if mem is not None:
            m = mem if ours else mem.swapped()
            out["mem"], out["mem_ids"] = m.arrays(opp_hc)
        elif ours and self._root_mem is not None:
            out["mem"], out["mem_ids"] = self._root_mem
        else:
            out["mem"] = np.zeros(MEM_F, dtype=np.float32)
            out["mem_ids"] = np.zeros(2 * N_MEM_ATK + N_MEM_REV, dtype=np.int16)
        return out

    def _get_io(self):
        if self.io is None and not self.io_broken:
            try:
                from .search_io import SearchIO
                self.io = SearchIO()
            except Exception:
                self.io_broken = True
        return self.io

    def _budget(self, obs: dict) -> float:
        turn = int((obs.get("current") or {}).get("turn") or 0)
        if turn < self.last_turn:      # new game (local eval reuses the agent)
            self.decisions = 0
        self.last_turn = turn
        self.decisions += 1
        over = obs.get("remainingOverageTime")
        if over is None:
            return self.fixed_budget
        usable = max(0.0, float(over) - self.reserve_s)
        est_remaining = max(25, 110 - self.decisions)
        b = usable * self.safety / est_remaining
        if b < self.min_budget:
            return 0.0
        return min(b, self.max_budget)

    # ------------------------------------------------------------- act

    def act(self, obs_dict: dict) -> list[int]:
        try:
            return self._act(obs_dict)
        except Exception:
            self.stats["fallback"] += 1
            try:
                return self.policy.act(obs_dict)
            except Exception:
                sel = obs_dict.get("select") or {}
                n = len(sel.get("option") or [])
                mn = max(0, int(sel.get("minCount") or 0))
                return list(range(min(n, max(mn, 1))))

    def _act(self, obs: dict) -> list[int]:
        sel = obs.get("select")
        if not sel:
            return []
        # advance game memory exactly once per decision (idempotent guard
        # makes the later policy.act fast paths / fallbacks safe)
        self.policy.observe(obs)
        cur = obs.get("current") or {}
        self._root_yi = int(cur.get("yourIndex") or 0)
        opp_hand = int((cur.get("players") or [{}, {}])[1 - self._root_yi].get("handCount") or 0)
        self._root_mem = self.policy.mem.arrays(opp_hand)
        n = len(sel.get("option") or [])
        mn = max(0, int(sel.get("minCount") or 0))
        mx = max(mn, int(sel.get("maxCount") or 1))
        if n == 0:
            return []
        if n == 1 and mn >= 1:
            self.stats["fast"] += 1
            return [0]
        if mn >= n:
            self.stats["fast"] += 1
            return list(range(n))
        if mx > 1:
            self.stats["policy"] += 1
            return self.policy.act(obs)

        budget = self._budget(obs)
        io = self._get_io()
        if budget <= 0.0 or io is None:
            self.stats["policy"] += 1
            return self.policy.act(obs)

        dets = []
        for _ in range(self.n_det):
            d = sample_determinization(obs, self.deck_list, self.prior, self.rng,
                                       known_hand=self.policy.mem.op_known)
            if d is not None:
                dets.append(d)
        if not dets:
            self.stats["policy"] += 1
            return self.policy.act(obs)

        r = self.mcts.choose(io, obs, dets, budget, skip_conf=self.skip_conf,
                             root_mem=self.policy.mem)
        if r is None:
            self.stats["policy"] += 1
            return self.policy.act(obs)
        select, st = r
        if st.get("zero_sim"):
            self.stats["zero_sim"] += 1
        if st.get("skipped"):
            self.stats["policy"] += 1
        else:
            self.stats["search"] += 1
        return select
