"""Root-parallel determinized PUCT over the engine search API.

Per decision: sample D determinizations of hidden information, open one
engine search tree per determinization, run PUCT simulations round-robin
(policy net priors, value head mapped to [-1, 1] according to the checkpoint's
value_mode, sign-flipped at opponent-to-move nodes), then pick the root action
with the highest total visit count across trees; if the budget dies before any
simulation, fall back to the policy prior.

Multi-select decisions (maxCount > 1) inside the tree are collapsed to a
single macro action chosen greedily by the policy. Multi-select decisions at
the root are handled by the caller (policy-only fast path).

Game memory (LogMemory) is threaded through the tree: each child clones its
parent's memory and folds in that step's logs (root player's perspective), so
deep nodes see up-to-date history instead of a frozen root snapshot.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .agent import batch_of_one
from .features import ARCH_F, encode_obs, own_remaining


class _Node:
    __slots__ = ("sid", "obs", "sign", "terminal", "expanded",
                 "acts", "P", "N", "W", "children", "mem")

    def __init__(self, sid: int, obs: dict | None, sign: int,
                 terminal: float | None):
        self.sid = sid
        self.obs = obs
        self.sign = sign            # +1 if we act here, -1 if opponent
        self.terminal = terminal    # value from OUR perspective, or None
        self.expanded = False
        self.acts: list[list[int]] = []
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.children: dict[int, _Node] = {}
        self.mem = None  # per-node LogMemory (root-player perspective)


def _result_value(obs: dict, my_index: int) -> float:
    r = (obs.get("current") or {}).get("result", -1)
    if r == my_index:
        return 1.0
    if r == 1 - my_index:
        return -1.0
    return 0.0  # draw / unknown


class MCTS:
    def __init__(self, model, my_index_hint: int | None = None,
                 c_puct: float = 1.5, max_sims: int = 256, extra_fn=None,
                 value_mode: str = "win_logit", opp_prior_mix: float = 0.5,
                 feat_v6: bool = False, deck_list: list[int] | None = None):
        self.model = model
        self.c_puct = c_puct
        self.max_sims = max_sims
        self.my_index = my_index_hint or 0
        # v6 nets need the wider encoding; deck_list feeds the deck-bag token
        # at our own nodes (opponent nodes get an empty bag: their true list
        # is hidden and the priors there are uniform-blended anyway)
        self.feat_v6 = feat_v6
        self.deck_list = deck_list
        # obs -> dict of extra encoder inputs ("arch", "mem", "mem_ids", ...)
        self.extra_fn = extra_fn
        # BC checkpoints emit win-probability logits, PPO checkpoints emit
        # signed returns; both must land in [-1, 1] for PUCT backups.
        self.value_mode = value_mode
        # policy net is trained on our deck only -> its priors at opponent
        # nodes are out-of-distribution; blend toward uniform to compensate
        self.opp_prior_mix = opp_prior_mix

    def _map_value(self, raw: float) -> float:
        if self.value_mode == "signed":
            return max(-1.0, min(1.0, raw))
        return 2.0 / (1.0 + math.exp(-raw)) - 1.0  # win logit -> [-1, 1]

    # ---------------------------------------------------------- evaluation

    @torch.no_grad()
    def _priors_value(self, obs: dict, mem=None) -> tuple[list[list[int]], np.ndarray, float]:
        """-> (actions, priors, value from acting player's perspective)."""
        sel = obs["select"]
        n = len(sel["option"])
        mn = max(0, int(sel.get("minCount") or 0))
        mx = max(mn, int(sel.get("maxCount") or 1))
        if n == 0:  # engine pass-through state: forced empty select
            return [[]], np.ones(1, dtype=np.float64), 0.0
        own = None
        if self.feat_v6 and self.deck_list:
            yi = int((obs.get("current") or {}).get("yourIndex") or 0)
            if yi == self.my_index:
                own = own_remaining(obs, self.deck_list)
        enc = encode_obs(obs, v6=self.feat_v6, own_remain=own)
        if self.extra_fn is not None:
            enc.update(self.extra_fn(obs, mem))
        if getattr(self.model, "arch_f", 0) > 0 and "arch" not in enc:
            enc["arch"] = np.zeros(int(self.model.arch_f), dtype=np.float32)
        b = batch_of_one(enc, n)
        logits, count_logits, value, _ = self.model.forward(b)
        v = self._map_value(float(value[0]))
        if mx <= 1:
            probs = torch.softmax(logits[0, :n], 0).numpy().astype(np.float64)
            if mn == 0:
                cl = count_logits[0, :2]
                cp = torch.softmax(cl, 0)
                p0, p1 = float(cp[0]), float(cp[1])
                acts = [[]] + [[i] for i in range(n)]
                pri = np.concatenate(([p0], p1 * probs))
            else:
                acts = [[i] for i in range(n)]
                pri = probs
        else:
            seq = self.model.sample_action(b, mn, min(mx, n), greedy=True)[0]
            acts = [seq]
            pri = np.ones(1, dtype=np.float64)
        pri = pri / max(pri.sum(), 1e-9)
        return acts, pri, v

    def _make_node(self, state: dict) -> _Node:
        obs = state["observation"]
        sid = int(state["searchId"])
        if not obs.get("select"):
            return _Node(sid, None, 1, _result_value(obs, self.my_index))
        sign = 1 if obs["current"]["yourIndex"] == self.my_index else -1
        return _Node(sid, obs, sign, None)

    def _expand(self, node: _Node) -> float:
        acts, pri, v = self._priors_value(node.obs, node.mem)
        if node.sign < 0 and len(pri) > 1:  # OOD mitigation at opponent nodes
            pri = (1.0 - self.opp_prior_mix) * pri + self.opp_prior_mix / len(pri)
        node.acts = acts
        node.P = pri
        node.N = np.zeros(len(acts), dtype=np.int32)
        node.W = np.zeros(len(acts), dtype=np.float64)
        node.expanded = True
        node.obs = None  # free
        return node.sign * v

    # ---------------------------------------------------------- search

    def _puct_pick(self, node: _Node) -> int:
        sqrt_n = math.sqrt(1.0 + node.N.sum())
        q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        u = self.c_puct * node.P * sqrt_n / (1.0 + node.N)
        return int(np.argmax(node.sign * q + u))

    def _simulate(self, io, root: _Node) -> None:
        node = root
        path: list[tuple[_Node, int]] = []
        while True:
            if node.terminal is not None:
                v = node.terminal
                break
            if not node.expanded:
                v = self._expand(node)
                break
            if not node.acts:  # defensive: should not happen
                node.terminal = 0.0
                v = 0.0
                break
            a = self._puct_pick(node)
            path.append((node, a))
            child = node.children.get(a)
            if child is None:
                st = io.step(node.sid, node.acts[a])
                child = self._make_node(st) if st is not None else \
                    _Node(-1, None, 1, 0.0)  # engine refusal -> neutral leaf
                if child.obs is not None and node.mem is not None:
                    child.mem = node.mem.clone()
                    cur_c = child.obs.get("current") or {}
                    if int(cur_c.get("yourIndex", -1)) == self.my_index:
                        # fold logs only at our-perspective nodes: the engine
                        # emits per-player windows (logIndex advances per
                        # side), so our windows tile the search timeline
                        # exactly once; folding opponent windows as well
                        # would double-count every event
                        child.mem.update(child.obs, as_player=self.my_index)
                node.children[a] = child
            node = child
        for n, a in path:
            n.N[a] += 1
            n.W[a] += v

    def choose(self, io, obs: dict, dets: list[dict], budget_s: float,
               skip_conf: float = 1.1, root_mem=None) -> tuple[list[int], dict] | None:
        """Run search; returns (select, stats) or None if unusable."""
        t0 = time.monotonic()
        deadline = t0 + budget_s
        self.my_index = int(obs["current"]["yourIndex"])
        acts, pri, root_v = self._priors_value(obs)
        t_root = time.monotonic() - t0
        if len(acts) <= 1:
            return (acts[0] if acts else []), {"sims": 0, "trees": 0,
                                               "root_v": root_v, "ms": 0.0}
        if float(pri.max()) >= skip_conf:
            return list(acts[int(pri.argmax())]), {
                "sims": 0, "trees": 0, "root_v": root_v, "skipped": True,
                "ms": (time.monotonic() - t0) * 1e3}
        roots: list[_Node] = []
        t1 = time.monotonic()
        for det in dets:
            # keep at least ~20% of the budget for actual simulations
            if roots and time.monotonic() > t0 + 0.8 * budget_s:
                break
            st = io.begin(obs, det)
            if st is None:
                continue
            r = self._make_node(st)
            if r.terminal is not None or not r.obs:
                continue
            n_opt = len((r.obs.get("select") or {}).get("option") or [])
            if n_opt != len((obs["select"] or {}).get("option") or []):
                continue  # search root out of sync with real decision
            r.acts = [list(a) for a in acts]
            r.P = pri.copy()
            r.N = np.zeros(len(acts), dtype=np.int32)
            r.W = np.zeros(len(acts), dtype=np.float64)
            r.expanded = True
            r.obs = None
            r.mem = root_mem.clone() if root_mem is not None else None
            roots.append(r)
        t_det = time.monotonic() - t1
        if not roots:
            io.end()
            return None
        sims = 0
        t_s0 = time.monotonic()
        try:
            while sims < self.max_sims:
                # stop early if the average sim would not fit in the budget
                now = time.monotonic()
                est = (now - t_s0) / sims if sims else 0.0
                if now + est > deadline:
                    break
                self._simulate(io, roots[sims % len(roots)])
                sims += 1
        finally:
            io.end()
        n_tot = np.sum([r.N for r in roots], axis=0)
        if sims == 0 or n_tot.sum() == 0:
            # budget exhausted before any simulation: fall back to the policy
            # prior instead of argmax over all-equal visit counts (= acts[0])
            best = int(pri.argmax())
            return list(acts[best]), {
                "sims": 0, "trees": len(roots), "root_v": root_v,
                "zero_sim": True, "root_ms": t_root * 1e3, "det_ms": t_det * 1e3,
                "ms": (time.monotonic() - t0) * 1e3}
        w_tot = np.sum([r.W for r in roots], axis=0)
        q = np.where(n_tot > 0, w_tot / np.maximum(n_tot, 1), -2.0)
        best = int(np.argmax(n_tot + 1e-3 * q))
        stats = {"sims": sims, "trees": len(roots), "root_v": root_v,
                 "q": float(q[best]), "n": int(n_tot[best]),
                 "root_ms": t_root * 1e3, "det_ms": t_det * 1e3,
                 "ms": (time.monotonic() - t0) * 1e3}
        return list(acts[best]), stats
