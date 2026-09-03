"""Thin dict-based wrapper over the engine's search API.

Bypasses cg.api's dataclass conversion (json -> dataclass is ~10x slower than
orjson -> dict, and our feature encoder consumes dicts directly).
"""

from __future__ import annotations

import ctypes
import json

try:
    import orjson as _oj

    def _loads(b):
        return _oj.loads(b)
except ImportError:  # pragma: no cover
    def _loads(b):
        return json.loads(b)


def _arr(xs: list[int]):
    return (ctypes.c_int * len(xs))(*xs)


class SearchIO:
    """One engine search arena. begin() starts a determinized battle copy,
    step() advances one selection, end() recycles all memory."""

    def __init__(self):
        from cg.sim import lib  # ImportError if binaries unavailable
        self.lib = lib
        self.ptr = lib.AgentStart()

    def begin(self, obs: dict, det: dict, manual_coin: bool = False) -> dict | None:
        sbi = obs.get("search_begin_input")
        if not sbi:
            return None
        your_deck = det["your_deck"]
        sel = obs.get("select") or {}
        if sel.get("deck") is not None:
            your_deck = []  # deck contents revealed; engine ignores prediction
        cur = obs["current"]
        opp = cur["players"][1 - cur["yourIndex"]]
        active = opp.get("active") or []
        facedown = bool(active) and active[0] is None
        opp_active = det["opponent_active"] if facedown else []
        if facedown and not opp_active:
            return None
        bs = self.lib.SearchBegin(
            self.ptr, sbi.encode("ascii"), len(sbi),
            _arr(your_deck), _arr(det["your_prize"]),
            _arr(det["opponent_deck"]), _arr(det["opponent_prize"]),
            _arr(det["opponent_hand"]), _arr(opp_active), int(manual_coin))
        r = _loads(bs)
        if r.get("error"):
            return None
        return r.get("state")

    def step(self, search_id: int, select: list[int]) -> dict | None:
        bs = self.lib.SearchStep(self.ptr, search_id,
                                 _arr(select), len(select))
        r = _loads(bs)
        if r.get("error"):
            return None
        return r.get("state")

    def end(self) -> None:
        self.lib.SearchEnd(self.ptr)

    def release(self, search_id: int) -> None:
        self.lib.SearchRelease(self.ptr, search_id)
