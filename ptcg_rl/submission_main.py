"""Kaggle submission entrypoint (copied into the bundle as main.py).

Inference is torch-free (numpy forward of the policy net; see np_model.py):
the Kaggle simulation container has no torch, which silently degraded earlier
torch-based bundles to a random fallback (episodes 9120xxxx, 2026-08-09).

Deliberately NO fallbacks of any kind: an erroring submission costs nothing
(it is marked as an error and does not burn a scored submission), whereas a
silent fallback plays out a full low-score run. Any failure must raise, and
the kaggle runner records the traceback in the episode logs.
"""

import json
import os
import sys

_MARKER = "net.npz"


def _agent_dir() -> str:
    """Kaggle may run this file via exec() where __file__ is undefined (or
    not pointing into the unpacked bundle); probe the known candidates."""
    cands = []
    fn = globals().get("__file__")
    if fn:
        cands.append(os.path.dirname(os.path.abspath(fn)))
    cands.append("/kaggle_simulations/agent")
    cands.append(os.getcwd())
    cands.extend(p for p in sys.path if isinstance(p, str) and p)
    probed = []
    for d in cands:
        p = os.path.abspath(d)
        if p in probed:
            continue
        probed.append(p)
        if (os.path.exists(os.path.join(p, "deck.csv"))
                and (os.path.exists(os.path.join(p, _MARKER))
                     or os.path.exists(os.path.join(p, "model.pt")))):
            return p
    raise RuntimeError(f"bundle dir not found; probed: {probed}")


_DIR = _agent_dir()
# Unconditional insert: the kaggle loader appends the agent dir before exec
# and pops it right after, so a membership-guarded insert would be skipped
# during exec and leave sys.path empty of _DIR at act() time (episode
# 2026-08-09: ModuleNotFoundError ptcg_rl). A duplicate entry is harmless.
sys.path.insert(0, _DIR)

# Import at module-exec time (the only moment the loader guarantees the agent
# dir on sys.path); once imported, ptcg_rl lives in sys.modules regardless of
# later sys.path mutations. Also fails the episode immediately if broken.
if os.path.exists(os.path.join(_DIR, "mcts_config.json")):
    from ptcg_rl.mcts_agent import MCTSAgent  # torch; local experiments only
else:
    from ptcg_rl.np_model import NumpyAgent

_AGENT = None


def _read_deck() -> list:
    with open(os.path.join(_DIR, "deck.csv")) as f:
        return [int(x) for x in f.read().split()[:60]]


def _build_agent():
    cards = os.path.join(_DIR, "cards.npz")
    prior = os.path.join(_DIR, "deck_prior.json")
    prior = prior if os.path.exists(prior) else None
    cfg_path = os.path.join(_DIR, "mcts_config.json")
    if os.path.exists(cfg_path):
        # torch-based search bundle (local experiments only): if it cannot
        # build, fail loudly instead of degrading to the plain policy
        with open(cfg_path) as f:
            kw = json.load(f)
        ag = MCTSAgent(os.path.join(_DIR, "model.pt"), cards,
                       os.path.join(_DIR, "deck.csv"),
                       os.path.join(_DIR, "deck_prior.json"), **kw)
        ag.warmup()  # torch/search-dll init outside decision budgets
        return ag
    # deck enables the engine-lookahead probe for nets trained on fwd
    # columns; per-decision probe failures degrade to zero columns
    ag = NumpyAgent(os.path.join(_DIR, _MARKER), cards, prior_path=prior,
                    deck=_read_deck())
    if ag.probe is not None:
        ag.probe._ensure_io()  # pay the engine load outside decision budgets
    import numpy as np
    a = np.ones((64, 64), np.float32)   # tiny matmul to init BLAS up front
    float((a @ a).sum())
    return ag


def agent(obs_dict: dict) -> list:
    global _AGENT
    if _AGENT is None:
        # first call is the deck-submission step, so the build happens
        # outside any real decision budget
        _AGENT = _build_agent()
    if obs_dict.get("select") is None:
        # deck-submission step == new episode: clear game memory explicitly
        # instead of relying on the turn-regression heuristic in observe()
        _AGENT.reset()
        return _read_deck()
    return _AGENT.act(obs_dict)
