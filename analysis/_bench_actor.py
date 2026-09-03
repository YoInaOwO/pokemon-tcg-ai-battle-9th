"""Where does an actor's wall time actually go?

Plays self-play games with the shipped agent and times the three costs a PPO
actor pays per decision: the engine's own step, the engine-lookahead probe
(one sandboxed execution per legal option, plus the macro expansion), and the
policy forward pass. Answers whether moving the network to GPU/JAX would move
the needle, or whether the actor is engine-bound.

    python build/_bench_actor.py --games 12
"""
import argparse
import os
import statistics as st
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))

import numpy as np  # noqa: E402

BUNDLE = os.path.join(ROOT, "build", "bench_bundle")

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=12)
ap.add_argument("--net", default=os.path.join(BUNDLE, "net.npz"))
ap.add_argument("--cards", default=os.path.join(BUNDLE, "cards.npz"))
ap.add_argument("--deck", default=os.path.join(BUNDLE, "deck.csv"))
args = ap.parse_args()

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from ptcg_rl.np_model import NumpyAgent  # noqa: E402

deck = [int(x) for x in open(args.deck).read().split()[:60]]
agent = NumpyAgent(args.net, args.cards, deck=deck)
if agent.probe is not None:
    agent.probe._ensure_io()
np.ones((64, 64), np.float32) @ np.ones((64, 64), np.float32)   # warm BLAS

t_probe, t_fwd, t_step, n_opts = [], [], [], []
decisions = 0
t0 = time.perf_counter()
for g in range(args.games):
    obs, sd = battle_start(list(deck), list(deck))
    if obs is None:
        raise RuntimeError(f"battle_start failed: {sd.errorType}")
    agent.reset()
    guard = 0
    while (obs is not None and obs["current"]["result"] < 0
           and obs.get("select") is not None and guard < 4000):
        guard += 1
        sel = obs["select"]
        k = len(sel["option"])
        n_opts.append(k)

        a = time.perf_counter()
        fwd = agent.probe.probe(obs) if agent.probe is not None else None
        b = time.perf_counter()
        agent.observe(obs)
        from ptcg_rl.features import encode_obs, own_remaining
        own = (own_remaining(obs, deck) if agent.net.feat_v6 and deck else None)
        enc = encode_obs(obs, mem=agent.mem, fwd=fwd, v6=agent.net.feat_v6,
                         own_remain=own)
        if agent.net.arch_f > 0:
            enc["arch"] = np.zeros(agent.net.arch_f, np.float32)
        logits, count_logits = agent.net.forward(enc)
        c = time.perf_counter()

        mn, mx = int(sel.get("minCount") or 0), int(sel.get("maxCount") or 0)
        if k == 0:
            break
        if mn == 1 and mx == 1:
            act = [int(logits.argmax())]
        else:
            kk = mn if mn == mx else min(mx, agent.net.count_classes - 1, k)
            kk = min(max(mn, min(kk, mx)), k)
            act = np.argsort(-logits, kind="stable")[:kk].tolist() if kk else []
        obs = battle_select(act)
        d = time.perf_counter()

        t_probe.append(b - a)
        t_fwd.append(c - b)
        t_step.append(d - c)
        decisions += 1
    battle_finish()
wall = time.perf_counter() - t0


def ms(xs):
    return f"{st.mean(xs)*1000:7.2f} ms   median {st.median(xs)*1000:6.2f}"


tot = sum(t_probe) + sum(t_fwd) + sum(t_step)
print(f"\n{args.games} self-play games, {decisions} decisions, "
      f"{wall:.1f} s wall ({decisions/wall:.1f} decisions/s, single process)")
print(f"legal options per decision: mean {st.mean(n_opts):.1f}  "
      f"median {st.median(n_opts):.0f}  max {max(n_opts)}\n")
for name, xs in (("engine lookahead probe", t_probe),
                 ("policy forward (numpy)", t_fwd),
                 ("engine battle_select ", t_step)):
    print(f"  {name}  {ms(xs)}   {sum(xs)/tot*100:5.1f}% of measured time")
print(f"\n  measured total {tot:.1f} s of {wall:.1f} s wall "
      f"({tot/wall*100:.0f}% accounted for)")
