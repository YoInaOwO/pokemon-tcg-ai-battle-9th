"""How fast is the policy forward, and how much would batching it on a GPU buy?

The actor benchmark (build/_bench_actor.py) shows the forward pass dominates an
actor's wall time. This measures the same network three ways on one captured
decision batch:

  * numpy, 1 thread          -- what the shipped agent runs
  * torch CPU, 1 thread      -- what a PPO actor runs
  * torch CUDA, batched      -- what an inference server would run

Batch-1 GPU latency is irrelevant here: an actor pool would send its decisions
to a shared server, so the number that matters is throughput at batch 64-512.

    python build/_bench_forward.py
"""
import os
import statistics as st
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
BUNDLE = os.path.join(ROOT, "build", "bench_bundle")

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from ptcg_rl.agent import batch_of_one  # noqa: E402
from ptcg_rl.features import encode_obs, own_remaining  # noqa: E402
from ptcg_rl.model import PolicyNet  # noqa: E402
from ptcg_rl.np_model import NumpyAgent  # noqa: E402

deck = [int(x) for x in open(os.path.join(BUNDLE, "deck.csv")).read().split()[:60]]
agent = NumpyAgent(os.path.join(BUNDLE, "net.npz"),
                   os.path.join(BUNDLE, "cards.npz"), deck=deck)
if agent.probe is not None:
    agent.probe._ensure_io()

# ---- capture a realistic set of decision encodings -----------------------
encs = []
obs, sd = battle_start(list(deck), list(deck))
agent.reset()
while (obs is not None and obs["current"]["result"] < 0
       and obs.get("select") is not None and len(encs) < 160):
    sel = obs["select"]
    if not sel["option"]:
        break
    fwd = agent.probe.probe(obs) if agent.probe is not None else None
    agent.observe(obs)
    own = own_remaining(obs, deck) if agent.net.feat_v6 and deck else None
    enc = encode_obs(obs, mem=agent.mem, fwd=fwd, v6=agent.net.feat_v6,
                     own_remain=own)
    if agent.net.arch_f > 0:
        enc["arch"] = np.zeros(agent.net.arch_f, np.float32)
    encs.append(enc)
    logits, _ = agent.net.forward(enc)
    mn, mx = int(sel.get("minCount") or 0), int(sel.get("maxCount") or 0)
    act = ([int(logits.argmax())] if mn == 1 and mx == 1
           else np.argsort(-logits, kind="stable")[:max(mn, 1)].tolist())
    obs = battle_select(act)
battle_finish()
K = [e["opt_feats"].shape[0] for e in encs]
print(f"captured {len(encs)} decisions, options per decision "
      f"mean {st.mean(K):.1f} max {max(K)}")


def timeit(fn, n, warm=5):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


# ---- 1. numpy, one decision at a time ------------------------------------
i = [0]


def np_step():
    e = encs[i[0] % len(encs)]
    i[0] += 1
    agent.net.forward(e)


np_ms = timeit(np_step, 120) * 1000
print(f"\nnumpy  1 thread, batch 1      {np_ms:7.2f} ms/decision   "
      f"{1000/np_ms:8.1f} decisions/s per process")

# ---- 2. torch CPU, one decision at a time --------------------------------
cards = np.load(os.path.join(BUNDLE, "cards.npz"))
net = PolicyNet(cards["card_feats"], cards["attack_feats"],
                d_model=384, n_layers=5, n_heads=4, ff=768,
                arch_f=agent.net.arch_f, opt_f=agent.net.opt_f,
                n_board=18 if agent.net.feat_v6 else 12,
                count_classes=agent.net.count_classes).eval()
batches = [batch_of_one(e, e["opt_feats"].shape[0]) for e in encs]
j = [0]


def cpu_step():
    b = batches[j[0] % len(batches)]
    j[0] += 1
    with torch.no_grad():
        net(b)


cpu_ms = timeit(cpu_step, 60) * 1000
print(f"torch  1 thread, batch 1      {cpu_ms:7.2f} ms/decision   "
      f"{1000/cpu_ms:8.1f} decisions/s per process")

# ---- 3. torch CUDA, batched ----------------------------------------------
if torch.cuda.is_available():
    dev = torch.device("cuda")
    gnet = PolicyNet(cards["card_feats"], cards["attack_feats"],
                     d_model=384, n_layers=5, n_heads=4, ff=768,
                     arch_f=agent.net.arch_f, opt_f=agent.net.opt_f,
                     n_board=18 if agent.net.feat_v6 else 12,
                     count_classes=agent.net.count_classes).to(dev).eval()
    print()
    for B in (1, 32, 128, 512):
        reps = [encs[k % len(encs)] for k in range(B)]
        kmax = max(e["opt_feats"].shape[0] for e in reps)
        bat = {}
        for key in batches[0]:
            if key in ("opt_feats", "opt_card", "opt_tgt", "opt_atk", "opt_mask"):
                continue
            bat[key] = torch.cat([batch_of_one(e, e["opt_feats"].shape[0])[key]
                                  for e in reps]).to(dev)
        of = torch.zeros(B, kmax, reps[0]["opt_feats"].shape[1])
        oc = torch.zeros(B, kmax, dtype=torch.long)
        ot = torch.zeros(B, kmax, dtype=torch.long)
        oa = torch.zeros(B, kmax, dtype=torch.long)
        om = torch.zeros(B, kmax, dtype=torch.bool)
        for r, e in enumerate(reps):
            k = e["opt_feats"].shape[0]
            of[r, :k] = torch.from_numpy(e["opt_feats"])
            oc[r, :k] = torch.from_numpy(e["opt_card"].astype(np.int64))
            ot[r, :k] = torch.from_numpy(e["opt_tgt"].astype(np.int64))
            oa[r, :k] = torch.from_numpy(e["opt_atk"].astype(np.int64))
            om[r, :k] = True
        bat.update(opt_feats=of.to(dev), opt_card=oc.to(dev), opt_tgt=ot.to(dev),
                   opt_atk=oa.to(dev), opt_mask=om.to(dev))

        def gpu_step(b=bat):
            with torch.no_grad():
                gnet(b)
            torch.cuda.synchronize()

        g_ms = timeit(gpu_step, 30) * 1000
        print(f"torch  CUDA, batch {B:4d}      {g_ms:7.2f} ms/batch      "
              f"{B*1000/g_ms:8.1f} decisions/s")
else:
    print("\n(no CUDA on this machine)")
