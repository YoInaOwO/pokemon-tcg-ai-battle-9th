"""Package the agent into a Kaggle submission tar.gz and self-test it.

    python -m ptcg_rl.make_submission --ckpt runs/bc_v1/model.pt          # policy only
    python -m ptcg_rl.make_submission --ckpt runs/ppo_v1/latest.pt --mcts # with search
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# torch-free bundle: the Kaggle simulation container has no torch, so plain
# bundles ship net.npz + numpy inference (np_model.py) instead of model.pt
BUNDLE_PKG_FILES = ["__init__.py", "features.py", "cards.py", "np_model.py"]
# engine-lookahead probe (nets with opt_f > 60): numpy + the cg engine
FWD_PKG_FILES = ["fwd_features.py", "search_io.py", "deck_infer.py"]
# the MCTS path is torch-based and only meaningful for local experiments
MCTS_PKG_FILES = ["model.py", "agent.py", "deck_infer.py", "search_io.py",
                  "mcts.py", "mcts_agent.py"]

# fixed_budget only applies locally (no remainingOverageTime); on Kaggle the
# budget controller derives per-move time from the overage pool.
MCTS_CONFIG = {"n_det": 4, "fixed_budget": 0.5, "max_budget": 2.5,
               "min_budget": 0.25, "reserve_s": 60.0, "safety": 0.5,
               "c_puct": 1.5, "max_sims": 400, "skip_conf": 0.97, "threads": 1}


def build(ckpt: str, deck: str, cards: str, out_dir: str, mcts: bool,
          prior: str) -> str:
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    arch_f = int(cfg.get("arch_f", 0))
    # authoritative opt_f from the trained weights (config may omit it and a
    # wrong fallback ships a bundle with silently zeroed fwd columns)
    from ptcg_rl.model import D_ATK, D_CARD
    opt_f_w = int(ck["model"]["opt_in.weight"].shape[1]) - 2 * D_CARD - D_ATK
    opt_f = int(cfg.get("opt_f") or opt_f_w)
    assert opt_f == opt_f_w, (f"config opt_f={opt_f} disagrees with weights "
                              f"({opt_f_w}); refusing to package")
    need_fwd = opt_f > 60  # net consumes engine-lookahead columns
    need_prior = mcts or arch_f > 0

    stage = os.path.join(out_dir, "submission")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "ptcg_rl"), exist_ok=True)

    src_pkg = os.path.dirname(os.path.abspath(__file__))
    extra = list(MCTS_PKG_FILES) if mcts else (["deck_infer.py"] if need_prior else [])
    if need_fwd:  # BCAgent inside the MCTS bundle also builds its probe
        extra += [f for f in FWD_PKG_FILES if f not in extra]
    for f in BUNDLE_PKG_FILES + extra:
        shutil.copy(os.path.join(src_pkg, f), os.path.join(stage, "ptcg_rl", f))
    shutil.copy(os.path.join(src_pkg, "submission_main.py"), os.path.join(stage, "main.py"))
    # ship the card-table generation the checkpoint was trained on
    from ptcg_rl.cards import load_matching
    _, cards_used = load_matching(cards,
                                  int(ck["model"]["card.proj.weight"].shape[1]),
                                  int(ck["model"]["attack.proj.weight"].shape[1]))
    from ptcg_rl.np_model import export_npz
    export_npz(ckpt, os.path.join(stage, "net.npz"), cards_used)
    if mcts:  # torch path (local only) still reads the raw checkpoint
        shutil.copy(ckpt, os.path.join(stage, "model.pt"))
    shutil.copy(cards_used, os.path.join(stage, "cards.npz"))
    shutil.copy(deck, os.path.join(stage, "deck.csv"))
    if need_prior:
        shutil.copy(prior, os.path.join(stage, "deck_prior.json"))
    if mcts or need_fwd:
        shutil.copytree(os.path.join(ROOT, "sample_submission", "sample_submission", "cg"),
                        os.path.join(stage, "cg"),
                        ignore=shutil.ignore_patterns("__pycache__"))
    if mcts:
        with open(os.path.join(stage, "mcts_config.json"), "w") as f:
            json.dump(MCTS_CONFIG, f, indent=1)
    print(f"bundle: mcts={mcts} arch_f={arch_f} opt_f={opt_f} "
          f"fwd={'yes' if need_fwd else 'no'} prior={'yes' if need_prior else 'no'} "
          f"cards={os.path.basename(cards_used)}")

    tar_path = os.path.join(out_dir, "submission.tar.gz")
    if os.path.exists(tar_path):
        os.remove(tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in sorted(os.listdir(stage)):
            tar.add(os.path.join(stage, name), arcname=name)
    mb = os.path.getsize(tar_path) / (1024 * 1024)
    assert mb < 190, f"submission too large: {mb:.1f} MiB"
    print(f"written {tar_path} ({mb:.1f} MiB)")
    return stage


SELF_TEST = r"""
import os, sys, random, time, types
stage = sys.argv[1]
root = sys.argv[2]
sys.path.insert(0, os.path.join(root, "sample_submission", "sample_submission"))
if not os.path.exists(os.path.join(stage, "mcts_config.json")):
    # emulate the Kaggle simulation container: no torch. A plain bundle that
    # touches torch anywhere must fail here, not degrade silently online.
    import importlib.abc
    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is blocked: Kaggle sim env has no torch")
            return None
    sys.meta_path.insert(0, _NoTorch())

# Load main.py exactly like kaggle_environments.agent.get_last_callable:
# compile + exec into a bare dict with NO __file__, agent dir appended to
# sys.path only for the duration of the exec and popped afterwards. This
# reproduces the loader semantics that broke the 2026-08-09 submission
# (membership-guarded sys.path insert skipped, then popped -> ModuleNotFound).
main_path = os.path.join(stage, "main.py")
with open(main_path) as f:
    code = compile(f.read(), main_path, "exec")
env = {}
sys.path.append(stage)
exec(code, env)
sys.path.pop()
sub = types.SimpleNamespace(**env)  # snapshot: only use for the callables
from cg.game import battle_start, battle_select, battle_finish

deck = sub._read_deck()
assert len(deck) == 60, "deck.csv broken"
obs, sd = battle_start(list(deck), list(deck))
assert obs is not None, f"deck rejected by engine: {sd.errorType}"
random.seed(0)
steps = 0
lat = []
while obs["current"]["result"] < 0 and steps < 1500:
    yi = obs["current"]["yourIndex"]
    if yi == 0:
        t = time.perf_counter()
        act = sub.agent(obs)
        lat.append(time.perf_counter() - t)
    else:
        sel = obs["select"]
        n_o = len(sel["option"])
        act = random.sample(range(n_o), min(sel["maxCount"], n_o))
    obs = battle_select(act)
    steps += 1
battle_finish()
res = obs["current"]["result"]
import statistics
print(f"SELF-TEST OK: result={res} (we are player 0) steps={steps} "
      f"agent latency p50={1000*statistics.median(lat):.0f}ms max={1000*max(lat):.0f}ms")
# no-fallback contract: main.py has no try/except at all, so reaching this
# point already proves the real agent built and played every decision
ag = env.get("_AGENT")  # live module globals, not the SimpleNamespace snapshot
assert ag is not None, "agent was never built!"
if not os.path.exists(os.path.join(stage, "mcts_config.json")):
    assert type(ag).__name__ == "NumpyAgent", f"expected NumpyAgent, got {type(ag)}"
if os.path.exists(os.path.join(stage, "mcts_config.json")):
    assert hasattr(ag, "stats"), "mcts bundle built a non-MCTS agent!"
if hasattr(ag, "stats"):
    print(f"mcts agent stats: {ag.stats}")
    assert ag.stats["search"] > 0, "search never engaged during self-test!"
    assert not ag.io_broken, "search backend failed to initialize!"
    if ag.stats.get("zero_sim", 0) > ag.stats["search"] // 2:
        print(f"WARNING: {ag.stats['zero_sim']} zero-sim decisions "
              f"(budget too tight for this hardware)")
if getattr(ag, "probe", None) is not None:
    print(f"fwd probe stats: {ag.probe.stats}")
    assert not ag.probe.disabled, "fwd probe engine failed to load!"
    assert ag.probe.stats["probes"] > 0, "fwd probe never engaged in self-test!"
    assert ag.probe.stats["errors"] == 0, f"fwd probe errors: {ag.probe.stats}"
"""


def parity_check(ckpt: str, stage: str, n_decisions: int = 60,
                 tol: float = 2e-3) -> None:
    """Play a short engine game and compare torch vs shipped-numpy logits at
    every decision. Catches any drift between model.py and np_model.py (the
    2026-08 discard-count clamp bug produced ~2e-5 here; real breaks are >1)."""
    import random

    import numpy as np

    sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))
    from cg.game import battle_finish, battle_select, battle_start

    from ptcg_rl.agent import BCAgent
    from ptcg_rl.np_model import NumpyAgent

    deck_ids = []
    with open(os.path.join(stage, "deck.csv")) as f:
        for tok in f.read().replace(",", " ").split():
            deck_ids.append(int(tok))
    prior = os.path.join(stage, "deck_prior.json")
    prior = prior if os.path.exists(prior) else None
    tg = BCAgent(ckpt, os.path.join(stage, "cards.npz"),
                 prior_path=prior, deck=deck_ids)
    npa = NumpyAgent(os.path.join(stage, "net.npz"),
                     os.path.join(stage, "cards.npz"),
                     prior_path=prior, deck=deck_ids)

    import torch

    from ptcg_rl.agent import batch_of_one
    random.seed(1)
    worst = 0.0
    checked = 0
    argmax_bad = 0
    # engine shuffles are nondeterministic: play as many games as needed
    for _game in range(6):
        if checked >= n_decisions:
            break
        obs, sd = battle_start(list(deck_ids), list(deck_ids))
        assert obs is not None, f"deck rejected in parity check: {sd.errorType}"
        tg.reset()
        npa.reset()
        steps = 0
        while obs["current"]["result"] < 0 and checked < n_decisions and steps < 1000:
            if obs["current"]["yourIndex"] == 0:
                n_opt = len(obs["select"]["option"])
                enc_np = npa.encode(obs)
                np_logits, np_cnt = npa.net.forward(enc_np)
                enc_t = tg.encode(obs)
                with torch.no_grad():
                    t_logits, t_cnt, _, _ = tg.model(batch_of_one(enc_t, n_opt))
                tl = t_logits[0].numpy()
                d = max(np.abs(tl - np_logits).max(),
                        np.abs(t_cnt[0].numpy() - np_cnt).max())
                worst = max(worst, float(d))
                # the functional check: both nets must pick the same option
                # (near-ties within numeric noise are excused)
                if int(tl.argmax()) != int(np_logits.argmax()):
                    s = np.sort(tl)
                    if len(s) < 2 or (s[-1] - s[-2]) > 5e-3:
                        argmax_bad += 1
                checked += 1
                act = npa.act(obs)
            else:
                sel = obs["select"]
                k = min(sel["maxCount"], len(sel["option"]))
                act = random.sample(range(len(sel["option"])), k)
            obs = battle_select(act)
            steps += 1
        battle_finish()
    assert checked > 10, f"parity check exercised only {checked} decisions"
    assert worst < tol, (f"torch vs numpy logits diverge: max|diff|={worst:.2e} "
                         f"over {checked} decisions (tol {tol})")
    assert argmax_bad == 0, (f"torch and numpy disagree on the argmax at "
                             f"{argmax_bad}/{checked} decisions")
    print(f"parity OK: {checked} decisions, max|torch-numpy|={worst:.2e}")


def self_test(stage: str) -> None:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SELF_TEST)
        path = f.name
    try:
        subprocess.run([sys.executable, path, stage, ROOT], check=True)
    finally:
        os.unlink(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deck", default=os.path.join(ROOT, "data", "decks", "mega_lopunny_220eddd2.csv"))
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "build"))
    ap.add_argument("--mcts", action="store_true", help="bundle determinized search")
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()
    stage = build(args.ckpt, args.deck, args.cards, args.out_dir, args.mcts, args.prior)
    if not args.no_test:
        if not args.mcts:  # mcts bundles ship the torch ckpt itself
            parity_check(args.ckpt, stage)
        self_test(stage)


if __name__ == "__main__":
    main()
