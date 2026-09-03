"""Local evaluation: BC agent vs random / another checkpoint, via the C++ engine.

    python -m ptcg_rl.eval_local --ckpt runs/bc_v1/model.pt --games 40 --workers 10
    python -m ptcg_rl.eval_local --ckpt A.pt --opponent ckpt:B.pt --games 100

The protagonist may also be a packaged submission (tar.gz with main.py +
deck.csv); its own deck is used automatically:

    python -m ptcg_rl.eval_local --ckpt build/submission_xxx.tar.gz --opponent random
"""

from __future__ import annotations

import os

# Must precede any numpy/torch import: BLAS / OpenMP pools size themselves
# from these at library load and otherwise spawn ncores threads in EVERY
# worker, thrashing the box (20 procs x 25 threads). Hard assignment, not
# setdefault: container images (e.g. AutoDL) often preset OMP_NUM_THREADS,
# which silently defeated the clamp.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib.util
import math
import multiprocessing
import random
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))
sys.path.insert(0, ROOT)

DEFAULT_DECK = os.path.join(ROOT, "data", "decks", "mega_lopunny_220eddd2.csv")
SAMPLE_DECK = os.path.join(ROOT, "sample_submission", "sample_submission", "deck.csv")
DECISION_CAP = 2000

_CFG: dict = {}
_AGENT = None
_OPP = None


def _read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split()[:60]]


def _random_act(obs: dict) -> list[int]:
    sel = obs["select"]
    n = len(sel["option"])
    return random.sample(range(n), min(int(sel.get("maxCount") or 1), n))


def prepare_submission(path: str) -> str:
    """tar.gz (or already-extracted dir) -> dir containing main.py + deck.csv."""
    if os.path.isdir(path):
        sub_dir = path
    else:
        sub_dir = tempfile.mkdtemp(prefix="sub_eval_")
        with tarfile.open(path) as tf:
            tf.extractall(sub_dir)
        # tolerate archives that wrap contents in one top-level folder
        if not os.path.exists(os.path.join(sub_dir, "main.py")):
            entries = os.listdir(sub_dir)
            if len(entries) == 1:
                sub_dir = os.path.join(sub_dir, entries[0])
    for req in ("main.py", "deck.csv"):
        if not os.path.exists(os.path.join(sub_dir, req)):
            raise FileNotFoundError(f"submission lacks {req}: {path}")
    return os.path.abspath(sub_dir)


class _SubmissionAgent:
    """Black-box wrapper around a submission's main.agent(obs) -> [indices].

    chdir()s into the submission dir permanently (per worker process):
    submission code loads deck.csv / model files via relative paths, so
    every other path in the eval config must already be absolute.
    """

    def __init__(self, sub_dir: str):
        os.chdir(sub_dir)
        sys.path.insert(0, sub_dir)
        spec = importlib.util.spec_from_file_location(
            "submission_main_under_eval", os.path.join(sub_dir, "main.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._agent = mod.agent

    def act(self, obs: dict) -> list[int]:
        return self._agent(obs)


def _init_worker(cfg: dict) -> None:
    global _CFG, _AGENT, _OPP
    _CFG = cfg
    import torch
    torch.set_num_threads(1)  # 20 procs x default threads oversubscribe the box
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized (e.g. forked after parent touched torch)
    try:  # runtime clamp for any pool that was created before the env vars hit
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    from ptcg_rl.agent import BCAgent
    prior = cfg.get("prior")
    m = cfg.get("mcts")
    dev = cfg.get("device", "cpu")
    if cfg.get("sub_dir"):
        _AGENT = _SubmissionAgent(cfg["sub_dir"])
    elif m:
        # MCTS stays on CPU: eval must rehearse the CPU-only Kaggle budget
        from ptcg_rl.mcts_agent import MCTSAgent
        _AGENT = MCTSAgent(cfg["ckpt"], cfg["cards"], cfg["deck"], m["prior"],
                           n_det=m["det"], fixed_budget=m["budget"],
                           seed=os.getpid())
    else:
        _AGENT = BCAgent(cfg["ckpt"], cfg["cards"], device=dev,
                         temperature=cfg["temperature"], threads=1,
                         prior_path=prior, deck=_read_deck(cfg["deck"]))
    if cfg["opponent"].startswith("ckpt:"):
        _OPP = BCAgent(cfg["opponent"][5:], cfg["cards"], device=dev,
                       temperature=cfg["temperature"], threads=1,
                       prior_path=prior, deck=_read_deck(cfg["opp_deck"]))
    elif cfg["opponent"] == "self":
        # must be an independent instance: BCAgent carries per-game LogMemory
        # and sharing one object would interleave both sides' histories
        _OPP = BCAgent(cfg["ckpt"], cfg["cards"], device=dev,
                       temperature=cfg["temperature"], threads=1,
                       prior_path=prior, deck=_read_deck(cfg["opp_deck"]))


def _play(seed: int) -> tuple[int, int, int, int, float, int]:
    """One game -> (result, seat, went_first, latency n, latency sum, decisions).

    result: 1 win, 0 loss, 2 draw (engine result), 3 abort (decision cap /
    engine stall) -- aborts are a robustness failure, never a half-win.
    """
    from cg.game import battle_start, battle_select, battle_finish

    random.seed(seed)
    my_deck = _read_deck(_CFG["deck"])
    opp_deck = _read_deck(_CFG["opp_deck"])
    seat = seed % 2
    decks = (my_deck, opp_deck) if seat == 0 else (opp_deck, my_deck)
    obs, sd = battle_start(list(decks[0]), list(decks[1]))
    if obs is None:
        raise RuntimeError(f"battle_start failed {sd.errorType}")
    lat_sum = 0.0
    lat_n = 0
    steps = 0
    first = -1
    try:
        while obs["current"]["result"] < 0:
            if first < 0:
                first = int(obs["current"].get("firstPlayer", -1))
            fp = 1 if first == seat else 0
            if obs.get("select") is None or steps >= DECISION_CAP:
                return 3, seat, fp, lat_n, lat_sum, steps
            yi = obs["current"]["yourIndex"]
            if yi == seat:
                t = time.perf_counter()
                act = _AGENT.act(obs)
                lat_sum += time.perf_counter() - t
                lat_n += 1
            else:
                act = _OPP.act(obs) if _OPP is not None else _random_act(obs)
            obs = battle_select(act)
            steps += 1
        res = obs["current"]["result"]
        fp = 1 if first == seat else 0
        if res == 2:
            return 2, seat, fp, lat_n, lat_sum, steps
        return (1 if res == seat else 0), seat, fp, lat_n, lat_sum, steps
    finally:
        battle_finish()


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - half) / den, (c + half) / den


def run_match(cfg: dict, games: int, workers: int, seed_base: int = 0) -> dict:
    """Play `games` with the given eval config; returns aggregate stats.

    Same seed_base + games -> identical deals/seats/opponent rolls, so two
    checkpoints evaluated with the same seeds form a paired comparison."""
    w = l = d = ab = 0
    seat_w = [0, 0]
    seat_n = [0, 0]
    first_w = [0, 0]   # [went second, went first]
    first_n = [0, 0]
    lat_sum = 0.0
    lat_n = 0
    t0 = time.time()
    # spawn, not fork: forked children inherit whatever thread pools the parent
    # already instantiated (their size is latched at library init, env vars set
    # after that are ignored). A spawned child re-runs this module's top-level
    # env clamp before its first numpy/torch import, so the cap always holds.
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(cfg,),
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        for res, seat, fp, n, s, _steps in ex.map(
                _play, range(seed_base, seed_base + games)):
            if res == 3:
                ab += 1
            else:
                seat_n[seat] += 1
                first_n[fp] += 1
                if res == 1:
                    w += 1
                    seat_w[seat] += 1
                    first_w[fp] += 1
                elif res == 0:
                    l += 1
                else:
                    d += 1
            lat_sum += s
            lat_n += n
    dt = time.time() - t0
    dec = max(1, w + l + d)  # decided games (aborts excluded from WR)
    wr = (w + 0.5 * d) / dec
    lo, hi = _wilson(wr, dec)
    return {"games": games, "w": w, "l": l, "d": d, "abort": ab,
            "wr": wr, "wr_lo": lo, "wr_hi": hi,
            "wr_seat0": seat_w[0] / max(1, seat_n[0]),
            "wr_seat1": seat_w[1] / max(1, seat_n[1]),
            "wr_first": first_w[1] / max(1, first_n[1]),
            "wr_second": first_w[0] / max(1, first_n[0]),
            "n_first": first_n[1], "n_second": first_n[0],
            "lat_ms": 1000 * lat_sum / max(1, lat_n),
            "games_per_s": games / dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--deck", default=DEFAULT_DECK)
    ap.add_argument("--opponent", default="random", help="random | self | ckpt:PATH")
    ap.add_argument("--opp-deck", default=None,
                    help="default: sample deck vs random, own deck otherwise")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--mcts-budget", type=float, default=0.0,
                    help="seconds per searched decision; 0 = plain policy")
    ap.add_argument("--mcts-det", type=int, default=4)
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--seed-base", type=int, default=0,
                    help="first game seed; reuse across runs for paired A/B")
    ap.add_argument("--device", default="cpu",
                    help="cuda offloads policy nets of both sides (not MCTS)")
    args = ap.parse_args()
    sub_dir = None
    if args.ckpt.endswith(".tar.gz") or (
            os.path.isdir(args.ckpt) and os.path.exists(os.path.join(args.ckpt, "main.py"))):
        sub_dir = prepare_submission(args.ckpt)
        args.deck = os.path.join(sub_dir, "deck.csv")
        if args.mcts_budget > 0:
            print("note: --mcts-* ignored for packaged submissions (search is baked in)")
            args.mcts_budget = 0.0
        if args.opponent == "self":
            raise SystemExit("--opponent self is unsupported for submission protagonists")
    opponent = args.opponent
    if opponent.startswith("ckpt:"):
        # workers chdir into the submission dir; all paths must be absolute
        opponent = "ckpt:" + os.path.abspath(opponent[5:])
    opp_deck = args.opp_deck or (SAMPLE_DECK if args.opponent == "random" else args.deck)
    mcts = ({"budget": args.mcts_budget, "det": args.mcts_det,
             "prior": os.path.abspath(args.prior)} if args.mcts_budget > 0 else None)
    cfg = {"ckpt": os.path.abspath(args.ckpt), "cards": os.path.abspath(args.cards),
           "deck": os.path.abspath(args.deck), "opp_deck": os.path.abspath(opp_deck),
           "opponent": opponent, "temperature": args.temperature, "mcts": mcts,
           "sub_dir": sub_dir, "device": args.device,
           "prior": os.path.abspath(args.prior) if os.path.exists(args.prior) else None}
    r = run_match(cfg, args.games, args.workers, seed_base=args.seed_base)
    print(f"games={r['games']} W/L/D/abort={r['w']}/{r['l']}/{r['d']}/{r['abort']}  "
          f"WR={r['wr']:.3f} [95% {r['wr_lo']:.3f}..{r['wr_hi']:.3f}]  "
          f"seat0={r['wr_seat0']:.3f} seat1={r['wr_seat1']:.3f}  "
          f"first={r['wr_first']:.3f}(n={r['n_first']}) "
          f"second={r['wr_second']:.3f}(n={r['n_second']})  "
          f"({r['games_per_s']:.2f} games/s)  latency {r['lat_ms']:.1f} ms")
    if r["abort"]:
        print(f"WARNING: {r['abort']} aborted games -- investigate before submitting")


if __name__ == "__main__":
    main()
