"""Self-play data generator for value-head fine-tuning.

Plays a FROZEN checkpoint against the opponent pool (arena-share sampling,
plus an optional mirror fraction) and dumps every recorded decision state
with the game's final outcome z in {-1, 0, +1}. No learning happens here;
the shards feed ptcg_rl.train_value.

Reuses the PPO actor's game loop (_play_game), so the stored encodings are
bit-identical to what the model sees in PPO training and (modulo sampling
temperature) at deployment. Only the frozen policy's own seats are recorded:
vs-pool games yield one trajectory, mirror games yield two.

    python -m ptcg_rl.gen_value_data --ckpt runs/bc_mega_lucario_v6/model.pt \
        --deck data/decks/mega_lucario_de29c8c2.csv \
        --opp-config configs/opponents_env_0810.json \
        --games 12000 --workers 20 --out build/value_data_lucario
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import multiprocessing as mp
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))
sys.path.insert(0, ROOT)

from ptcg_rl.selfplay_ppo import (FIXED_F32, FIXED_INT, RAGGED,  # noqa: E402
                                  _play_game)

SHARD_DECISIONS = 60_000  # ~decisions per output shard


def _flush_shard(trajs: list[dict], out_dir: str, rank: int, shard_no: int) -> int:
    """Concatenate buffered trajectories into one npz shard. -> decisions"""
    out: dict = {}
    for k in FIXED_F32 + FIXED_INT + RAGGED:
        out[k] = np.concatenate([t[k] for t in trajs], axis=0)
    for k in ("stadium_id", "min_c", "max_c"):
        out[k] = np.concatenate([t[k] for t in trajs])
    out["lens"] = np.concatenate([np.diff(t["opt_offsets"]) for t in trajs]) \
        .astype(np.int32)
    out["z"] = np.concatenate([np.full(t["n"], float(t["reward"]), dtype=np.float32)
                               for t in trajs])
    # fractional game progress of each decision (for calibration buckets)
    out["prog"] = np.concatenate([np.arange(t["n"], dtype=np.float32)
                                  / max(1, t["n"] - 1) for t in trajs])
    out["traj_lens"] = np.array([t["n"] for t in trajs], dtype=np.int32)
    opp_names = sorted({t["opp"] for t in trajs})
    opp_idx = {o: i for i, o in enumerate(opp_names)}
    out["traj_opp"] = np.array([opp_idx[t["opp"]] for t in trajs], dtype=np.int16)
    out["opp_names"] = np.array(opp_names)
    n = int(out["z"].shape[0])
    path = os.path.join(out_dir, f"shard_{rank:02d}_{shard_no:03d}.npz")
    np.savez_compressed(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    return n


def _worker(rank: int, a: dict) -> None:
    torch.set_num_threads(1)
    rng = random.Random(a["seed"] * 100003 + rank)
    torch.manual_seed(rng.randint(0, 2**31))

    from ptcg_rl.agent import resolve_net_cfg
    from ptcg_rl.cards import load as load_cards
    from ptcg_rl.features import ARCH_F, N_BOARD_V6  # noqa: F401
    from ptcg_rl.model import PolicyNet
    from ptcg_rl.opponents import OpponentPool

    init = torch.load(a["ckpt"], map_location="cpu", weights_only=True)
    net_cfg = resolve_net_cfg(init["config"], a["ckpt"])
    tables = load_cards(a["cards"])
    policy = PolicyNet(tables["card_feats"], tables["attack_feats"], **net_cfg)
    policy.load_state_dict(init["model"])
    policy.eval()

    arch_fn = None
    if net_cfg.get("arch_f", 0) > 0 and a.get("prior") and os.path.exists(a["prior"]):
        from ptcg_rl.deck_infer import DeckPrior, basics_from_cards
        prior = DeckPrior(a["prior"], basics_from_cards(tables["card_feats"]))
        arch_fn = prior.arch_feature
    pool = OpponentPool(a["opp_config"], a["deck"], a["cards"],
                        quiet=rank != 0, prior_path=a.get("prior"))
    feat_v6 = net_cfg.get("n_board", 12) >= N_BOARD_V6
    probe = None
    if net_cfg.get("opt_f", 0) > 60:
        from ptcg_rl.fwd_features import (FWD_MAX_OPTIONS, FWD_MAX_OPTIONS_V6,
                                          ForwardProbe)
        probe = ForwardProbe(tables["card_feats"], deck=pool.our_deck,
                             n_cols=net_cfg["opt_f"] - 60,
                             max_options=(FWD_MAX_OPTIONS_V6 if feat_v6
                                          else FWD_MAX_OPTIONS))
        if not probe._ensure_io():
            raise RuntimeError("policy wants forward features but the cg "
                               "engine failed to load")

    buf: list[dict] = []
    buf_dec = 0
    shard_no = 0
    done = 0
    err = 0
    t0 = time.time()
    while done < a["games_per_worker"]:
        if rng.random() < a["mirror_frac"]:
            name, opp, opp_deck = "mirror", "mirror", pool.our_deck
        else:
            name, opp, opp_deck = pool.sample(rng)
        try:
            with torch.no_grad():
                trajs = _play_game(policy, name, opp, pool.our_deck, opp_deck,
                                   rng, arch_fn=arch_fn, shape_c=0.0,
                                   gamma=1.0, probe=probe, v6=feat_v6)
        except Exception as e:  # noqa: BLE001
            err += 1
            if err % 20 == 1:
                print(f"[gen{rank}] game error #{err}: "
                      f"{type(e).__name__}: {e}", flush=True)
            continue
        done += 1
        for t in trajs:
            buf.append(t)
            buf_dec += t["n"]
        if buf_dec >= SHARD_DECISIONS:
            _flush_shard(buf, a["out"], rank, shard_no)
            shard_no += 1
            buf, buf_dec = [], 0
        if rank == 0 and done % 100 == 0:
            rate = done / max(1e-9, time.time() - t0)
            print(f"[gen0] {done}/{a['games_per_worker']} games "
                  f"({rate:.2f}/s per worker, ~{rate * a['workers']:.1f}/s total)",
                  flush=True)
    if buf:
        _flush_shard(buf, a["out"], rank, shard_no)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="frozen policy checkpoint")
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opp-config", required=True)
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--games", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--mirror-frac", type=float, default=0.3,
                    help="fraction of games played as mirror (records both seats)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gen_config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    a = vars(args) | {"games_per_worker": (args.games + args.workers - 1)
                      // args.workers}
    a["opp_config"] = os.path.abspath(args.opp_config)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, a)) for r in range(args.workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise SystemExit(f"{len(bad)} workers failed (exit codes {bad})")
    shards = [f for f in os.listdir(args.out) if f.endswith(".npz")]
    print(f"done: {args.games} games -> {len(shards)} shards in {args.out} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
