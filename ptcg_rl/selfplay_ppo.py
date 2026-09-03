"""Self-play PPO: CPU actor pool + GPU learner + league opponent pool.

Init from the BC checkpoint; actors play full games vs sampled opponents
(mirror / league snapshots / fixed BC agents / random) and stream decision
trajectories to the learner, which runs clipped PPO and publishes weights.

Training box:
    python -m ptcg_rl.selfplay_ppo --init runs/bc_v1/model.pt --actors 20 \
        --updates 300 --out runs/ppo_v1
Smoke:
    python -m ptcg_rl.selfplay_ppo --init runs/bc_smoke/model.pt --actors 2 \
        --rollout 192 --minibatch 96 --updates 2 --league-every 1 --out runs/ppo_smoke
"""

from __future__ import annotations

import os

# Must precede numpy/torch: spawned actors re-import this module, and BLAS /
# OpenMP pools latch their size at library load. Hard assignment because
# container images often preset OMP_NUM_THREADS (setdefault would be a no-op).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))
sys.path.insert(0, ROOT)

from ptcg_rl.agent import batch_of_one  # noqa: E402
from ptcg_rl.cards import load as load_cards  # noqa: E402
from ptcg_rl.features import (ARCH_F, N_BOARD_V6, LogMemory, encode_obs,  # noqa: E402
                              oracle_feats, own_remaining)
from ptcg_rl.model import PolicyNet  # noqa: E402

FIXED_F32 = ("glob", "ctx", "board", "arch", "logs", "look_feats", "mem", "ora")
FIXED_INT = ("ctx_ids", "board_ids", "hand_ids", "look_ids", "disc_ids",
             "deck_ids", "logs_ids", "mem_ids", "ora_ids")
RAGGED = ("opt_feats", "opt_card", "opt_tgt", "opt_atk")
MAX_SEQ = 24  # padded action-sequence buffer (measured max forced pick: 21)
DECISION_CAP = 2000


def _load_ckpt_into(model: PolicyNet, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])


def _save_ckpt(model: PolicyNet, cfg: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": cfg}, tmp)
    os.replace(tmp, path)


# ============================================================== actor

def _record_decision(cols: dict, enc: dict, seq: list[int], logp: float, value: float,
                     mn: int, mx: int) -> None:
    for k in FIXED_F32 + FIXED_INT + RAGGED:
        cols[k].append(enc[k])
    cols["stadium_id"].append(int(enc["stadium_id"]))
    s = np.full(MAX_SEQ, -1, dtype=np.int16)
    s[:len(seq)] = seq
    cols["seq"].append(s)
    cols["k"].append(len(seq))
    cols["min_c"].append(mn)
    cols["max_c"].append(min(mx, 127))
    cols["logp"].append(logp)
    cols["value"].append(value)
    cols["rew"].append(0.0)  # shaped rewards accumulate here; terminal added at pack


def _pack_traj(cols: dict, reward: float, opp: str) -> dict | None:
    n = len(cols["k"])
    if n == 0:
        return None
    out: dict = {"n": n, "reward": np.float32(reward), "opp": opp}
    for k in FIXED_F32 + FIXED_INT:
        out[k] = np.stack(cols[k])
    for k in RAGGED:
        out[k] = np.concatenate(cols[k], axis=0)
    lens = np.array([len(x) for x in cols["opt_card"]], dtype=np.int64)
    out["opt_offsets"] = np.concatenate([[0], np.cumsum(lens)])
    out["stadium_id"] = np.array(cols["stadium_id"], dtype=np.int16)
    out["seq"] = np.stack(cols["seq"])
    for k, dt in (("k", np.int16), ("min_c", np.int16), ("max_c", np.int16)):
        out[k] = np.array(cols[k], dtype=dt)
    out["logp"] = np.array(cols["logp"], dtype=np.float32)
    out["value"] = np.array(cols["value"], dtype=np.float32)
    rewards = np.array(cols["rew"], dtype=np.float32)
    rewards[-1] += reward  # terminal outcome on top of shaping deltas
    out["rewards"] = rewards
    return out


def _prize_phi(cur: dict, side: int) -> float:
    """Prize-race potential from `side`'s perspective: taken_by_me - taken_by_opp."""
    pls = cur.get("players") or [{}, {}]
    return float(len(pls[1 - side].get("prize") or []) - len(pls[side].get("prize") or []))


def _play_game(policy: PolicyNet, opp_name: str, opp_actor, our_deck: list[int],
               opp_deck: list[int], rng: random.Random, arch_fn=None,
               shape_c: float = 0.0, gamma: float = 1.0, probe=None,
               v6: bool = False):
    """-> list of trajectory dicts (2 for mirror games, else 1).

    Truncated games (decision cap / engine stall: result still < 0) are
    dropped entirely -- treating them as terminal with reward 0 would teach
    the critic that unfinished positions are worthless."""
    from cg.game import battle_start, battle_select, battle_finish
    from ptcg_rl.opponents import opponent_act

    mirror = opp_actor == "mirror"
    seat = rng.randint(0, 1)
    decks = (our_deck, opp_deck) if seat == 0 else (opp_deck, our_deck)
    obs, sd = battle_start(list(decks[0]), list(decks[1]))
    if obs is None:
        raise RuntimeError(f"battle_start failed: {sd.errorType}")
    new_cols = lambda: defaultdict(list)  # noqa: E731
    rec = {0: new_cols(), 1: new_cols()}
    mems = {0: LogMemory(), 1: LogMemory()}
    hand_snap: dict[int, list[int]] = {0: [], 1: []}  # oracle: hand at own last decision
    phi_prev: dict[int, float | None] = {0: None, 1: None}
    poisoned = {0: False, 1: False}  # missing step -> whole trajectory invalid
    if hasattr(opp_actor, "reset"):
        opp_actor.reset()  # cached league/bc agents carry per-game memory
    steps = 0
    result = 2
    first_player = -1
    try:
        while obs["current"]["result"] < 0 and steps < DECISION_CAP:
            if obs.get("select") is None:
                break
            if first_player < 0:
                first_player = int(obs["current"].get("firstPlayer", -1))
            yi = int(obs["current"]["yourIndex"])
            sel = obs["select"]
            hand = obs["current"]["players"][yi].get("hand")
            if hand is not None:
                hand_snap[yi] = [int(c.get("id") or 0) for c in hand if c]
            if yi == seat or mirror:
                mems[yi].update(obs)
                if shape_c > 0.0:
                    # potential-based prize shaping F = c*(gamma*phi' - phi),
                    # credited to the previous decision of this side
                    phi = _prize_phi(obs["current"], yi)
                    if phi_prev[yi] is not None and rec[yi]["rew"]:
                        rec[yi]["rew"][-1] += shape_c * (gamma * phi - phi_prev[yi])
                    phi_prev[yi] = phi
                fwd = None
                if probe is not None:
                    probe.set_deck(decks[yi])
                    fwd = probe.probe(obs)
                own = own_remaining(obs, decks[yi]) if v6 else None
                enc = encode_obs(obs, mem=mems[yi], fwd=fwd, v6=v6,
                                 own_remain=own)
                enc["arch"] = (arch_fn(obs, extra_ids=mems[yi].op_known)
                               if arch_fn is not None
                               else np.zeros(ARCH_F, dtype=np.float32))
                opp_hc = int(obs["current"]["players"][1 - yi].get("handCount") or 0)
                enc["ora"], enc["ora_ids"] = oracle_feats(hand_snap[1 - yi], opp_hc)
                b = batch_of_one(enc, len(sel["option"]))
                b["ora"] = torch.from_numpy(enc["ora"].astype(np.float32)).unsqueeze(0)
                b["ora_ids"] = torch.from_numpy(enc["ora_ids"].astype(np.int32)).unsqueeze(0)
                mn = int(sel.get("minCount") or 0)
                mx = int(sel.get("maxCount") or 0)
                seq, logp, _v_blind, v_ora = policy.sample_action(b, mn, mx)
                # oracle value drives GAE (lower-variance critic)
                if len(seq) <= MAX_SEQ:
                    _record_decision(rec[yi], enc, seq, logp, v_ora, mn, mx)
                else:
                    # cannot store this step; a trajectory with a hole would
                    # mis-assign credit for everything after it -> drop it all
                    poisoned[yi] = True
                act = seq
            else:
                act = opponent_act(opp_actor, obs, rng)
            obs = battle_select(act)
            steps += 1
        result = obs["current"]["result"]
    finally:
        battle_finish()

    if result < 0:
        return []  # truncated, not terminal: no outcome signal to learn from

    def rw(side: int) -> float:
        if result == 2:
            return 0.0
        return 1.0 if result == side else -1.0

    trajs = []
    sides = (0, 1) if mirror else (seat,)
    for s in sides:
        if poisoned[s]:
            continue
        if shape_c > 0.0 and phi_prev[s] is not None and rec[s]["rew"]:
            phi_end = _prize_phi(obs.get("current") or {}, s)
            rec[s]["rew"][-1] += shape_c * (gamma * phi_end - phi_prev[s])
        t = _pack_traj(rec[s], rw(s), opp_name)
        if t is not None:
            # diagnostics: how the game was lost/won, not just that it was
            t["prize_diff"] = _prize_phi(obs.get("current") or {}, s)
            t["first"] = 1 if first_player == s else 0
            trajs.append(t)
    return trajs


def actor_main(rank: int, a: dict, version, queue, stop, shape_v=None) -> None:
    torch.set_num_threads(1)
    rng = random.Random(10007 * rank + a["seed"])
    torch.manual_seed(rng.randint(0, 2**31))
    from ptcg_rl.opponents import OpponentPool

    current = os.path.join(a["out"], "current.pt")
    while not os.path.exists(current):
        time.sleep(0.5)
    tables = load_cards(a["cards"])
    policy = PolicyNet(tables["card_feats"], tables["attack_feats"], **a["net"])
    _load_ckpt_into(policy, current)
    policy.eval()
    arch_fn = None
    if a["net"].get("arch_f", 0) > 0 and a.get("prior") and os.path.exists(a["prior"]):
        from ptcg_rl.deck_infer import DeckPrior, basics_from_cards
        prior = DeckPrior(a["prior"], basics_from_cards(tables["card_feats"]))
        arch_fn = prior.arch_feature
    pool = OpponentPool(a["opp_config"], a["deck"], a["cards"],
                        league_dir=os.path.join(a["out"], "pool"), quiet=rank != 0,
                        prior_path=a.get("prior"))
    feat_v6 = a["net"].get("n_board", 12) >= N_BOARD_V6
    probe = None
    if a["net"].get("opt_f", 0) > 60:  # net consumes engine-lookahead columns
        from ptcg_rl.fwd_features import (FWD_MAX_OPTIONS, FWD_MAX_OPTIONS_V6,
                                          ForwardProbe)
        probe = ForwardProbe(tables["card_feats"], deck=pool.our_deck,
                             n_cols=a["net"]["opt_f"] - 60,
                             max_options=(FWD_MAX_OPTIONS_V6 if feat_v6
                                          else FWD_MAX_OPTIONS))
        if not probe._ensure_io():
            raise RuntimeError("policy wants forward features but the cg "
                               "engine failed to load in the actor")
    local_ver = version.value
    err = 0
    opp_w_path = os.path.join(a["out"], "opp_weights.json")
    while not stop.value:
        if version.value != local_ver:
            try:
                _load_ckpt_into(policy, current)
                local_ver = version.value
                pool.reload_weights(opp_w_path)  # adaptive opponent sampling
            except Exception:
                time.sleep(0.2)
        name, opp, opp_deck = pool.sample(rng)
        shape_c = float(shape_v.value) if shape_v is not None else a.get("shape", 0.0)
        try:
            with torch.no_grad():
                trajs = _play_game(policy, name, opp, pool.our_deck, opp_deck, rng,
                                   arch_fn=arch_fn, shape_c=shape_c,
                                   gamma=a.get("gamma", 1.0), probe=probe,
                                   v6=feat_v6)
        except Exception as e:  # noqa: BLE001
            err += 1
            if err % 20 == 1:
                print(f"[actor{rank}] game error #{err}: {type(e).__name__}: {e}", flush=True)
            continue
        for t in trajs:
            t["ver"] = local_ver  # learner drops rollouts from stale policies
            queue.put(t)


# ============================================================== learner

class _PPOShim(torch.nn.Module):
    """DDP only syncs gradients for work routed through its own forward();
    wrap action_logprob_entropy so the reducer hooks fire."""

    def __init__(self, net: PolicyNet):
        super().__init__()
        self.net = net

    def forward(self, b, seq, kk):
        return self.net.action_logprob_entropy(b, seq, kk)


class Buffer:
    def __init__(self):
        self.trajs: list[dict] = []
        self.decisions = 0

    def add(self, t: dict) -> None:
        self.trajs.append(t)
        self.decisions += t["n"]

    def finalize(self, gamma: float, lam: float, device) -> dict:
        flat: dict[str, list] = defaultdict(list)
        adv_all, ret_all = [], []
        for t in self.trajs:
            v = t["value"]
            n = t["n"]
            rews = t.get("rewards")
            adv = np.zeros(n, dtype=np.float32)
            last = 0.0
            for i in reversed(range(n)):
                nxt = v[i + 1] if i + 1 < n else 0.0
                if rews is not None:
                    r = float(rews[i])
                else:
                    r = float(t["reward"]) if i == n - 1 else 0.0
                delta = r + gamma * nxt - v[i]
                last = delta + gamma * lam * last
                adv[i] = last
            adv_all.append(adv)
            ret_all.append(adv + v)
            for k in FIXED_F32 + FIXED_INT + RAGGED + (
                    "stadium_id", "seq", "k", "min_c", "max_c", "logp", "value"):
                flat[k].append(t[k])
            off = t["opt_offsets"]
            flat["lens"].append(np.diff(off))
        out: dict = {}
        for k in FIXED_F32 + FIXED_INT + ("stadium_id", "seq", "k", "min_c", "max_c",
                                          "logp", "value"):
            out[k] = np.concatenate(flat[k], axis=0)
        for k in RAGGED:
            out[k] = np.concatenate(flat[k], axis=0)
        out["lens"] = np.concatenate(flat["lens"])
        out["opt_offsets"] = np.concatenate([[0], np.cumsum(out["lens"])])
        adv = np.concatenate(adv_all)
        # pre-normalization signal strength: ~0 means the critic already
        # explains the outcomes and PPO has no gradient left to follow
        out["adv_abs"] = float(np.abs(adv).mean())
        ret = np.concatenate(ret_all)
        var_ret = float(np.var(ret))
        out["ev"] = (1.0 - float(np.var(ret - out["value"])) / var_ret
                     if var_ret > 1e-8 else 0.0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        out["adv"] = adv
        out["ret"] = ret
        return out

    def stats(self):
        by_opp: dict[str, list[float]] = defaultdict(list)
        ep_len = []
        per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for t in self.trajs:
            by_opp[t["opp"]].append(float(t["reward"]))
            ep_len.append(t["n"])
            # diag per archetype: "bc:dragapult__f12c8bec" -> "bc:dragapult";
            # per-exact-deck samples are too few per update to be meaningful
            d = per[t["opp"].split("__")[0]]
            d["rw"].append(float(t["reward"]))
            d["ep"].append(t["n"])
            d["pd"].append(float(t.get("prize_diff", 0.0)))
            d["first"].append(int(t.get("first", -1)))
            d["v0"].append(float(t["value"][0]))
        diag = {}
        for name, d in per.items():
            rw = np.array(d["rw"])
            fi = np.array(d["first"])
            win = (rw > 0).astype(np.float64) + 0.5 * (rw == 0)
            pd = np.array(d["pd"])
            diag[name] = {
                "n": len(rw),
                "ep": round(float(np.mean(d["ep"])), 1),
                # avg prize differential at game end (blowout vs close loss)
                "pdiff": round(float(pd.mean()), 2),
                "pdiff_loss": (round(float(pd[rw < 0].mean()), 2)
                               if (rw < 0).any() else None),
                # critic's first-decision estimate: if it matches the final
                # mean outcome, the matchup is decided before play starts
                "v0": round(float(np.mean(d["v0"])), 2),
                "wr_first": (round(float(win[fi == 1].mean()), 3)
                             if (fi == 1).any() else None),
                "wr_second": (round(float(win[fi == 0].mean()), 3)
                              if (fi == 0).any() else None),
            }
        return by_opp, float(np.mean(ep_len)) if ep_len else 0.0, diag

    def clear(self):
        self.trajs.clear()
        self.decisions = 0


_SHARD_FIXED = FIXED_F32 + FIXED_INT + ("stadium_id", "seq", "k", "min_c",
                                        "max_c", "logp", "value", "adv", "ret")


def _shard(data: dict, idx: np.ndarray) -> dict:
    """Row-subset of a finalized rollout (ragged option arrays re-gathered).

    Lets the DDP learner scatter per-rank shards instead of broadcasting the
    full rollout: at 8M+ decisions a full copy per rank would exhaust host RAM.
    """
    out: dict = {}
    for k in _SHARD_FIXED:
        out[k] = data[k][idx]
    off = data["opt_offsets"]
    lens = data["lens"][idx]
    segs = [slice(off[j], off[j + 1]) for j in idx]
    for k in RAGGED:
        arr = data[k]
        out[k] = np.concatenate([arr[s] for s in segs], axis=0)
    out["lens"] = lens
    out["opt_offsets"] = np.concatenate([[0], np.cumsum(lens)])
    for k in ("adv_abs", "ev"):
        out[k] = data[k]
    return out


def _minibatch(data: dict, idx: np.ndarray, device) -> dict[str, torch.Tensor]:
    off = data["opt_offsets"]
    lens = data["lens"]
    K = int(lens[idx].max())
    B = len(idx)
    b: dict[str, torch.Tensor] = {}
    for k in FIXED_F32:
        b[k] = torch.from_numpy(data[k][idx].astype(np.float32))
    for k in FIXED_INT + ("stadium_id",):
        b[k] = torch.from_numpy(data[k][idx].astype(np.int32))
    opt_feats = np.zeros((B, K, data["opt_feats"].shape[1]), dtype=np.float32)
    oc = np.zeros((B, K), dtype=np.int32)
    ot = np.zeros((B, K), dtype=np.int32)
    oa = np.zeros((B, K), dtype=np.int32)
    mask = np.zeros((B, K), dtype=bool)
    for i, j in enumerate(idx):
        s, e = off[j], off[j + 1]
        n = e - s
        opt_feats[i, :n] = data["opt_feats"][s:e]
        oc[i, :n] = data["opt_card"][s:e]
        ot[i, :n] = data["opt_tgt"][s:e]
        oa[i, :n] = data["opt_atk"][s:e]
        mask[i, :n] = True
    b["opt_feats"] = torch.from_numpy(opt_feats)
    b["opt_card"] = torch.from_numpy(oc)
    b["opt_tgt"] = torch.from_numpy(ot)
    b["opt_atk"] = torch.from_numpy(oa)
    b["opt_mask"] = torch.from_numpy(mask)
    for k in ("min_c", "max_c"):
        b[k] = torch.from_numpy(data[k][idx].astype(np.int64))
    b = {kk: vv.to(device, non_blocking=True) for kk, vv in b.items()}
    extra = {
        "seq": torch.from_numpy(data["seq"][idx].astype(np.int64)).to(device),
        "kk": torch.from_numpy(data["k"][idx].astype(np.int64)).to(device),
        "logp": torch.from_numpy(data["logp"][idx]).to(device),
        "adv": torch.from_numpy(data["adv"][idx]).to(device),
        "ret": torch.from_numpy(data["ret"][idx]).to(device),
        "v_old": torch.from_numpy(data["value"][idx]).to(device),
    }
    return b, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="BC checkpoint to start from")
    ap.add_argument("--out", default="runs/ppo_v1")
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--deck", default=os.path.join(ROOT, "data", "decks", "mega_lopunny_220eddd2.csv"))
    ap.add_argument("--opp-config", default=os.path.join(ROOT, "configs", "opponents_default.json"))
    ap.add_argument("--actors", type=int, default=20)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--rollout", type=int, default=16384, help="decisions per update")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--kl-stop", type=float, default=0.03)
    ap.add_argument("--shrink-value", type=float, default=0.25,
                    help="scale BC value head at init (BC logit scale != return scale)")
    ap.add_argument("--prize-shaping", type=float, default=0.05,
                    help="potential-based prize-diff reward shaping coef (0 = off)")
    ap.add_argument("--shape-anneal", type=int, default=0,
                    help="linearly anneal shaping to 0 over N updates (0 = constant)")
    ap.add_argument("--kl-bc", type=float, default=0.05,
                    help="KL(pi||pi_BC) regularizer initial coef (0 = off)")
    ap.add_argument("--kl-bc-anneal", type=int, default=150,
                    help="updates over which the KL-to-BC coef decays linearly to 0")
    ap.add_argument("--league-every", type=int, default=20)
    ap.add_argument("--adapt-opp", type=float, default=2.0,
                    help="adaptive opponent sampling exponent beta: sampling "
                         "weight = arena_share * ((1-wr)/0.5)^beta clamped to "
                         "[1, 4] (upweight-only; easy opponents shrink via "
                         "renormalization). 0 = off (sample by arena share). "
                         "wr_pool stays share-weighted either way")
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # -------- distributed learner (launch with torchrun --nproc_per_node=N).
    # rank0 owns the actor pool, collects the rollout and broadcasts it; every
    # rank trains on an interleaved shard of minibatches (DDP averages grads,
    # so N ranks x minibatch M behave like one step on a N*M batch).
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp_backend = None
    if world > 1:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
        ddp_backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        from datetime import timedelta
        # ranks >0 idle in broadcast while rank0 collects the rollout; keep the
        # collective timeout well above any plausible collect stall
        dist.init_process_group(ddp_backend, timeout=timedelta(hours=2))
        # rollout broadcast goes over this CPU group: NCCL object broadcast
        # would stage the multi-GB pickled rollout on rank0's GPU
        pg_cpu = (dist.new_group(backend="gloo", timeout=timedelta(hours=2))
                  if ddp_backend == "nccl" else None)
        if rank == 0:
            print(f"[ddp] {world} ranks, backend={ddp_backend}", flush=True)
    else:
        dist = None

    os.makedirs(os.path.join(args.out, "pool"), exist_ok=True)
    init = torch.load(args.init, map_location="cpu", weights_only=True)
    from ptcg_rl.agent import resolve_net_cfg
    net_cfg = resolve_net_cfg(init["config"], args.init)
    tables = load_cards(args.cards)
    model = PolicyNet(tables["card_feats"], tables["attack_feats"], **net_cfg)
    model.load_state_dict(init["model"])
    with torch.no_grad():
        model.value_head.weight.mul_(args.shrink_value)
        model.value_head.bias.mul_(args.shrink_value)
        # oracle head drives GAE in actors -> must be rescaled the same way
        model.value_oracle[-1].weight.mul_(args.shrink_value)
        model.value_oracle[-1].bias.mul_(args.shrink_value)
    model.to(args.device)
    ref_model = None
    if args.kl_bc > 0:
        ref_model = PolicyNet(tables["card_feats"], tables["attack_feats"], **net_cfg)
        ref_model.load_state_dict(init["model"])
        ref_model.to(args.device).eval()
        for prm in ref_model.parameters():
            prm.requires_grad_(False)
    ckpt_cfg = dict(init["config"])
    ckpt_cfg["value_mode"] = "signed"  # PPO regresses [-1,1] returns (see MCTS)

    train_model: torch.nn.Module = _PPOShim(model)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_model = DDP(
            train_model,
            device_ids=[local_rank] if ddp_backend == "nccl" else None,
            find_unused_parameters=True)

    actors: list = []
    version = stop = shape_v = queue = None
    if rank == 0:
        _save_ckpt(model, ckpt_cfg, os.path.join(args.out, "current.pt"))
        ctx = mp.get_context("spawn")
        version = ctx.Value("i", 1)
        stop = ctx.Value("i", 0)
        shape_v = ctx.Value("d", args.prize_shaping)
        queue = ctx.Queue(maxsize=256)
        a = {"out": args.out, "cards": args.cards, "deck": args.deck,
             "opp_config": args.opp_config, "net": net_cfg, "prior": args.prior,
             "shape": args.prize_shaping, "gamma": args.gamma,
             "seed": int(time.time()) % 100000}
        actors = [ctx.Process(target=actor_main,
                              args=(r, a, version, queue, stop, shape_v), daemon=True)
                  for r in range(args.actors)]
        for p in actors:
            p.start()
        print(f"spawned {args.actors} actors; waiting for rollouts...", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    buf = Buffer()
    wr_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
    # arena shares of the bc opponents actually available to the actors; used
    # for the share-weighted wr_pool (sampling-distribution independent) and
    # as the base for adaptive opponent sampling
    with open(args.opp_config) as f:
        _opp_entries = json.load(f)
    arena_share: dict[str, float] = {}
    fixed_w: dict[str, float] = {}
    for e in _opp_entries:
        if e.get("kind") == "bc":
            # mutants are domain randomization: trained against, but excluded
            # from the arena-share wr_pool metric (they are not real decks)
            if e.get("name", "").startswith("mut_"):
                continue
            if os.path.exists(e["ckpt"]) and os.path.exists(e["deck"]):
                arena_share[f"bc:{e['name']}"] = float(e.get("weight", 0.0))
        else:
            fixed_w[e["kind"]] = fixed_w.get(e["kind"], 0.0) + float(e.get("weight", 0.0))
    share_tot = sum(arena_share.values())
    best_wr, best_upd = -1.0, 0
    log_path = os.path.join(args.out, "log.jsonl")
    t_start = time.time()

    for upd in range(1, args.updates + 1):
        t0 = time.time()
        stale = 0
        by_opp = ep_len = diag = None
        data = None
        if rank == 0:
            if args.shape_anneal:  # late-training shaping anneal (plan requirement)
                with shape_v.get_lock():
                    shape_v.value = args.prize_shaping * max(
                        0.0, 1.0 - (upd - 1) / max(1, args.shape_anneal))
            while buf.decisions < args.rollout:
                try:
                    t = queue.get(timeout=600)
                except Exception as exc:  # queue.Empty
                    alive = sum(p.is_alive() for p in actors)
                    raise RuntimeError(
                        f"no rollouts for 10min at upd {upd}; actors alive: "
                        f"{alive}/{len(actors)}") from exc
                if t.get("ver", 0) < version.value - 1:
                    stale += 1  # rollout from a 2+ versions old policy: drop
                    continue
                buf.add(t)
            by_opp, ep_len, diag = buf.stats()
            for name, rews in by_opp.items():
                wr_hist[name].extend(rews)
            data = buf.finalize(args.gamma, args.lam, args.device)
            n_games = len(buf.trajs)
            buf.clear()  # trajs are tens of GB at large rollouts; free now
            n_glob = len(data["k"])
        if world > 1:
            # scatter per-rank row shards (sequentially, so rank0 only holds
            # one extra shard at a time); full-rollout broadcast would put a
            # complete copy on every rank and exhaust host RAM at 8M+ rows
            if rank == 0:
                perm = np.random.default_rng(9_999_991 * upd).permutation(n_glob)
                for r in range(1, world):
                    dist.send_object_list(
                        [_shard(data, np.sort(perm[r::world]))], dst=r,
                        group=pg_cpu)
                data = _shard(data, np.sort(perm[0::world]))
            else:
                obj = [None]
                dist.recv_object_list(obj, src=0, group=pg_cpu)
                data = obj[0]
        n = n_glob if rank == 0 else len(data["k"])
        n_rows = len(data["k"])
        collect_s = time.time() - t0

        t1 = time.time()
        train_model.train()
        kl_bc_coef = args.kl_bc * max(0.0, 1.0 - (upd - 1) / max(1, args.kl_bc_anneal))
        pg_l = v_l = ent_l = kl = clipfrac = klbc_l = 0.0
        nb = 0
        stop_early = False
        for ep_i in range(args.ppo_epochs):
            # each rank shuffles its own shard (world=1: the whole rollout)
            perm = np.random.default_rng(
                1_000_003 * upd + 131 * ep_i + rank).permutation(n_rows)
            slices = [perm[s0:s0 + args.minibatch]
                      for s0 in range(0, n_rows, args.minibatch)]
            slices = [s for s in slices
                      if len(s) >= max(64, args.minibatch // 4)]  # degenerate tails
            if world > 1:
                # equal minibatch count per rank keeps grad all-reduces aligned
                cnt = torch.tensor([len(slices)], device=args.device)
                dist.all_reduce(cnt, op=dist.ReduceOp.MIN)
                slices = slices[:int(cnt.item())]
            for idx in slices:
                b, ex = _minibatch(data, idx, args.device)
                lp, ent, v, v_ora, logits, _cnt = train_model(
                    b, ex["seq"], ex["kk"])
                ratio = (lp - ex["logp"]).exp()
                adv = ex["adv"]
                pg = -torch.min(ratio * adv,
                                ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                # both critics regress to returns; oracle one feeds GAE in actors,
                # blind one is what MCTS consumes at inference
                vloss = 0.5 * (v - ex["ret"]).pow(2).mean()
                if v_ora is not None:
                    vloss = vloss + 0.5 * (v_ora - ex["ret"]).pow(2).mean()
                loss = pg + args.vf_coef * vloss - args.ent_coef * ent.mean()
                if ref_model is not None and kl_bc_coef > 0:
                    # anti-forgetting anchor: KL(pi || pi_BC) on the first-pick
                    # distribution + the count head (variable-count rows)
                    with torch.no_grad():
                        ref_lg, ref_cnt, _, _ = ref_model(b)
                    m = b["opt_mask"]
                    _fm = torch.finfo(logits.dtype).min
                    lpc = torch.log_softmax(logits.masked_fill(~m, _fm), dim=1)
                    lpr = torch.log_softmax(ref_lg.masked_fill(~m, _fm), dim=1)
                    klbc = (torch.where(m, lpc.exp() * (lpc - lpr),
                                        torch.zeros_like(lpc)).sum(1)).mean()
                    var_rows = b["min_c"] != b["max_c"]
                    if var_rows.any():
                        n_opt = m.sum(1)
                        nc = _cnt.shape[1]
                        rng_c = torch.arange(nc, device=_cnt.device).unsqueeze(0)
                        cm = ((rng_c >= b["min_c"].unsqueeze(1))
                              & (rng_c <= torch.minimum(
                                  b["max_c"], n_opt).clamp(max=nc - 1).unsqueeze(1)))
                        cm = cm & var_rows.unsqueeze(1)
                        cm[:, 0] |= ~var_rows  # keep one valid slot on other rows
                        lqc = torch.log_softmax(_cnt.masked_fill(~cm, _fm), 1)
                        lqr = torch.log_softmax(ref_cnt.masked_fill(~cm, _fm), 1)
                        klc = torch.where(cm, lqc.exp() * (lqc - lqr),
                                          torch.zeros_like(lqc)).sum(1)
                        klbc = klbc + (klc * var_rows).sum() / var_rows.sum().clamp(min=1)
                    loss = loss + kl_bc_coef * klbc
                    klbc_l += float(klbc.detach())
                finite = bool(torch.isfinite(loss))
                if world > 1:
                    ok = torch.tensor([float(finite)], device=args.device)
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    finite = bool(ok.item() > 0.5)
                if not finite:
                    if world > 1:
                        # DDP needs one backward per forward on every rank;
                        # run it on a zeroed surrogate, then drop the step
                        torch.nan_to_num(loss, nan=0.0, posinf=0.0,
                                         neginf=0.0).backward()
                    opt.zero_grad(set_to_none=True)
                    if rank == 0:
                        print(f"[warn] non-finite loss at upd {upd}, minibatch skipped", flush=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    kl_mb = float((ex["logp"] - lp).mean())
                    if world > 1:
                        # pooled estimate; identical on all ranks so the
                        # kl-stop decision below cannot diverge and deadlock
                        _t = torch.tensor([kl_mb], device=args.device)
                        dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                        kl_mb = float(_t.item()) / world
                    clipfrac += float(((ratio - 1).abs() > args.clip).float().mean())
                pg_l += float(pg.detach()); v_l += float(vloss.detach())
                ent_l += float(ent.mean().detach())
                kl += kl_mb
                nb += 1
                # stop on the running-mean KL: a single-minibatch estimate is
                # too noisy to either trigger or survive the early stop
                if kl / nb > args.kl_stop:
                    stop_early = True
                    break
            if stop_early:
                break
        kl = kl / max(1, nb)
        train_s = time.time() - t1

        if rank != 0:
            continue  # checkpointing / stats / logging are rank0's job
        _save_ckpt(model, ckpt_cfg, os.path.join(args.out, "current.pt"))
        with version.get_lock():
            version.value += 1
        if upd % args.league_every == 0:
            _save_ckpt(model, ckpt_cfg, os.path.join(args.out, "pool", f"upd{upd:05d}.pt"))

        # arena-replica WR: per-opponent rolling means weighted by arena share.
        # Unlike a plain mean over bc games this stays unbiased when adaptive
        # sampling skews how often each opponent is played.
        num = den = 0.0
        bc_games = 0
        opp_wr: dict[str, float] = {}
        for name, w in arena_share.items():
            v = wr_hist.get(name)
            if v:
                opp_wr[name] = float(np.mean([(x + 1) / 2 for x in v]))
                num += w * opp_wr[name]
                den += w
                bc_games += len(v)
        wr_pool = num / den if den > 0 else None
        coverage = den / share_tot if share_tot > 0 else 0.0
        # only trust the estimate once nearly every opponent has a filled-in
        # window (avoids early-noise "best")
        if (wr_pool is not None and coverage >= 0.9 and bc_games >= 3000
                and wr_pool > best_wr):
            best_wr, best_upd = wr_pool, upd
            _save_ckpt(model, ckpt_cfg, os.path.join(args.out, "best.pt"))

        if args.adapt_opp > 0 and arena_share:
            # low-WR opponents get upsampled (they carry learnable signal),
            # farmed ones get downsampled; cold-start entries sample at share
            wmap = dict(fixed_w)
            bc_mass = max(1e-9, 1.0 - sum(fixed_w.values()))
            boosted = {}
            for name, w in arena_share.items():
                v = wr_hist.get(name)
                if v is not None and len(v) >= 30:
                    m = float(np.mean([(x + 1) / 2 for x in v]))
                    # upweight-only: hard opponents get up to 4x; farmed ones
                    # shrink solely via renormalization. A hard floor below
                    # share caused catastrophic forgetting on strong matchups
                    # (ppo_lucario_de29_v6: lopunny 0.88 -> 0.65 while boosted
                    # matchups gained the same amount, net zero)
                    boost = min(4.0, max(1.0, ((1.0 - m) / 0.5) ** args.adapt_opp))
                else:
                    boost = 1.0
                boosted[name] = w * boost
            bt = sum(boosted.values())
            for name, bw in boosted.items():
                wmap[name] = bc_mass * bw / bt
            tmp = os.path.join(args.out, "opp_weights.json.tmp")
            with open(tmp, "w") as f:
                json.dump(wmap, f)
            os.replace(tmp, os.path.join(args.out, "opp_weights.json"))

        wr = {k: round(float(np.mean([(x + 1) / 2 for x in v])), 3)
              for k, v in wr_hist.items() if v}
        # archetype-level rollup: pools the per-list windows, so rare lists
        # ride on their archetype's sample rate instead of showing noise
        arch_hist: dict[str, list] = defaultdict(list)
        for k, v in wr_hist.items():
            arch_hist[k.split("__")[0]].extend(v)
        wr_arch = {k: round(float(np.mean([(x + 1) / 2 for x in v])), 3)
                   for k, v in arch_hist.items() if v}
        line = {"upd": upd, "n": n, "games": n_games, "ep_len": round(ep_len, 1),
                "pg": round(pg_l / max(1, nb), 4), "v": round(v_l / max(1, nb), 4),
                "ent": round(ent_l / max(1, nb), 3), "kl": round(kl, 4),
                "kl_bc": round(klbc_l / max(1, nb), 4),
                "clipfrac": round(clipfrac / max(1, nb), 3), "stale": stale,
                "shape": round(float(shape_v.value), 4),
                "collect_s": round(collect_s, 1), "train_s": round(train_s, 1),
                "dec_per_s": round(n / max(0.1, collect_s)), "wr": wr,
                "wr_arch": wr_arch,
                "wr_pool": round(wr_pool, 4) if wr_pool is not None else None,
                "cov": round(coverage, 3),
                "best": [round(best_wr, 4), best_upd],
                "ev": round(float(data["ev"]), 3),
                "adv_abs": round(float(data["adv_abs"]), 4),
                "diag": diag,
                "elapsed_min": round((time.time() - t_start) / 60, 1)}
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")
        buf.clear()

    if rank == 0:
        _save_ckpt(model, ckpt_cfg, os.path.join(args.out, "model.pt"))
        if best_upd:
            print(f"best.pt: wr_pool={best_wr:.4f} at upd {best_upd}")
        stop.value = 1
        time.sleep(1.0)
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:  # noqa: BLE001
                break
        for p in actors:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        print(f"done; final model at {os.path.join(args.out, 'model.pt')}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
