"""Value-head fine-tuning on frozen-policy self-play data.

Loads a BC/PPO checkpoint, freezes EVERYTHING except the two value branches
(blind value_head; oracle_in + value_oracle), and regresses them onto the
final game outcomes recorded by ptcg_rl.gen_value_data. The policy is
untouched by construction: trunk parameters are frozen, so play behaviour of
the output checkpoint is bit-identical to the input -- only value estimates
change (they feed MCTS backups and PPO advantage estimation).

Loss follows the checkpoint's value_mode: win_logit -> BCE on (z+1)/2,
signed -> MSE on z. Calibration is reported as AUC (win vs loss states,
draws excluded), overall and by game-progress bucket.

Train:
    python -m ptcg_rl.train_value --init runs/bc_mega_lucario_v6/model.pt \
        --data 'build/value_data_lucario/*.npz' --out runs/value_lucario

Benchmark a checkpoint without training (e.g. before/after comparison):
    python -m ptcg_rl.train_value --init <ckpt> --data '...' --eval-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ptcg_rl.agent import resolve_net_cfg  # noqa: E402
from ptcg_rl.cards import load as load_cards  # noqa: E402
from ptcg_rl.model import PolicyNet  # noqa: E402
from ptcg_rl.selfplay_ppo import FIXED_F32, FIXED_INT, RAGGED  # noqa: E402

BUCKETS = ((0.0, 0.33, "early"), (0.33, 0.66, "mid"), (0.66, 1.01, "late"))


def _load_shard(path: str) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["opt_offsets"] = np.concatenate([[0], np.cumsum(d["lens"])])
    return d


def _collate(d: dict, idx: np.ndarray, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    off = d["opt_offsets"]
    lens = d["lens"]
    K = int(lens[idx].max())
    B = len(idx)
    b: dict[str, torch.Tensor] = {}
    for k in FIXED_F32:
        b[k] = torch.from_numpy(d[k][idx].astype(np.float32))
    for k in FIXED_INT + ("stadium_id",):
        b[k] = torch.from_numpy(d[k][idx].astype(np.int32))
    opt_feats = np.zeros((B, K, d["opt_feats"].shape[1]), dtype=np.float32)
    oc = np.zeros((B, K), dtype=np.int32)
    ot = np.zeros((B, K), dtype=np.int32)
    oa = np.zeros((B, K), dtype=np.int32)
    mask = np.zeros((B, K), dtype=bool)
    for i, j in enumerate(idx):
        s, e = off[j], off[j + 1]
        n = e - s
        opt_feats[i, :n] = d["opt_feats"][s:e]
        oc[i, :n] = d["opt_card"][s:e]
        ot[i, :n] = d["opt_tgt"][s:e]
        oa[i, :n] = d["opt_atk"][s:e]
        mask[i, :n] = True
    b["opt_feats"] = torch.from_numpy(opt_feats)
    b["opt_card"] = torch.from_numpy(oc)
    b["opt_tgt"] = torch.from_numpy(ot)
    b["opt_atk"] = torch.from_numpy(oa)
    b["opt_mask"] = torch.from_numpy(mask)
    for k in ("min_c", "max_c"):
        b[k] = torch.from_numpy(d[k][idx].astype(np.int64))
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    zt = torch.from_numpy(d["z"][idx]).to(device)
    pg = torch.from_numpy(d["prog"][idx]).to(device)
    return b, zt, pg


def _loss(v: torch.Tensor, z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "signed":
        return 0.5 * (v - z).pow(2).mean()
    return F.binary_cross_entropy_with_logits(v, (z + 1.0) / 2.0)


def _auc(score: np.ndarray, z: np.ndarray) -> float:
    """Mann-Whitney AUC of score separating wins (z>0) from losses (z<0)."""
    m = z != 0
    s, y = score[m], (z[m] > 0)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


@torch.no_grad()
def evaluate(model: PolicyNet, shards: list[str], mode: str, device,
             minibatch: int) -> dict:
    model.eval()
    vs, os_, zs, pgs = [], [], [], []
    loss_b = loss_o = 0.0
    nb = 0
    for path in shards:
        d = _load_shard(path)
        n = len(d["z"])
        for s0 in range(0, n, minibatch):
            idx = np.arange(s0, min(n, s0 + minibatch))
            b, z, pg = _collate(d, idx, device)
            _, _, v, v_ora = model.forward(b)
            loss_b += float(_loss(v, z, mode))
            loss_o += float(_loss(v_ora, z, mode)) if v_ora is not None else 0.0
            nb += 1
            vs.append(v.float().cpu().numpy())
            os_.append(v_ora.float().cpu().numpy() if v_ora is not None
                       else np.zeros(len(idx), dtype=np.float32))
            zs.append(z.cpu().numpy())
            pgs.append(pg.cpu().numpy())
    v = np.concatenate(vs)
    vo = np.concatenate(os_)
    z = np.concatenate(zs)
    pg = np.concatenate(pgs)
    out = {"n": int(len(z)), "loss": round(loss_b / max(1, nb), 4),
           "loss_ora": round(loss_o / max(1, nb), 4),
           "auc": round(_auc(v, z), 4), "auc_ora": round(_auc(vo, z), 4)}
    for lo, hi, name in BUCKETS:
        m = (pg >= lo) & (pg < hi)
        out[f"auc_{name}"] = round(_auc(v[m], z[m]), 4)
        out[f"auc_ora_{name}"] = round(_auc(vo[m], z[m]), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True, help="glob of gen_value_data shards")
    ap.add_argument("--out", default="runs/value_tuned")
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of shards held out (tail of sorted list)")
    ap.add_argument("--eval-only", action="store_true",
                    help="report calibration of --init on the val split and exit")
    ap.add_argument("--eval-all", action="store_true",
                    help="with --eval-only: use ALL shards, not just the val split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.data))
    if not shards:
        raise SystemExit(f"no shards match {args.data}")
    n_val = max(1, round(args.val_frac * len(shards)))
    train_shards, val_shards = shards[:-n_val], shards[-n_val:]

    init = torch.load(args.init, map_location="cpu", weights_only=True)
    net_cfg = resolve_net_cfg(init["config"], args.init)
    tables = load_cards(args.cards)
    model = PolicyNet(tables["card_feats"], tables["attack_feats"], **net_cfg)
    model.load_state_dict(init["model"])
    model.to(args.device)
    mode = init["config"].get("value_mode", "win_logit")

    if args.eval_only:
        use = shards if args.eval_all else val_shards
        r = evaluate(model, use, mode, args.device, args.minibatch)
        print(json.dumps({"ckpt": args.init, "shards": len(use), **r}))
        return

    # freeze everything except the two value branches; with the trunk frozen
    # and inputs grad-free, autograd stops at the heads automatically
    head_prefixes = ("value_head.", "oracle_in.", "value_oracle.")
    n_train_p = 0
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(head_prefixes))
        n_train_p += p.numel() if name.startswith(head_prefixes) else 0
    print(f"trainable params: {n_train_p} "
          f"(value_head + oracle_in + value_oracle only)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)

    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(row: dict) -> None:
        print(json.dumps(row), flush=True)
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

    base = evaluate(model, val_shards, mode, args.device, args.minibatch)
    log({"epoch": 0, "note": "baseline (before tuning)", **base})
    best_loss = base["loss"]

    def save(path: str) -> None:
        cfg = dict(init["config"])
        cfg["value_tuned"] = True
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": cfg}, path)

    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_b = tr_o = 0.0
        nb = 0
        order = rng.permutation(len(train_shards))
        for si in order:
            d = _load_shard(train_shards[si])
            n = len(d["z"])
            perm = rng.permutation(n)
            for s0 in range(0, n, args.minibatch):
                idx = perm[s0:s0 + args.minibatch]
                if len(idx) < 32:
                    continue
                b, z, _pg = _collate(d, idx, args.device)
                _, _, v, v_ora = model.forward(b)
                lb = _loss(v, z, mode)
                lo = (_loss(v_ora, z, mode) if v_ora is not None
                      else torch.zeros((), device=v.device))
                opt.zero_grad(set_to_none=True)
                (lb + lo).backward()
                opt.step()
                tr_b += float(lb.detach())
                tr_o += float(lo.detach())
                nb += 1
        r = evaluate(model, val_shards, mode, args.device, args.minibatch)
        row = {"epoch": ep, "train_loss": round(tr_b / max(1, nb), 4),
               "train_loss_ora": round(tr_o / max(1, nb), 4),
               "sec": round(time.time() - t0, 1), **r}
        if r["loss"] < best_loss:
            best_loss = r["loss"]
            save(os.path.join(args.out, "model.pt"))
            row["saved"] = True
        log(row)
    save(os.path.join(args.out, "last.pt"))
    log_f.close()
    print(f"done; best val loss {best_loss:.4f} -> {args.out}/model.pt")


if __name__ == "__main__":
    main()
