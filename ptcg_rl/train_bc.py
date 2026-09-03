"""Behavioral-cloning training.

Loss = CE (single-select) + BCE (unordered multi-select) + sequential NLL
(order-sensitive multi-select: SKILL_ORDER / TO_DECK_BOTTOM) + 0.2*count-CE
+ 0.5*value-BCE, per-sample weighted (winner 1.0 / loser 0.3 / recency decay).

Training box (RTX 5090):
    python -m ptcg_rl.train_bc --data "data/bc/*_220eddd2.npz" --epochs 4 --bs 1024 --amp
Smoke (CPU):
    python -m ptcg_rl.train_bc --data "data/bc_smoke/*.npz" --epochs 1 --bs 64 --steps 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cards import load as load_cards
from .dataset import BCDataset, collate, load_shards
from .features import (ARCH_F, ARCH_UNK, FEAT_V6, FEAT_VERSION, MAX_COUNT,
                       MAX_COUNT_V6, N_BOARD, N_BOARD_V6, OPT_F, OPT_F_V6,
                       ORDERED_CTX, file_md5)
from .model import PolicyNet


def ordered_nll(lg: torch.Tensor, sq: torch.Tensor, step_mask: torch.Tensor,
                kk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stepwise CE without replacement along the expert sequence.

    lg (B,K) logits; sq (B,S) expert option order padded with -1;
    step_mask (B,K) legal options (consumed as steps are taken); kk (B,)
    sequence lengths. -> (mean NLL per step (B,), greedy hits (B,)).
    Order-sensitive by construction: [2,0,1] and [0,1,2] score differently.
    """
    K = lg.shape[1]
    step_mask = step_mask.clone()
    nll = torch.zeros(lg.shape[0], device=lg.device)
    hits = torch.zeros(lg.shape[0], device=lg.device)
    for s in range(int(kk.max())):
        act = sq[:, s]
        active = act >= 0
        if not active.any():
            break
        ls = torch.log_softmax(
            lg.masked_fill(~step_mask, torch.finfo(lg.dtype).min), dim=1)
        a = act.clamp(min=0)
        nll = nll - torch.where(active, ls.gather(1, a.unsqueeze(1)).squeeze(1),
                                torch.zeros_like(nll))
        hits = hits + torch.where(
            active, (ls.argmax(1) == a).float(), torch.zeros_like(hits))
        step_mask = step_mask & ~(F.one_hot(a, K).bool() & active.unsqueeze(1))
    kf = kk.clamp(min=1).float()
    return nll / kf, hits / kf


def compute_loss(model, b, device, value_scale: float = 1.0):
    logits, count_logits, value, v_ora = model(b)
    single = (b["min_c"] == 1) & (b["max_c"] == 1)
    w = b["w"]
    losses = {}

    if single.any():
        lg = logits[single]
        tgt = b["label"][single].argmax(1)
        ce = F.cross_entropy(lg, tgt, reduction="none")
        losses["pol_s"] = (ce * w[single]).sum() / w[single].sum().clamp(min=1e-6)
        losses["acc_s"] = ((lg.argmax(1) == tgt).float() * w[single]).sum() / w[single].sum().clamp(min=1e-6)
    multi = ~single
    # order-sensitive contexts: the expert's selection ORDER is the label
    ordered = multi & (b["k"] > 1) & torch.isin(
        b["ctx_id"], torch.tensor(ORDERED_CTX, device=b["ctx_id"].device))
    multi_u = multi & ~ordered
    if multi_u.any():
        lg = logits[multi_u]
        lab = b["label"][multi_u]
        msk = b["opt_mask"][multi_u]
        bce = F.binary_cross_entropy_with_logits(lg.masked_fill(~msk, 0.0), lab, reduction="none")
        bce = (bce * msk).sum(1) / msk.sum(1).clamp(min=1)
        losses["pol_m"] = (bce * w[multi_u]).sum() / w[multi_u].sum().clamp(min=1e-6)
        # exact-set accuracy with true k
        kk = b["k"][multi_u]
        topk = torch.zeros_like(lab, dtype=torch.bool)
        for i in range(lg.shape[0]):
            if kk[i] > 0:
                topk[i, lg[i].topk(int(kk[i])).indices] = True
        exact = (topk == lab.bool()).all(1).float()
        losses["acc_m"] = (exact * w[multi_u]).sum() / w[multi_u].sum().clamp(min=1e-6)
    if ordered.any():
        kk = b["k"][ordered].clamp(max=b["seq"].shape[1])
        nll, acc = ordered_nll(logits[ordered], b["seq"][ordered],
                               b["opt_mask"][ordered], kk)
        wo = w[ordered]
        losses["pol_o"] = (nll * wo).sum() / wo.sum().clamp(min=1e-6)
        losses["acc_o"] = (acc * wo).sum() / wo.sum().clamp(min=1e-6)

    # count head is only consulted at inference when min != max; forced-k rows
    # (e.g. "discard exactly 21") would otherwise inject clamped garbage labels
    cmask = b["min_c"] != b["max_c"]
    if cmask.any():
        cnt_tgt = b["k"][cmask].clamp(max=count_logits.shape[1] - 1)
        cnt_ce = F.cross_entropy(count_logits[cmask], cnt_tgt, reduction="none")
        wc = w[cmask]
        losses["count"] = (cnt_ce * wc).sum() / wc.sum().clamp(min=1e-6)
        losses["acc_c"] = ((count_logits[cmask].argmax(1) == cnt_tgt).float() * wc).sum() \
            / wc.sum().clamp(min=1e-6)

    vmask = b["value"] >= 0
    if vmask.any():
        wv = w[vmask]
        vb = F.binary_cross_entropy_with_logits(value[vmask], b["value"][vmask],
                                                reduction="none")
        losses["value"] = (vb * wv).sum() / wv.sum().clamp(min=1e-6)
        losses["acc_v"] = (((value[vmask] > 0) == (b["value"][vmask] > 0.5)).float()).mean()
        if v_ora is not None:  # asymmetric critic warm start (train-only head)
            vo = F.binary_cross_entropy_with_logits(
                v_ora[vmask], b["value"][vmask], reduction="none")
            losses["value_o"] = (vo * wv).sum() / wv.sum().clamp(min=1e-6)

    # aux head (train only): prizes taken by [me, opp] within next 2/4 turns
    if getattr(model, "aux_head", None) is not None and "aux" in b:
        am = b["aux_m"] * w
        if am.sum() > 0:
            se = (model._aux - b["aux"]).pow(2).mean(1)
            losses["aux"] = (se * am).sum() / am.sum().clamp(min=1e-6)

    # value_scale is the anti-memorization brake: the value loss backprops
    # through the shared encoder, and once the val value loss starts rising
    # (head memorizing episode outcomes) its gradients only pollute the
    # policy trunk -- the brake scales them down without touching policy
    total = (losses.get("pol_s", 0.0) + losses.get("pol_m", 0.0)
             + losses.get("pol_o", 0.0)
             + 0.2 * losses.get("count", 0.0)
             + value_scale * 0.5 * losses.get("value", 0.0)
             + value_scale * 0.5 * losses.get("value_o", 0.0)
             + value_scale * getattr(model, "aux_w", 0.25) * losses.get("aux", 0.0))
    return total, {k: float(v) for k, v in losses.items()}


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 0):
    model.eval()
    agg: dict[str, list] = {}
    for bi, b in enumerate(loader):
        b = {k: v.to(device) for k, v in b.items()}
        _, m = compute_loss(model, b, device)
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
        if max_batches and bi + 1 >= max_batches:
            break
    model.train()
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of BC npz shards")
    ap.add_argument("--val-days", type=int, default=1,
                    help="legacy: hold out newest N day-shards (ignored if --val-frac>0)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="random episode holdout across all days (default: "
                         "10%% of games); set 0 to fall back to holding out "
                         "the newest --val-days day-shards")
    ap.add_argument("--val-unseen-opp", action="store_true",
                    help="group the --val-frac holdout by opponent decklist "
                         "fingerprint: val opponents never appear in training "
                         "(OOD generalization slice)")
    ap.add_argument("--epochs", type=int, default=4, help="max epochs")
    ap.add_argument("--patience", type=int, default=5,
                    help="early stop after N epochs without a new best val "
                         "score (0 = train all epochs)")
    ap.add_argument("--value-brake", type=float, default=1.5,
                    help="halve the value/aux loss weight whenever the val "
                         "value loss exceeds its best-so-far by this factor, "
                         "so a memorizing value head stops polluting the "
                         "shared encoder (0 = off)")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loser-w", type=float, default=0.3)
    ap.add_argument("--recency-tau", type=float, default=10.0)
    ap.add_argument("--score-beta", type=float, default=0.3,
                    help="upweight samples from higher-rated episodes (0 = off)")
    ap.add_argument("--cards", default="data/cards.npz")
    ap.add_argument("--out", default="runs/bc_v1")
    ap.add_argument("--arch", action="store_true",
                    help="enable opponent-model features (needs shards with arch)")
    ap.add_argument("--arch-dropout", type=float, default=0.15,
                    help="P(zero the arch posterior + set the unknown bit) per "
                         "sample: the policy must stay competent when the "
                         "opponent deck cannot be identified (OOD lists)")
    ap.add_argument("--id-dropout", type=float, default=0.1,
                    help="P(zero a card/attack id embedding, keeping static "
                         "attributes) per slot: rare/unseen ids have junk "
                         "embeddings at play time")
    ap.add_argument("--aux-w", type=float, default=0.25,
                    help="aux head loss weight: predict prizes taken by each "
                         "side within 2/4 turns (train-only head, 0 = off)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0, help="cap steps/epoch (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="resume from last.pt")
    ap.add_argument("--init", default=None,
                    help="hot-start weights from this checkpoint (fine-tuning "
                         "a shared base); net dims are taken from it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_sh, val_sh = load_shards(args.data, val_days=args.val_days,
                                   val_frac=args.val_frac, seed=args.seed,
                                   unseen_opp=args.val_unseen_opp)
    train_ds = BCDataset(train_sh, args.loser_w, recency_tau=args.recency_tau,
                         score_beta=args.score_beta)
    val_ds = BCDataset(val_sh, uniform=True)  # metrics must stay unweighted
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    # feature version comes from the shards (extract_bc --v6 stamps 6)
    feats = {sh.feat for sh in train_sh + val_sh}
    assert len(feats) == 1, f"mixed feature versions in shards: {feats}"
    feat = feats.pop()
    v6 = feat >= FEAT_V6
    opt_f = OPT_F_V6 if v6 else OPT_F
    n_board = N_BOARD_V6 if v6 else N_BOARD
    count_classes = MAX_COUNT_V6 if v6 else MAX_COUNT

    arch_f = ARCH_F if args.arch else 0
    if args.arch and not any(float(np.abs(sh.arr["arch"]).sum()) > 0 for sh in train_sh):
        print("[warn] --arch set but all arch features are zeros (re-extract with --prior)")
    tables = load_cards(args.cards)
    if args.init:  # fine-tune: net dims must match the base checkpoint
        from .agent import resolve_net_cfg
        ick = torch.load(args.init, map_location="cpu", weights_only=True)
        icfg = resolve_net_cfg(ick["config"], args.init)
        assert icfg["opt_f"] == opt_f and icfg["n_board"] == n_board, (
            f"--init was trained on feature widths {icfg}, shards are v{feat}")
        args.d_model, args.layers, args.heads = (
            icfg["d_model"], icfg["n_layers"], icfg["n_heads"])
        arch_f = icfg["arch_f"]
        print(f"init from {args.init}: d_model={args.d_model} "
              f"layers={args.layers} heads={args.heads} arch_f={arch_f}")
    model = PolicyNet(tables["card_feats"], tables["attack_feats"],
                      d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                      dropout=args.dropout, arch_f=arch_f,
                      aux=args.aux_w > 0, opt_f=opt_f, n_board=n_board,
                      count_classes=count_classes).to(args.device)
    if args.init:
        missing, unexpected = model.load_state_dict(ick["model"], strict=False)
        # aux head may be freshly added on top of a base without one
        bad = [k for k in missing if not k.startswith("aux_head")]
        assert not bad and not unexpected, (bad, unexpected)
        if missing:
            print(f"init: fresh params {missing}")
    model.card.id_dropout = args.id_dropout
    model.attack.id_dropout = args.id_dropout
    model.aux_w = args.aux_w
    model.detach_oracle = True  # BC: oracle loss must not shape the encoder
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M  device: {args.device}")

    dl_kw = dict(batch_size=args.bs, collate_fn=collate, num_workers=args.num_workers,
                 pin_memory=(args.device == "cuda"))
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    steps_per_epoch = args.steps or len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # patience mode: cosine over --epochs would never anneal when we stop at
    # epoch ~15/70, so use warmup -> constant with plateau halving instead
    # (lr_scale is halved on every epoch without a new best val score)
    lr_state = {"scale": 1.0}

    def lr_at(step):
        if step < args.warmup:
            return lr_state["scale"] * step / max(1, args.warmup)
        if args.patience:
            return lr_state["scale"]
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    net_cfg = {"d_model": args.d_model, "layers": args.layers,
               "heads": args.heads, "arch_f": arch_f, "opt_f": opt_f,
               "feat_version": feat, "value_mode": "win_logit",
               "n_board": n_board, "count_classes": count_classes,
               "aux_head": args.aux_w > 0, "init_from": args.init or "",
               "cards_hash": file_md5(args.cards) if os.path.exists(args.cards) else "",
               "seed": args.seed,
               # data manifest: what this checkpoint was trained on
               "data_glob": args.data, "val_days": args.val_days,
               "val_frac": args.val_frac, "val_unseen_opp": args.val_unseen_opp,
               "n_train": len(train_ds), "n_val": len(val_ds),
               "n_shards": len(train_sh) + len(val_sh)}
    best = -1.0
    bad = 0  # epochs since the last val-score improvement
    v_best = float("inf")   # best val value loss (brake reference)
    v_scale = 1.0           # current value/aux weight scale
    step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rk = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(rk["model"])
        if "opt" in rk:
            opt.load_state_dict(rk["opt"])
            sched.load_state_dict(rk["sched"])
            scaler.load_state_dict(rk["scaler"])
            step = int(rk.get("step", 0))
            start_epoch = int(rk.get("epoch", -1)) + 1
            best = float(rk.get("best", -1.0))
            bad = int(rk.get("bad", 0))
            lr_state["scale"] = float(rk.get("lr_scale", 1.0))
            v_best = float(rk.get("v_best", float("inf")))
            v_scale = float(rk.get("v_scale", 1.0))
        print(f"resumed from {args.resume}: epoch {start_epoch}, step {step}")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        for b in train_dl:
            b = {k: v.to(args.device, non_blocking=True) for k, v in b.items()}
            if arch_f > 0 and args.arch_dropout > 0:
                # opponent-identity dropout: sometimes the deck cannot be
                # matched at play time; train on exactly that input state
                drop = (torch.rand(b["arch"].shape[0], device=b["arch"].device)
                        < args.arch_dropout)
                b["arch"][drop] = 0.0
                if b["arch"].shape[1] > ARCH_UNK:
                    b["arch"][drop, ARCH_UNK] = 1.0
            with torch.amp.autocast(args.device.split(":")[0], enabled=args.amp):
                loss, metrics = compute_loss(model, b, args.device,
                                             value_scale=v_scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0 or step == 1:
                sps = step * args.bs / (time.time() - t0)
                line = {"step": step, "epoch": epoch, "loss": float(loss),
                        "sps": round(sps), **{k: round(v, 4) for k, v in metrics.items()}}
                print(line, flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(line) + "\n")
            if args.steps and step >= args.steps * (epoch + 1):
                break
        ev = evaluate(model, val_dl, args.device, max_batches=(20 if args.steps else 0))
        print(f"[val e{epoch}] {ev}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"val": ev, "epoch": epoch}) + "\n")
        # value brake: selection stays policy-driven, but once the val value
        # loss blows past its best the head is memorizing episode outcomes
        # and its trunk gradients only hurt -- scale value/aux weights down
        vl = ev.get("value")
        if args.value_brake and vl is not None:
            v_best = min(v_best, float(vl))
            if float(vl) > args.value_brake * v_best and v_scale > 0.05:
                v_scale *= 0.5
                print(f"value brake: val value {float(vl):.3f} > "
                      f"{args.value_brake:g}x best {v_best:.3f} -> "
                      f"value/aux weight scale {v_scale:.3f}", flush=True)
        # composite offline score; final model choice should still come from
        # a fixed-seeds gauntlet, this only pre-selects checkpoints
        score = ev.get("acc_s", 0.0) + 0.5 * ev.get("acc_m", 0.0) \
            + 0.25 * ev.get("acc_o", 0.0)
        if score > best:
            best = score
            bad = 0
            torch.save({"model": model.state_dict(), "config": net_cfg},
                       os.path.join(args.out, "model.pt"))
            print(f"saved best (score={score:.4f}, acc_s={ev.get('acc_s', 0.0):.4f})",
                  flush=True)
        else:
            bad += 1
            if args.patience:
                lr_state["scale"] *= 0.5
                print(f"no improvement ({bad}/{args.patience}): "
                      f"lr scale -> {lr_state['scale']:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "config": net_cfg,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "step": step, "best": best, "bad": bad,
                    "lr_scale": lr_state["scale"],
                    "v_best": v_best, "v_scale": v_scale},
                   os.path.join(args.out, "last.pt"))
        if args.patience and bad >= args.patience:
            print(f"early stop at epoch {epoch}: no val improvement for "
                  f"{bad} epochs (best={best:.4f} stays in model.pt)", flush=True)
            break
    print(f"done in {(time.time()-t0)/60:.1f} min; best val score={best:.4f}")


if __name__ == "__main__":
    main()
