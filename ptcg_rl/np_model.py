"""Torch-free inference: PolicyNet forward in pure numpy + weight export.

The Kaggle simulation container has no torch (the 2026-08-09 episode replays
show the v4 bundle silently degrading to the random fallback: the agent-build
step returned within ~40ms, i.e. `import torch` raised immediately).  This
module mirrors ptcg_rl.model.PolicyNet exactly for the inference paths the
submission needs (policy logits + count head, greedy decode).

Export (run where torch IS available, e.g. the training box):

    python -m ptcg_rl.np_model --ckpt runs/ppo_v4/best.pt --out build/net.npz

Load + act (numpy only):

    agent = NumpyAgent("net.npz", "cards.npz", prior_path="deck_prior.json")
    action = agent.act(obs_dict)
"""

from __future__ import annotations

import json
import os

import numpy as np

from .cards import load_matching
from .features import (ARCH_F, FEAT_V6, FEAT_VERSION, N_MEM_ATK, OPT_F,
                       OPT_F_V6, LogMemory, encode_obs, file_md5,
                       own_remaining)

F32 = np.float32


# --------------------------------------------------------------------- export

def export_npz(ckpt_path: str, out_path: str, cards_path: str | None = None) -> dict:
    """torch checkpoint -> npz of fp32 weights + config json. Returns config."""
    import torch  # packaging-time only; never imported on Kaggle

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["config"])
    arrays = {k: v.detach().cpu().numpy().astype(F32)
              for k, v in ckpt["model"].items()}
    arrays["__config__"] = np.frombuffer(
        json.dumps(cfg).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(out_path, **arrays)
    if cards_path and cfg.get("cards_hash") and cfg["cards_hash"] != file_md5(cards_path):
        print(f"[np_model] WARNING: cards.npz hash mismatch with {ckpt_path}")
    return cfg


# -------------------------------------------------------------- numpy forward

def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _mha(q, kv, in_w, in_b, out_w, out_b, n_heads: int,
         key_padding_mask: np.ndarray | None = None) -> np.ndarray:
    """nn.MultiheadAttention (batch_first, B=1): q (Tq,d), kv (Tk,d) -> (Tq,d)."""
    d = q.shape[-1]
    dh = d // n_heads
    qp = q @ in_w[:d].T + in_b[:d]
    kp = kv @ in_w[d:2 * d].T + in_b[d:2 * d]
    vp = kv @ in_w[2 * d:].T + in_b[2 * d:]
    qh = qp.reshape(-1, n_heads, dh).transpose(1, 0, 2)          # (h,Tq,dh)
    kh = kp.reshape(-1, n_heads, dh).transpose(1, 0, 2)          # (h,Tk,dh)
    vh = vp.reshape(-1, n_heads, dh).transpose(1, 0, 2)
    sc = qh @ kh.transpose(0, 2, 1) / np.sqrt(F32(dh))           # (h,Tq,Tk)
    if key_padding_mask is not None and key_padding_mask.any():
        sc[:, :, key_padding_mask] = F32(-1e30)
    out = _softmax(sc, -1) @ vh                                  # (h,Tq,dh)
    out = out.transpose(1, 0, 2).reshape(-1, d)
    return out @ out_w.T + out_b


def _masked_mean(emb: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """emb (...,N,D), ids (...,N) -> masked mean over N (ids>0), zeros if empty."""
    m = (ids > 0).astype(F32)[..., None]
    return (emb * m).sum(-2) / np.maximum(m.sum(-2), 1.0)


class NumpyPolicy:
    """PolicyNet forward (policy + count heads) with numpy weights."""

    def __init__(self, npz_path: str, cards_path: str):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(F32) for k in z.files if k != "__config__"}
        self.cfg = json.loads(bytes(z["__config__"]).decode("utf-8"))
        ck_feat = self.cfg.get("feat_version")
        if ck_feat is not None and int(ck_feat) not in (4, FEAT_VERSION, FEAT_V6):
            raise ValueError(f"{npz_path}: trained with feature v{ck_feat}, "
                             f"but this code encodes v{FEAT_VERSION}/v{FEAT_V6}")
        self.feat_v6 = int(ck_feat or FEAT_VERSION) >= FEAT_V6
        # widths the net was trained with; newer-encoder batches get sliced
        self.opt_f = int(self.cfg.get("opt_f")
                         or {4: 60, FEAT_V6: OPT_F_V6}.get(
                             int(ck_feat or FEAT_VERSION), OPT_F))
        self.count_classes = int(self.w["count_head.weight"].shape[0])
        tables, used_path = load_matching(
            cards_path, int(self.w["card.proj.weight"].shape[1]),
            int(self.w["attack.proj.weight"].shape[1]))
        ck_hash = self.cfg.get("cards_hash")
        if ck_hash and ck_hash != file_md5(used_path):
            print(f"[np_model] WARNING: cards table hash mismatch with {npz_path}",
                  flush=True)
        cf = np.asarray(tables["card_feats"], dtype=F32)
        af = np.asarray(tables["attack_feats"], dtype=F32)
        # persistent=False buffers are rebuilt exactly as in CardRepr/AttackRepr
        self.card_static = np.concatenate([cf, np.zeros((1, cf.shape[1]), F32)], 0)
        self.atk_static = np.concatenate([af, np.zeros((1, af.shape[1]), F32)], 0)
        self.n_card = cf.shape[0]
        self.n_atk = af.shape[0]
        self.arch_f = int(self.cfg.get("arch_f", 0))
        self.n_heads = int(self.cfg["heads"])
        self.n_layers = int(self.cfg["layers"])
        self.cards = tables

    def _card(self, ids) -> np.ndarray:
        ids = np.clip(np.asarray(ids, dtype=np.int64), 0, self.n_card)
        return (self.w["card.emb.weight"][ids]
                + self.card_static[ids] @ self.w["card.proj.weight"].T
                + self.w["card.proj.bias"])

    def _atk(self, ids) -> np.ndarray:
        ids = np.clip(np.asarray(ids, dtype=np.int64), 0, self.n_atk)
        return (self.w["attack.emb.weight"][ids]
                + self.atk_static[ids] @ self.w["attack.proj.weight"].T
                + self.w["attack.proj.bias"])

    def _lin(self, name: str, x: np.ndarray) -> np.ndarray:
        return _linear(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

    def forward(self, enc: dict) -> tuple[np.ndarray, np.ndarray]:
        """features.encode_obs output (+ optional 'arch') -> (logits (K,),
        count_logits (MAX_COUNT,)). Mirrors PolicyNet.forward with B=1."""
        w = self.w
        toks = []
        types = []

        g = enc["glob"].astype(F32)
        if self.arch_f:
            g = np.concatenate([g, enc["arch"].astype(F32)[:self.arch_f]])
        toks.append(self._lin("glob_in", g)[None]); types.append([0])

        ctx_cards = self._card(enc["ctx_ids"]).reshape(-1)
        toks.append(self._lin("ctx_in", np.concatenate(
            [enc["ctx"].astype(F32), ctx_cards]))[None]); types.append([1])

        toks.append(self._lin("stadium_in",
                              self._card([int(enc["stadium_id"])])))
        types.append([2])

        nb = enc["board_ids"].shape[0]                   # 12 (v5) or 18 (v6)
        board_cards = self._card(enc["board_ids"]).reshape(nb, -1)
        toks.append(self._lin("board_in", np.concatenate(
            [enc["board"].astype(F32), board_cards], -1)))
        types.append([3] * nb)

        hand_ids = np.asarray(enc["hand_ids"])
        toks.append(self._lin("hand_in", self._card(hand_ids)))
        types.append([4] * hand_ids.shape[0])

        look_ids = np.asarray(enc["look_ids"])
        look_feats = enc["look_feats"].astype(F32)
        toks.append(self._lin("look_in", np.concatenate(
            [self._card(look_ids), look_feats], -1)))
        types.append([5] * look_ids.shape[0])

        disc_ids = np.asarray(enc["disc_ids"])                   # (2,60)
        disc_mean = _masked_mean(self._card(disc_ids), disc_ids)  # (2,D)
        # torch reuses the mean divisor (clamped to >=1) as the count feature;
        # empty piles must feed 1/60 here too or logits drift from training
        cnt = np.maximum((disc_ids > 0).sum(-1, keepdims=True), 1).astype(F32)
        toks.append(self._lin("disc_in",
                              np.concatenate([disc_mean, cnt / 60.0], -1)))
        types.append([6] * 2)

        logs_ids = np.asarray(enc["logs_ids"])
        my_atk = self._atk([logs_ids[0]])[0]
        op_atk = self._atk([logs_ids[1]])[0]
        rev_mean = _masked_mean(self._card(logs_ids[2:]), logs_ids[2:])
        toks.append(self._lin("logs_in", np.concatenate(
            [enc["logs"].astype(F32), my_atk, op_atk, rev_mean]))[None])
        types.append([7])

        na = N_MEM_ATK
        mem_ids = np.asarray(enc["mem_ids"])
        mem_my = _masked_mean(self._atk(mem_ids[:na]), mem_ids[:na])
        mem_op = _masked_mean(self._atk(mem_ids[na:2 * na]), mem_ids[na:2 * na])
        mem_rev = _masked_mean(self._card(mem_ids[2 * na:]), mem_ids[2 * na:])
        toks.append(self._lin("mem_in", np.concatenate(
            [enc["mem"].astype(F32), mem_my, mem_op, mem_rev]))[None])
        types.append([8])

        deck_ids = np.asarray(enc["deck_ids"])
        dk_cnt = np.array([(deck_ids > 0).sum()], dtype=F32)
        toks.append(self._lin("deck_in", np.concatenate(
            [_masked_mean(self._card(deck_ids), deck_ids), dk_cnt / 60.0]))[None])
        types.append([9])

        x = np.concatenate(toks, 0)                              # (T,d)
        types = np.concatenate([np.asarray(t, dtype=np.int64) for t in types])
        x = x + w["type_emb.weight"][types] + w["pos_emb.weight"][:x.shape[0]]

        pad = np.zeros(x.shape[0], dtype=bool)
        h0 = 3 + nb
        pad[h0:h0 + hand_ids.shape[0]] = hand_ids == 0
        l0 = h0 + hand_ids.shape[0]
        pad[l0:l0 + look_ids.shape[0]] = look_feats[..., 0] == 0

        for i in range(self.n_layers):
            p = f"encoder.layers.{i}"
            h = _ln(x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"])
            x = x + _mha(h, h, w[f"{p}.self_attn.in_proj_weight"],
                         w[f"{p}.self_attn.in_proj_bias"],
                         w[f"{p}.self_attn.out_proj.weight"],
                         w[f"{p}.self_attn.out_proj.bias"],
                         self.n_heads, key_padding_mask=pad)
            h = _ln(x, w[f"{p}.norm2.weight"], w[f"{p}.norm2.bias"])
            ff = self._lin(f"{p}.linear2",
                           np.maximum(self._lin(f"{p}.linear1", h), 0.0))
            x = x + ff

        # ------- options
        q = self._lin("opt_in", np.concatenate(
            [enc["opt_feats"].astype(F32)[:, :self.opt_f],
             self._card(enc["opt_card"]),
             self._card(enc["opt_tgt"]), self._atk(enc["opt_atk"])], -1))
        ca = _mha(q, x, w["opt_attn.in_proj_weight"], w["opt_attn.in_proj_bias"],
                  w["opt_attn.out_proj.weight"], w["opt_attn.out_proj.bias"],
                  self.n_heads, key_padding_mask=pad)
        opt = _ln(q + ca, w["opt_ln.weight"], w["opt_ln.bias"])   # (K,d)

        hh = x[0]
        hk = np.broadcast_to(hh, opt.shape)
        sc_in = np.concatenate([opt, hk, opt * hk], -1)
        logits = self._lin("score.2",
                           np.maximum(self._lin("score.0", sc_in), 0.0))[:, 0]
        opt_mean = opt.mean(0)                                   # opt_mask all-true
        count_logits = self._lin("count_head", np.concatenate([hh, opt_mean]))
        return logits.astype(F32), count_logits.astype(F32)


# ----------------------------------------------------------------- the agent

class NumpyAgent:
    """Greedy inference twin of ptcg_rl.agent.BCAgent (temperature 0)."""

    def __init__(self, npz_path: str, cards_path: str,
                 prior_path: str | None = None,
                 deck: list[int] | None = None):
        self.net = NumpyPolicy(npz_path, cards_path)
        self.mem = LogMemory()
        self._last_turn = -1
        self._last_obs = None
        self.arch_prior = None
        if self.net.arch_f > 0 and prior_path and os.path.exists(prior_path):
            from .deck_infer import DeckPrior, basics_from_cards
            self.arch_prior = DeckPrior(
                prior_path, basics_from_cards(self.net.cards["card_feats"]))
        # engine lookahead: nets trained on fwd columns probe each option in
        # a search sandbox; per-decision failures degrade to zero columns
        # (valid bit 0 -- a state the net saw in training), never crash
        self.deck = list(deck) if deck else None
        self.probe = None
        if self.net.opt_f > 60 and deck:
            from .fwd_features import (FWD_MAX_OPTIONS, FWD_MAX_OPTIONS_V6,
                                       ForwardProbe)
            self.probe = ForwardProbe(
                self.net.cards["card_feats"], deck=deck,
                n_cols=self.net.opt_f - 60,
                max_options=(FWD_MAX_OPTIONS_V6 if self.net.feat_v6
                             else FWD_MAX_OPTIONS))

    def reset(self) -> None:
        self.mem.reset()
        self._last_turn = -1
        self._last_obs = None

    def observe(self, obs_dict: dict) -> None:
        if obs_dict is self._last_obs:
            return
        turn = int((obs_dict.get("current") or {}).get("turn") or 0)
        if turn < self._last_turn:  # new game in a reused process
            self.mem.reset()
        self._last_turn = turn
        self._last_obs = obs_dict
        self.mem.update(obs_dict)

    def encode(self, obs_dict: dict) -> dict:
        self.observe(obs_dict)
        fwd = self.probe.probe(obs_dict) if self.probe is not None else None
        own = (own_remaining(obs_dict, self.deck)
               if self.net.feat_v6 and self.deck else None)
        enc = encode_obs(obs_dict, mem=self.mem, fwd=fwd,
                         v6=self.net.feat_v6, own_remain=own)
        if self.net.arch_f > 0:
            if self.arch_prior is not None:
                enc["arch"] = self.arch_prior.arch_feature(
                    obs_dict, extra_ids=self.mem.op_known)
            else:
                enc["arch"] = np.zeros(ARCH_F, dtype=F32)
        return enc

    def act(self, obs_dict: dict) -> list[int]:
        sel = obs_dict["select"]
        n_opt = len(sel["option"])
        logits, count_logits = self.net.forward(self.encode(obs_dict))
        mn = int(sel.get("minCount") or 0)
        mx = int(sel.get("maxCount") or 0)

        if mn == 1 and mx == 1:
            return [int(logits.argmax())]

        if mn == mx:
            k = mn
        else:
            hi = min(mx, self.net.count_classes - 1, n_opt)
            cl = count_logits.copy()
            mask = np.full_like(cl, -np.inf)
            mask[mn:hi + 1] = 0.0
            k = int((cl + mask).argmax())
        k = min(max(mn, min(k, mx)), n_opt)
        if k == 0:
            return []
        return np.argsort(-logits, kind="stable")[:k].tolist()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards", default=None)
    args = ap.parse_args()
    cfg = export_npz(args.ckpt, args.out, args.cards)
    mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"exported {args.out} ({mb:.1f} MiB) config={cfg}")


if __name__ == "__main__":
    main()
