"""Inference-time BC agent: raw obs dict -> option index list. CPU-friendly."""

from __future__ import annotations

import os

import numpy as np
import torch

from .cards import load_matching
from .features import (ARCH_F, FEAT_V6, FEAT_VERSION, MAX_COUNT, MAX_COUNT_V6,
                       N_BOARD, N_BOARD_V6, OPT_F, OPT_F_V6, LogMemory,
                       encode_obs, file_md5, own_remaining)
from .model import PolicyNet


def batch_of_one(enc: dict, n_opt: int) -> dict[str, torch.Tensor]:
    """features.encode_obs output -> model input batch (B=1, CPU tensors)."""
    b: dict[str, torch.Tensor] = {}
    for k in ("glob", "ctx", "board", "logs", "look_feats", "mem", "opt_feats"):
        b[k] = torch.from_numpy(enc[k].astype(np.float32)).unsqueeze(0)
    for k in ("ctx_ids", "board_ids", "hand_ids", "look_ids", "disc_ids",
              "deck_ids", "logs_ids", "mem_ids", "opt_card", "opt_tgt", "opt_atk"):
        b[k] = torch.from_numpy(enc[k].astype(np.int32)).unsqueeze(0)
    b["stadium_id"] = torch.tensor([int(enc["stadium_id"])], dtype=torch.int32)
    b["opt_mask"] = torch.ones(1, n_opt, dtype=torch.bool)
    if "arch" in enc:
        b["arch"] = torch.from_numpy(enc["arch"].astype(np.float32)).unsqueeze(0)
    return b


def resolve_net_cfg(cfg: dict, ckpt_path: str) -> dict:
    """Checkpoint config -> PolicyNet kwargs, tolerating older feature versions.

    v5 appended option columns (OPT_FWD) and the arch unknown bit; v6 widened
    the board to 18 tokens, the count head to 24 classes and the probe block
    to 28 columns. Older nets keep their trained widths: opt_feats are sliced
    down (columns are append-only) and the encoder is asked for the matching
    board width. Anything else is a hard mismatch."""
    ck_feat = cfg.get("feat_version")
    if ck_feat is not None and int(ck_feat) not in (4, FEAT_VERSION, FEAT_V6):
        raise ValueError(f"{ckpt_path}: trained with feature v{ck_feat}, "
                         f"but this code encodes v{FEAT_VERSION}/v{FEAT_V6}")
    v = int(ck_feat or FEAT_VERSION)
    opt_f = int(cfg.get("opt_f") or {4: 60, FEAT_V6: OPT_F_V6}.get(v, OPT_F))
    n_board = int(cfg.get("n_board")
                  or (N_BOARD_V6 if v >= FEAT_V6 else N_BOARD))
    count_classes = int(cfg.get("count_classes")
                        or (MAX_COUNT_V6 if v >= FEAT_V6 else MAX_COUNT))
    return {"d_model": cfg["d_model"], "n_layers": cfg["layers"],
            "n_heads": cfg["heads"], "arch_f": cfg.get("arch_f", 0),
            "opt_f": opt_f, "aux": bool(cfg.get("aux_head")),
            "n_board": n_board, "count_classes": count_classes}


class BCAgent:
    def __init__(self, ckpt_path: str, cards_path: str, device: str = "cpu",
                 temperature: float = 0.0, threads: int = 1,
                 prior_path: str | None = None, deck: list[int] | None = None):
        # Default 1: multiprocess eval/self-play spawn many processes; any
        # threads>1 here silently undoes the worker's set_num_threads(1) and
        # oversubscribes the box (20 procs x 2 = thrash). Single-process
        # Kaggle inference is fast enough at 1 thread for this model size.
        if device == "cpu":
            torch.set_num_threads(threads)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        cfg = ckpt["config"]
        # pick the card-table generation this checkpoint was trained on
        tables, used_path = load_matching(
            cards_path, int(ckpt["model"]["card.proj.weight"].shape[1]),
            int(ckpt["model"]["attack.proj.weight"].shape[1]))
        ck_hash = cfg.get("cards_hash")
        if ck_hash and ck_hash != file_md5(used_path):
            print(f"[agent] WARNING: cards table hash mismatch with checkpoint "
                  f"({ckpt_path}); card ids may be inconsistent", flush=True)
        self.model = PolicyNet(tables["card_feats"], tables["attack_feats"],
                               **resolve_net_cfg(cfg, ckpt_path))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.cfg = cfg
        self.value_mode = cfg.get("value_mode", "win_logit")
        self.cards = tables
        self.device = device
        self.temperature = temperature
        self.mem = LogMemory()
        self._last_turn = -1
        self._last_obs = None  # same-object guard: one mem update per decision
        self.arch_prior = None
        if self.model.arch_f > 0 and prior_path and os.path.exists(prior_path):
            from .deck_infer import DeckPrior, basics_from_cards
            self.arch_prior = DeckPrior(prior_path, basics_from_cards(tables["card_feats"]))
        # encode with the feature version the net was trained on: a v6 net
        # gets the wide board / deck-bag / delta columns, a v5 net gets the
        # bit-identical legacy encoding
        self.feat_v6 = int(cfg.get("feat_version") or FEAT_VERSION) >= FEAT_V6
        self.deck = list(deck) if deck else None
        # engine lookahead columns: only nets trained on them (opt_f > 60)
        # get a probe; older nets slice the batch back down anyway
        self.probe = None
        if self.model.opt_f > 60 and deck:
            from .fwd_features import (FWD_MAX_OPTIONS, FWD_MAX_OPTIONS_V6,
                                       ForwardProbe)
            self.probe = ForwardProbe(
                tables["card_feats"], deck=deck,
                n_cols=self.model.opt_f - 60,
                max_options=(FWD_MAX_OPTIONS_V6 if self.feat_v6
                             else FWD_MAX_OPTIONS))

    def reset(self) -> None:
        self.mem.reset()
        self._last_turn = -1
        self._last_obs = None

    def observe(self, obs_dict: dict) -> None:
        """Advance game memory. Idempotent for the same obs object, so a
        wrapper (MCTS agent) and a fallback path can both call it safely."""
        if obs_dict is self._last_obs:
            return
        turn = int((obs_dict.get("current") or {}).get("turn") or 0)
        if turn < self._last_turn:  # a new game started (agent object reused)
            self.mem.reset()
        self._last_turn = turn
        self._last_obs = obs_dict
        self.mem.update(obs_dict)

    def encode(self, obs_dict: dict) -> dict:
        """encode_obs + memory + opponent-model feature when the model wants it."""
        self.observe(obs_dict)
        fwd = self.probe.probe(obs_dict) if self.probe is not None else None
        own = (own_remaining(obs_dict, self.deck)
               if self.feat_v6 and self.deck else None)
        enc = encode_obs(obs_dict, mem=self.mem, fwd=fwd,
                         v6=self.feat_v6, own_remain=own)
        if self.model.arch_f > 0:
            if self.arch_prior is not None:
                enc["arch"] = self.arch_prior.arch_feature(
                    obs_dict, extra_ids=self.mem.op_known)
            else:
                enc["arch"] = np.zeros(ARCH_F, dtype=np.float32)
        return enc

    @torch.no_grad()
    def act(self, obs_dict: dict) -> list[int]:
        sel = obs_dict["select"]
        n_opt = len(sel["option"])
        enc = self.encode(obs_dict)
        b = {k: v.to(self.device) for k, v in batch_of_one(enc, n_opt).items()}

        logits, count_logits, _, _ = self.model(b)
        logits = logits[0]
        mn = int(sel.get("minCount") or 0)
        mx = int(sel.get("maxCount") or 0)

        if mn == 1 and mx == 1:
            if self.temperature > 0:
                p = torch.softmax(logits / self.temperature, 0)
                return [int(torch.multinomial(p, 1))]
            return [int(logits.argmax())]

        if mn == mx:
            k = mn
        else:
            cl = count_logits[0].clone()
            hi = min(mx, self.model.count_classes - 1, n_opt)
            mask = torch.full_like(cl, float("-inf"))
            mask[mn:hi + 1] = 0.0
            k = int((cl + mask).argmax())
        k = min(max(mn, min(k, mx)), n_opt)  # never exceed n_opt (topk safety)
        if k == 0:
            return []
        if self.temperature > 0:
            noisy = logits + torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits) * self.temperature
            return noisy.topk(k).indices.tolist()
        return logits.topk(k).indices.tolist()
