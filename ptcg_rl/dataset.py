"""BC shard loading + torch Dataset/collate (ragged options -> padded batches)."""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import ARCH_F, FEAT_V6, FEAT_VERSION, MAX_HAND

FIXED_F32 = ("glob", "ctx", "board", "arch", "logs", "look_feats", "mem", "ora")
FIXED_INT = ("ctx_ids", "board_ids", "hand_ids", "look_ids", "disc_ids",
             "deck_ids", "logs_ids", "mem_ids", "ora_ids")
# per-sample keys that ride along for filtering / weighting (not all fed to model)
PER_SAMPLE = FIXED_F32 + FIXED_INT + (
    "stadium_id", "seq", "ctx_id", "k", "min_c", "max_c", "value",
    "side", "turn", "score", "ep_id", "aux", "aux_m", "opp_fp",
)
RAGGED = ("opt_feats", "opt_card", "opt_tgt", "opt_atk", "label")


class Shard:
    def __init__(self, path: str, day_idx: int, arr: dict | None = None):
        if arr is None:
            z = np.load(path)
            arr = {k: z[k] for k in z.files}
        self.arr = arr
        self.n = len(self.arr["k"])
        self.day_idx = day_idx
        self.path = path
        sh_feat = int(self.arr["feat_version"][0]) if "feat_version" in self.arr else -1
        if (sh_feat not in (FEAT_VERSION, FEAT_V6) or "seq" not in self.arr
                or self.arr["hand_ids"].shape[1] != MAX_HAND):
            raise ValueError(f"{path}: stale shard (v{sh_feat} vs code "
                             f"v{FEAT_VERSION}/v{FEAT_V6}, hand={MAX_HAND}); "
                             f"re-run extract_bc")
        self.feat = sh_feat
        if "arch" not in self.arr:  # pre-opponent-model shards
            self.arr["arch"] = np.zeros((self.n, ARCH_F), dtype=np.float16)
        if "aux" not in self.arr:  # pre-aux-head shards: masked out of aux loss
            self.arr["aux"] = np.zeros((self.n, 4), dtype=np.float16)
            self.arr["aux_m"] = np.zeros(self.n, dtype=np.float16)
        elif "aux_m" not in self.arr:
            self.arr["aux_m"] = np.ones(self.n, dtype=np.float16)

    def select(self, idx: np.ndarray) -> "Shard":
        """Return a view keeping only the listed sample indices (episode split)."""
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("empty shard select")
        a = self.arr
        out: dict = {}
        for k in PER_SAMPLE:
            if k in a and len(a[k]) == self.n:
                out[k] = a[k][idx]
        offs = a["opt_offsets"]
        pieces = {k: [] for k in RAGGED}
        new_offs = [0]
        for i in idx:
            o0, o1 = int(offs[i]), int(offs[i + 1])
            for k in RAGGED:
                pieces[k].append(a[k][o0:o1])
            new_offs.append(new_offs[-1] + (o1 - o0))
        for k in RAGGED:
            out[k] = np.concatenate(pieces[k], axis=0)
        out["opt_offsets"] = np.asarray(new_offs, dtype=np.int64)
        for k in ("feat_version", "cards_hash"):
            if k in a:
                out[k] = a[k]
        return Shard(self.path, self.day_idx, arr=out)


class BCDataset(Dataset):
    def __init__(self, shards: list[Shard], loser_w: float = 0.3, draw_w: float = 0.5,
                 recency_tau: float = 10.0, score_beta: float = 0.0,
                 uniform: bool = False):
        """uniform=True: every weight is exactly 1.0 (validation metrics must
        not be skewed by loser/draw/recency/score reweighting)."""
        self.shards = shards
        self.index: list[tuple[int, int]] = [
            (si, i) for si, sh in enumerate(shards) for i in range(sh.n)
        ]
        max_day = max(sh.day_idx for sh in shards) if shards else 0
        # imitate stronger play: upweight samples from higher-rated episodes
        sc = np.concatenate([sh.arr["score"] for sh in shards]) if score_beta > 0 else None
        if sc is not None:
            known = sc[sc > 0]
            mu = float(known.mean()) if len(known) else 0.0
            sd = float(known.std()) + 1e-6
        self.weights = np.empty(len(self.index), dtype=np.float32)
        if uniform:
            self.weights.fill(1.0)
            return
        pos = 0
        for sh in shards:
            v = sh.arr["value"]
            w = np.where(v >= 0.99, 1.0, np.where(v <= 0.01, loser_w, draw_w)).astype(np.float32)
            w[v < 0] = 0.25  # unknown outcome
            if recency_tau > 0:
                w *= np.exp((sh.day_idx - max_day) / recency_tau).astype(np.float32)
            if sc is not None:
                z = np.where(sh.arr["score"] > 0, (sh.arr["score"] - mu) / sd, 0.0)
                w *= np.clip(np.exp(score_beta * z), 0.5, 2.0).astype(np.float32)
            self.weights[pos:pos + sh.n] = w
            pos += sh.n

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        si, j = self.index[i]
        a = self.shards[si].arr
        o0, o1 = a["opt_offsets"][j], a["opt_offsets"][j + 1]
        s = {k: a[k][j] for k in FIXED_F32 + FIXED_INT + (
            "stadium_id", "seq", "ctx_id", "k", "min_c", "max_c", "value",
            "aux", "aux_m")}
        s["opt_feats"] = a["opt_feats"][o0:o1]
        s["opt_card"] = a["opt_card"][o0:o1]
        s["opt_tgt"] = a["opt_tgt"][o0:o1]
        s["opt_atk"] = a["opt_atk"][o0:o1]
        s["label"] = a["label"][o0:o1]
        s["w"] = self.weights[i]
        return s


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    B = len(samples)
    K = max(len(s["label"]) for s in samples)
    out: dict[str, torch.Tensor] = {}
    for k in FIXED_F32:
        out[k] = torch.from_numpy(np.stack([s[k] for s in samples])).float()
    for k in FIXED_INT + ("stadium_id",):
        out[k] = torch.from_numpy(np.stack([s[k] for s in samples]).astype(np.int32))
    opt_feats = np.zeros((B, K, samples[0]["opt_feats"].shape[1]), dtype=np.float32)
    opt_card = np.zeros((B, K), dtype=np.int32)
    opt_tgt = np.zeros((B, K), dtype=np.int32)
    opt_atk = np.zeros((B, K), dtype=np.int32)
    mask = np.zeros((B, K), dtype=bool)
    label = np.zeros((B, K), dtype=np.float32)
    for i, s in enumerate(samples):
        n = len(s["label"])
        opt_feats[i, :n] = s["opt_feats"]
        opt_card[i, :n] = s["opt_card"]
        opt_tgt[i, :n] = s["opt_tgt"]
        opt_atk[i, :n] = s["opt_atk"]
        mask[i, :n] = True
        label[i, :n] = s["label"]
    out["opt_feats"] = torch.from_numpy(opt_feats)
    out["opt_card"] = torch.from_numpy(opt_card)
    out["opt_tgt"] = torch.from_numpy(opt_tgt)
    out["opt_atk"] = torch.from_numpy(opt_atk)
    out["opt_mask"] = torch.from_numpy(mask)
    out["label"] = torch.from_numpy(label)
    for k in ("k", "min_c", "max_c", "ctx_id"):
        out[k] = torch.tensor([int(s[k]) for s in samples], dtype=torch.long)
    out["seq"] = torch.from_numpy(np.stack([s["seq"] for s in samples]).astype(np.int64))
    out["value"] = torch.tensor([float(s["value"]) for s in samples], dtype=torch.float32)
    out["w"] = torch.tensor([float(s["w"]) for s in samples], dtype=torch.float32)
    out["aux"] = torch.from_numpy(np.stack([s["aux"] for s in samples])).float()
    out["aux_m"] = torch.tensor([float(s["aux_m"]) for s in samples], dtype=torch.float32)
    return out


def _episode_ids(sh: Shard) -> np.ndarray:
    """Per-sample episode ids; synthesise 1-sample eps if shard lacks ep_id."""
    if "ep_id" in sh.arr and len(sh.arr["ep_id"]) == sh.n:
        return sh.arr["ep_id"].astype(np.int64)
    return np.arange(sh.n, dtype=np.int64)


def load_shards(pattern: str, val_days: int = 1, val_frac: float = 0.0,
                seed: int = 0, unseen_opp: bool = False
                ) -> tuple[list[Shard], list[Shard]]:
    """Split into train/val.

    Prefer ``val_frac`` (random episode holdout across all days) when > 0;
    otherwise hold out the newest ``val_days`` day-shards (legacy).
    With ``unseen_opp`` the holdout is grouped by opponent decklist
    fingerprint: every val episode faces a list absent from training, so the
    val metrics measure generalization to unseen opponent decks (OOD slice).
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    days = []
    for p in paths:
        m = re.match(r"(\d+)_", os.path.basename(p))
        days.append(m.group(1) if m else os.path.basename(p))
    order = {d: i for i, d in enumerate(sorted(set(days)))}
    shards = [Shard(p, order[d]) for p, d in zip(paths, days)]

    if val_frac > 0:
        rng = np.random.default_rng(seed)
        # global episode keys: (shard_i, local_ep_id)
        ep_keys: list[tuple[int, int]] = []
        ep_fp: dict[tuple[int, int], int] = {}
        for si, sh in enumerate(shards):
            eids = _episode_ids(sh)
            fps = sh.arr.get("opp_fp")
            for eid in np.unique(eids):
                key = (si, int(eid))
                ep_keys.append(key)
                if fps is not None and len(fps) == sh.n:
                    ep_fp[key] = int(fps[np.argmax(eids == eid)])
        n_val = max(1, int(round(len(ep_keys) * val_frac))) if ep_keys else 0
        if unseen_opp and len(ep_fp) == len(ep_keys):
            # hold out whole opponent decklists: val measures play against
            # lists never seen in training
            groups: dict[int, list[tuple[int, int]]] = {}
            for key in ep_keys:
                groups.setdefault(ep_fp[key], []).append(key)
            fp_order = sorted(groups)
            rng.shuffle(fp_order)
            val_keys: set = set()
            n_fp = 0
            for fp in fp_order:
                if len(val_keys) >= n_val:
                    break
                val_keys.update(groups[fp])
                n_fp += 1
            print(f"unseen-opp val: {n_fp} opponent lists held out "
                  f"({len(val_keys)} episodes)")
        else:
            if unseen_opp:
                print("[warn] shards lack opp_fp; falling back to random "
                      "episode split (re-run extract_bc for the OOD slice)")
            rng.shuffle(ep_keys)
            val_keys = set(ep_keys[:n_val])
        train, val = [], []
        for si, sh in enumerate(shards):
            eids = _episode_ids(sh)
            is_val = np.array([(si, int(e)) in val_keys for e in eids], dtype=bool)
            tr_idx = np.flatnonzero(~is_val)
            va_idx = np.flatnonzero(is_val)
            if len(tr_idx):
                train.append(sh.select(tr_idx))
            if len(va_idx):
                val.append(sh.select(va_idx))
        has_ep = all("ep_id" in sh.arr for sh in shards)
        unit = "episodes" if has_ep else "samples (no ep_id; re-extract for true games)"
        print(f"val_frac={val_frac}: {len(ep_keys)} {unit}, "
              f"val={n_val} ({n_val / max(1, len(ep_keys)):.1%})")
        if not train:
            train = val
        if not val:
            raise RuntimeError("val_frac split produced empty val set")
        return train, val

    val_set = set(sorted(set(days))[-val_days:]) if val_days > 0 else set()
    train = [sh for sh, d in zip(shards, days) if d not in val_set]
    val = [sh for sh, d in zip(shards, days) if d in val_set]
    if not train:  # single-day smoke: reuse it for both
        train = val
    return train, val
