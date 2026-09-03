"""Extract a behavioral-cloning dataset from daily replay zips.

For every episode side that plays the target deck (exact fingerprint or
archetype), emit one sample per decision point of that side:
  state encoding (features.encode_obs) + expert option set + outcome.

Options are stored ragged (concatenated + offsets) to keep shards compact.

Usage (training box, ~minutes with 20 workers):
    python -m ptcg_rl.extract_bc --fp 220eddd2
    python -m ptcg_rl.extract_bc --arch mega_lopunny --out-dir data/bc_arch
Smoke:
    python -m ptcg_rl.extract_bc --fp 220eddd2 --days 0807 --limit 120
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import orjson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
# engine (cg) for the forward-feature probe in worker processes
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))

from replay_loader import classify_deck, deck_fingerprint, load_manifest  # noqa: E402
from ptcg_rl.features import (ARCH_F, FEAT_V6, FEAT_VERSION, SEQ_PAD,  # noqa: E402
                              LogMemory, encode_obs, file_md5, oracle_feats,
                              own_remaining)

DEFAULT_WORKERS = 20

_ZF: zipfile.ZipFile | None = None
_FP: str | None = None
_ARCH: str | None = None
_ALL = False   # extract every side of every episode (shared-base BC)
_V6 = False    # v6 feature encoding (wide board, deck-bag, 28 probe cols)
_PRIOR = None
_PROBE = None  # engine lookahead probe (None = disabled)

FIXED_KEYS = ("glob", "ctx", "ctx_ids", "board", "board_ids", "hand_ids",
              "look_ids", "look_feats", "disc_ids", "deck_ids",
              "logs", "logs_ids", "mem", "mem_ids")
RAGGED_KEYS = ("opt_feats", "opt_card", "opt_tgt", "opt_atk")
F16_KEYS = {"glob", "ctx", "board", "logs", "look_feats", "mem"}


def _init_worker(zip_path: str, fp: str | None, arch: str | None,
                 prior_path: str | None, cards_path: str,
                 forward: bool = True, all_sides: bool = False,
                 v6: bool = False) -> None:
    global _ZF, _FP, _ARCH, _ALL, _V6, _PRIOR, _PROBE
    _ZF = zipfile.ZipFile(zip_path)
    _FP, _ARCH, _ALL, _V6 = fp, arch, all_sides, v6
    tables = None
    if prior_path or forward:
        from ptcg_rl.cards import load as load_cards
        tables = load_cards(cards_path)
    if prior_path:
        from ptcg_rl.deck_infer import DeckPrior, basics_from_cards
        _PRIOR = DeckPrior(prior_path, basics_from_cards(tables["card_feats"]))
    if forward:
        from ptcg_rl.features import OPT_FWD, OPT_FWD_V6
        from ptcg_rl.fwd_features import (FWD_MAX_OPTIONS, FWD_MAX_OPTIONS_V6,
                                          ForwardProbe)
        _PROBE = ForwardProbe(
            tables["card_feats"],
            n_cols=OPT_FWD_V6 if v6 else OPT_FWD,
            max_options=FWD_MAX_OPTIONS_V6 if v6 else FWD_MAX_OPTIONS)
        if not _PROBE._ensure_io():  # fail loud: silent all-zero shards are worse
            raise RuntimeError("forward features requested but the cg engine "
                               "failed to load; pass --no-forward-features to skip")


def _deck_matches(deck: list[int]) -> bool:
    if _ALL:
        return True
    if _FP is not None:
        return deck_fingerprint(deck) == _FP
    return classify_deck(deck) == _ARCH


def _episode_samples(name: str):
    """-> (ep_id, per-key list-of-arrays dict, n_samples) or None."""
    d = orjson.loads(_ZF.read(name))
    steps = d.get("steps") or []
    if not steps:
        return None
    try:
        decks = steps[0][0]["visualize"][0]["action"]
    except (KeyError, IndexError, TypeError):
        return None
    if not decks or len(decks[0]) != 60 or len(decks[1]) != 60:
        return None
    sides = [i for i in (0, 1) if _deck_matches([int(x) for x in decks[i]])]
    if not sides:
        return None
    rewards = d.get("rewards") or [None, None]

    cols: dict[str, list] = {k: [] for k in FIXED_KEYS + RAGGED_KEYS}
    cols.update({"stadium_id": [], "label": [], "k": [], "min_c": [], "max_c": [],
                 "value": [], "side": [], "turn": [], "arch": [],
                 "ora": [], "ora_ids": [], "seq": [], "ctx_id": [], "aux": [],
                 "opp_fp": []})
    # opponent decklist fingerprint per side: lets training hold out entire
    # opponent lists as an unseen-decklist validation slice
    opp_fp = {i: int(deck_fingerprint([int(x) for x in decks[1 - i]]), 16)
              for i in sides}
    # prize timeline for the aux head: (turn, prizes_left_p0, prizes_left_p1),
    # chronological; prize counts only ever decrease, so "prizes taken by side s
    # within H turns" = prizes_left(now) - prizes_left(end of turn now+H)
    tl_turn: list[int] = []
    tl_p: tuple[list[int], list[int]] = ([], [])
    for t in range(len(steps)):
        for j in (0, 1):
            stj = steps[t][j] or {}
            cj = (stj.get("observation") or {}).get("current")
            if stj.get("status") != "ACTIVE" or not cj:
                continue
            pls = cj.get("players") or [{}, {}]
            tl_turn.append(int(cj.get("turn") or 0))
            tl_p[0].append(len(pls[0].get("prize") or []))
            tl_p[1].append(len(pls[1].get("prize") or []))
            break  # one snapshot per step suffices
    tl_turns = np.asarray(tl_turn, dtype=np.int32)
    tl_prizes = (np.asarray(tl_p[0], dtype=np.int8), np.asarray(tl_p[1], dtype=np.int8))
    n = 0
    mems = {i: LogMemory() for i in sides}
    hand_snap: dict[int, list[int]] = {0: [], 1: []}  # each side's hand at their
    for t in range(len(steps) - 1):                   # last decision (oracle)
        for j in (0, 1):
            stj = steps[t][j]
            if stj.get("status") != "ACTIVE":
                continue
            oj = stj.get("observation") or {}
            cj = oj.get("current")
            if not cj:
                continue
            hand = (cj.get("players") or [{}, {}])[j].get("hand")
            if hand is not None:
                hand_snap[j] = [int(c.get("id") or 0) for c in hand if c]
            if j in mems:
                mems[j].update(oj)  # windows are per-side disjoint & complete
        for i in sides:
            st = steps[t][i]
            if st.get("status") != "ACTIVE":
                continue
            obs = st.get("observation") or {}
            sel = obs.get("select")
            cur = obs.get("current")
            if not sel or not cur:
                continue
            act = steps[t + 1][i].get("action")
            n_opt = len(sel.get("option") or [])
            if act is None or n_opt == 0:
                continue
            if not all(isinstance(a, int) and 0 <= a < n_opt for a in act):
                continue
            try:
                fwd = None
                deck_i = [int(x) for x in decks[i]]
                if _PROBE is not None:
                    _PROBE.set_deck(deck_i)
                    fwd = _PROBE.probe(obs)
                own = own_remaining(obs, deck_i) if _V6 else None
                enc = encode_obs(obs, mem=mems[i], fwd=fwd, v6=_V6,
                                 own_remain=own)
                if _PRIOR is not None:
                    arch_vec = _PRIOR.arch_feature(obs, extra_ids=mems[i].op_known)
                else:
                    arch_vec = np.zeros(ARCH_F, dtype=np.float32)
                opp_hc = int((cur.get("players") or [{}, {}])[1 - i].get("handCount") or 0)
                ora, ora_ids = oracle_feats(hand_snap[1 - i], opp_hc)
            except Exception:  # noqa: BLE001  malformed obs: skip decision
                continue
            for key in FIXED_KEYS + RAGGED_KEYS:
                cols[key].append(enc[key])
            cols["arch"].append(arch_vec)
            cols["ora"].append(ora)
            cols["ora_ids"].append(ora_ids)
            cols["stadium_id"].append(enc["stadium_id"])
            lab = np.zeros(n_opt, dtype=bool)
            lab[list(act)] = True
            cols["label"].append(lab)
            sq = np.full(SEQ_PAD, -1, dtype=np.int16)  # expert order preserved
            sq[:min(len(act), SEQ_PAD)] = act[:SEQ_PAD]
            cols["seq"].append(sq)
            cols["ctx_id"].append(int(sel.get("context") or 0))
            cols["k"].append(len(act))
            cols["min_c"].append(int(sel.get("minCount") or 0))
            cols["max_c"].append(int(sel.get("maxCount") or 0))
            rw = rewards
            if rw[i] is not None and rw[1 - i] is not None:
                v = 1.0 if rw[i] > rw[1 - i] else (0.0 if rw[i] < rw[1 - i] else 0.5)
            else:
                v = -1.0  # unknown -> masked in training
            cols["value"].append(v)
            cols["side"].append(i)
            cols["turn"].append(min(int(cur.get("turn") or 0), 127))
            # aux labels: prizes taken by [me, opp] within the next 2 / 4 turns
            aux = np.zeros(4, dtype=np.float32)
            if len(tl_turns):
                pls_now = cur.get("players") or [{}, {}]
                T = int(cur.get("turn") or 0)
                now = (len(pls_now[i].get("prize") or []),
                       len(pls_now[1 - i].get("prize") or []))
                for hi, H in enumerate((2, 4)):
                    idx = max(0, int(np.searchsorted(tl_turns, T + H, side="right")) - 1)
                    aux[2 * hi] = max(0, now[0] - int(tl_prizes[i][idx])) / 3.0
                    aux[2 * hi + 1] = max(0, now[1] - int(tl_prizes[1 - i][idx])) / 3.0
            cols["aux"].append(aux)
            cols["opp_fp"].append(opp_fp[i])
            n += 1
    if n == 0:
        return None
    ep = name.rsplit("/", 1)[-1].removesuffix(".json")
    return ep, cols, n


def extract_day(zip_path: str, out_path: str, fp: str | None, arch: str | None,
                workers: int, limit: int = 0, prior_path: str | None = None,
                cards_path: str = "", forward: bool = True,
                all_sides: bool = False, v6: bool = False) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
    if limit:
        names = names[:limit]
    manifest = load_manifest(zip_path)

    agg: dict[str, list] = {}
    ep_rows: list[tuple[str, int]] = []
    total = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(zip_path, fp, arch, prior_path,
                                       cards_path, forward, all_sides,
                                       v6)) as ex:
        for res in ex.map(_episode_samples, names, chunksize=8):
            if res is None:
                continue
            ep, cols, n = res
            for key, vals in cols.items():
                agg.setdefault(key, []).extend(vals)
            ep_rows.append((ep, n))
            total += n
    if total == 0:
        print(f"  no matching games in {os.path.basename(zip_path)}")
        return 0

    out: dict[str, np.ndarray] = {}
    for key in FIXED_KEYS:
        arr = np.stack(agg[key])
        out[key] = arr.astype(np.float16) if key in F16_KEYS else arr
    for key in RAGGED_KEYS + ("label",):
        cat = np.concatenate(agg[key], axis=0)
        out[key] = cat.astype(np.float16) if key == "opt_feats" else cat
    lens = np.array([len(x) for x in agg["label"]], dtype=np.int64)
    out["opt_offsets"] = np.concatenate([[0], np.cumsum(lens)])
    out["arch"] = np.stack(agg["arch"]).astype(np.float16)
    out["ora"] = np.stack(agg["ora"]).astype(np.float16)
    out["ora_ids"] = np.stack(agg["ora_ids"])
    out["stadium_id"] = np.array(agg["stadium_id"], dtype=np.int16)
    out["seq"] = np.stack(agg["seq"])
    out["ctx_id"] = np.array(agg["ctx_id"], dtype=np.int16)
    out["k"] = np.array(agg["k"], dtype=np.int8)
    out["min_c"] = np.array(agg["min_c"], dtype=np.int8)
    out["max_c"] = np.array(agg["max_c"], dtype=np.int8)
    out["value"] = np.array(agg["value"], dtype=np.float32)
    out["side"] = np.array(agg["side"], dtype=np.int8)
    out["turn"] = np.array(agg["turn"], dtype=np.int8)
    out["aux"] = np.stack(agg["aux"]).astype(np.float16)
    out["aux_m"] = np.ones(total, dtype=np.float16)
    out["opp_fp"] = np.array(agg["opp_fp"], dtype=np.uint32)
    score = np.zeros(total, dtype=np.float32)
    ep_id = np.zeros(total, dtype=np.int32)
    pos = 0
    for ei, (ep, n) in enumerate(ep_rows):
        score[pos:pos + n] = manifest.get(ep, {}).get("avg_score") or 0.0
        ep_id[pos:pos + n] = ei
        pos += n
    out["score"] = score
    out["ep_id"] = ep_id  # per-sample episode index within this day shard
    out["feat_version"] = np.array([FEAT_V6 if v6 else FEAT_VERSION],
                                   dtype=np.int16)
    if cards_path and os.path.exists(cards_path):
        out["cards_hash"] = np.array(file_md5(cards_path))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    games = len(ep_rows)
    print(f"  {os.path.basename(out_path)}: {games} sides, {total} samples "
          f"({time.time()-t0:.0f}s)", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", default=None, help="exact deck fingerprint, e.g. 220eddd2")
    ap.add_argument("--arch", default=None, help="or: archetype name, e.g. mega_lopunny")
    ap.add_argument("--all", action="store_true",
                    help="or: every side of every episode (shared-base BC)")
    ap.add_argument("--v6", action="store_true",
                    help="v6 feature encoding (wide board, deck-bag, 28 fwd cols)")
    ap.add_argument("--days", default=None, help="comma list, default all zips")
    ap.add_argument("--zips-dir", default=os.path.join(ROOT, "replays_zip"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "bc"))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="episodes per day (smoke)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"),
                    help="deck prior for opponent-model features; '' to disable")
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--no-forward-features", action="store_true",
                    help="skip engine lookahead probing (fwd columns stay 0)")
    args = ap.parse_args()
    if sum(x is not None for x in (args.fp, args.arch)) + int(args.all) != 1:
        sys.exit("specify exactly one of --fp / --arch / --all")
    prior = args.prior if args.prior and os.path.exists(args.prior) else None
    if prior is None:
        print("[warn] no deck prior: arch features will be zeros")

    zips = sorted(glob.glob(os.path.join(args.zips_dir, "*.zip")))
    if args.days:
        want = set(args.days.split(","))
        zips = [z for z in zips if os.path.splitext(os.path.basename(z))[0] in want]
    tag = args.fp or args.arch or "all"
    grand = 0
    for zp in zips:
        day = os.path.splitext(os.path.basename(zp))[0]
        out = os.path.join(args.out_dir, f"{day}_{tag}.npz")
        if os.path.exists(out) and not args.force:
            print(f"skip {day} (exists)", flush=True)
            continue
        print(f"=== {day} ===", flush=True)
        grand += extract_day(zp, out, args.fp, args.arch, args.workers, args.limit,
                             prior, args.cards,
                             forward=not args.no_forward_features,
                             all_sides=args.all, v6=args.v6)
    print(f"TOTAL {grand} samples")


if __name__ == "__main__":
    main()
