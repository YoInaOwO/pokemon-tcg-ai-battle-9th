"""Gauntlet: evaluate one checkpoint against the whole opponent roster.

    python -m ptcg_rl.eval_gauntlet --ckpt runs/ppo_v1/model.pt \
        --opp-config configs/opponents_default.json --league-dir runs/ppo_v1/pool \
        --games-per 60 --workers 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ptcg_rl.eval_local import (DEFAULT_DECK, SAMPLE_DECK,  # noqa: E402
                                prepare_submission, run_match)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.npz"))
    ap.add_argument("--deck", default=DEFAULT_DECK)
    ap.add_argument("--opp-config", default=os.path.join(ROOT, "configs", "opponents_default.json"))
    ap.add_argument("--league-dir", default=None)
    ap.add_argument("--league-n", type=int, default=2, help="newest league snapshots to face")
    ap.add_argument("--games-per", type=int, default=60)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--mcts-budget", type=float, default=0.0,
                    help="run OUR side with search (BC/PPO x policy/MCTS grids)")
    ap.add_argument("--mcts-det", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="reuse across gauntlets for paired A/B comparisons")
    ap.add_argument("--no-random", action="store_true",
                    help="skip the random-policy smoke opponent")
    ap.add_argument("--no-mirror", action="store_true",
                    help="skip the self-mirror row")
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

    roster: list[tuple[str, str, str]] = []
    if not args.no_random:
        roster.append(("random", "random", SAMPLE_DECK))
    pool_w: dict[str, float] = {}  # roster name -> arena-share weight
    with open(args.opp_config) as f:
        for e in json.load(f):
            if e["kind"] == "bc" and os.path.exists(e["ckpt"]) and os.path.exists(e["deck"]):
                roster.append((f"bc:{e['name']}",
                               f"ckpt:{os.path.abspath(e['ckpt'])}",
                               os.path.abspath(e["deck"])))
                pool_w[f"bc:{e['name']}"] = float(e.get("weight", 0.0))
    if args.league_dir:
        snaps = sorted(glob.glob(os.path.join(args.league_dir, "*.pt")))[-args.league_n:]
        for p in snaps:
            roster.append((f"league:{os.path.basename(p)}",
                           f"ckpt:{os.path.abspath(p)}", args.deck))
    if sub_dir is None and not args.no_mirror:
        # a packaged submission cannot also serve as the opponent (its main.py
        # is a stateful singleton per process), so no mirror row in sub mode
        roster.append(("self-mirror", f"ckpt:{args.ckpt}", args.deck))

    mcts = ({"budget": args.mcts_budget, "det": args.mcts_det,
             "prior": os.path.abspath(args.prior)} if args.mcts_budget > 0 else None)
    if mcts:
        print(f"our side: MCTS budget={args.mcts_budget}s det={args.mcts_det}")
    print(f"{'opponent':<28} {'games':>5} {'W/L/D/A':>12} {'WR':>6} "
          f"{'95% CI':>14} {'lat ms':>7}")
    results = {}
    for name, opp, opp_deck in roster:
        cfg = {"ckpt": os.path.abspath(args.ckpt), "cards": os.path.abspath(args.cards),
               "deck": os.path.abspath(args.deck), "opp_deck": os.path.abspath(opp_deck),
               "opponent": opp, "temperature": 0.0, "mcts": mcts, "sub_dir": sub_dir,
               "device": args.device,
               "prior": os.path.abspath(args.prior) if os.path.exists(args.prior) else None}
        r = run_match(cfg, args.games_per, args.workers, seed_base=args.seed_base)
        results[name] = r["wr"]
        wld = f"{r['w']}/{r['l']}/{r['d']}/{r['abort']}"
        ci = f"[{r['wr_lo']:.3f},{r['wr_hi']:.3f}]"
        print(f"{name:<28} {r['games']:>5} {wld:>12} {r['wr']:>6.3f} "
              f"{ci:>14} {r['lat_ms']:>7.1f}", flush=True)
    print("\nsummary:", json.dumps({k: round(v, 3) for k, v in results.items()}))
    # pool-weighted WR: each BC opponent counts by its arena share from the
    # opponents config (mirror/random/league rows excluded)
    num = den = 0.0
    for name, w in pool_w.items():
        if name in results and w > 0:
            num += w * results[name]
            den += w
    if den > 0:
        print(f"pool-weighted WR (bc opponents, arena shares): {num / den:.4f} "
              f"(weight covered: {den:.3f})")


if __name__ == "__main__":
    main()
