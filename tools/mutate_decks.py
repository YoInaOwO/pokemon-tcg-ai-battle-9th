"""Domain randomization: mutated in-meta decklists for the PPO opponent pool.

For each major archetype (that has a BC pilot anchor), take its highest-share
prior list and produce variants with 5-10 random card swaps. Replacement cards
are drawn from the union of that archetype's other prior lists (per-card count
capped at the max seen in any single list), so mutants stay thematically legal;
every mutant is validated by an actual engine battle_start before being kept.

The point is not realism -- opponents that play "slightly wrong" lists with a
competent (BC anchor) pilot are exactly the OOD signal the policy never sees
from clean meta lists.

    python tools/mutate_decks.py --per-arch 2 \
        --merge-base configs/opponents_env.json \
        --budget 0.06 --out configs/opponents_env_mut.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sample_submission", "sample_submission"))

from make_env_pool import ARCH_CKPT  # noqa: E402


def engine_ok(deck: list[int]) -> bool:
    """True iff the engine accepts this 60-list at battle_start."""
    from cg.game import battle_start, battle_finish
    obs, _ = battle_start(list(deck), list(deck))
    ok = obs is not None
    battle_finish()
    return ok


def mutate(base: list[int], pool_cap: dict[int, int], k: int,
           rng: random.Random) -> list[int] | None:
    """Swap k random cards of `base` for pool cards, respecting per-card caps."""
    deck = list(base)
    cnt = collections.Counter(deck)
    candidates = [c for c in pool_cap]
    for _ in range(k):
        out_i = rng.randrange(len(deck))
        removed = deck[out_i]
        cnt[removed] -= 1
        choices = [c for c in candidates
                   if c != removed and cnt[c] < pool_cap[c]]
        if not choices:
            return None
        add = rng.choice(choices)
        deck[out_i] = add
        cnt[add] += 1
    return sorted(deck)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=os.path.join(ROOT, "data", "deck_prior.json"))
    ap.add_argument("--per-arch", type=int, default=2, help="mutants per archetype")
    ap.add_argument("--swaps", type=int, nargs=2, default=[5, 10],
                    help="min/max card swaps per mutant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-base", default=None,
                    help="existing opponents config to blend the mutants into")
    ap.add_argument("--budget", type=float, default=0.06,
                    help="total weight given to mutants when merging")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs",
                                                  "opponents_mut.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.prior, encoding="utf-8") as f:
        prior = json.load(f)
    by_arch: dict[str, list[dict]] = collections.defaultdict(list)
    for e in prior:
        if e.get("arch") in ARCH_CKPT:
            by_arch[e["arch"]].append(e)

    out_dir = os.path.join(ROOT, "data", "decks_mut")
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for arch, lists in sorted(by_arch.items(),
                              key=lambda kv: -sum(e["w"] for e in kv[1])):
        if not os.path.exists(os.path.join(ROOT, ARCH_CKPT[arch])):
            continue
        lists.sort(key=lambda e: -e["w"])
        base = [int(x) for x in lists[0]["cards"]]
        # per-card cap = max copies seen in any single list of this archetype
        pool_cap: dict[int, int] = {}
        for e in lists:
            for c, k in collections.Counter(int(x) for x in e["cards"]).items():
                pool_cap[c] = max(pool_cap.get(c, 0), k)
        made = 0
        tries = 0
        while made < args.per_arch and tries < 40:
            tries += 1
            k = rng.randint(args.swaps[0], args.swaps[1])
            mut = mutate(base, pool_cap, k, rng)
            if mut is None or not engine_ok(mut):
                continue
            name = f"mut_{arch}_{made}"
            csv = os.path.join("data", "decks_mut", f"{arch}_mut{made}.csv")
            with open(os.path.join(ROOT, csv), "w", newline="\n") as f:
                f.write("\n".join(str(x) for x in mut) + "\n")
            entries.append({"kind": "bc", "name": name, "ckpt": ARCH_CKPT[arch],
                            "deck": csv.replace(os.sep, "/"), "weight": 0.0})
            print(f"  {name}: {k} swaps, engine OK")
            made += 1
        if made < args.per_arch:
            print(f"  [warn] {arch}: only {made}/{args.per_arch} legal mutants")

    if not entries:
        sys.exit("no mutants produced")
    w_each = round(args.budget / len(entries), 5)
    for e in entries:
        e["weight"] = w_each

    if args.merge_base:
        with open(args.merge_base, encoding="utf-8") as f:
            base_cfg = json.load(f)
        scale = 1.0 - args.budget
        for e in base_cfg:
            e["weight"] = round(e.get("weight", 0.0) * scale, 5)
        entries = base_cfg + entries
    with open(args.out, "w", newline="\n") as f:
        f.write(json.dumps(entries, indent=2) + "\n")
    print(f"written {args.out}: {len(entries)} entries "
          f"(mutants carry {args.budget:.0%} total weight)")


if __name__ == "__main__":
    main()
