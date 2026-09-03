"""ETL every daily replay zip under replays_zip/ into data/meta/{day}.jsonl.

Skips days whose output already exists (resume-friendly); --force redoes all.

Usage (on the training box):
    python tools/etl_all.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_loader
from etl_day import DEFAULT_WORKERS, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_complete(out_path: str, zip_path: str) -> tuple[bool, str]:
    """A day is done only if the JSONL has exactly one line per zip episode.

    Guards against partial files left by an interrupted / crashed earlier run.
    """
    if not os.path.exists(out_path):
        return False, "missing"
    with open(out_path, "rb") as f:
        n_lines = f.read().count(b"\n")
    n_eps = len(replay_loader.list_episode_names(zip_path))
    if n_lines == n_eps:
        return True, f"complete ({n_eps} eps)"
    return False, f"partial {n_lines}/{n_eps}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips-dir", default=os.path.join(ROOT, "replays_zip"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "meta"))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(args.zips_dir, "*.zip")))
    if not zips:
        sys.exit(f"no zips found in {args.zips_dir}")
    for zp in zips:
        day = os.path.splitext(os.path.basename(zp))[0]
        out = os.path.join(args.out_dir, f"{day}.jsonl")
        if not args.force:
            ok, why = is_complete(out, zp)
            if ok:
                print(f"skip {day} ({why})", flush=True)
                continue
            print(f"=== {day} ({why}) ===", flush=True)
        else:
            print(f"=== {day} (force) ===", flush=True)
        run(zp, out, args.workers)


if __name__ == "__main__":
    main()
