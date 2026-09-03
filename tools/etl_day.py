"""ETL one daily replay zip into a JSONL of per-episode metadata.

Usage:
    python tools/etl_day.py --zip replays_zip/0807.zip --out data/meta/0807.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orjson

import replay_loader

_ZIP_PATH: str | None = None
_ZF: zipfile.ZipFile | None = None


def _init_worker(zip_path: str) -> None:
    global _ZIP_PATH, _ZF
    _ZIP_PATH = zip_path
    _ZF = zipfile.ZipFile(zip_path)


_EMPTY = {
    "teams": [None, None], "rewards": [None, None], "statuses": [None, None],
    "num_steps": 0, "deck0": None, "deck1": None,
    "first_player": None, "turns": None,
}


def _work(name: str) -> bytes:
    """One bad episode must never kill the whole day: emit an error row instead."""
    ep = name.rsplit("/", 1)[-1].removesuffix(".json")
    try:
        raw = _ZF.read(name)
        meta = replay_loader.parse_episode(raw, ep_id=ep)
    except Exception as e:  # noqa: BLE001
        meta = {"ep": ep, **_EMPTY, "error": f"{type(e).__name__}: {e}"}
    return orjson.dumps(meta)


DEFAULT_WORKERS = 20  # rented box: 25 vCPU, keep 5 headroom


def run(zip_path: str, out_path: str, workers: int = DEFAULT_WORKERS) -> int:
    names = replay_loader.list_episode_names(zip_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    done = errors = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(zip_path,)
    ) as ex, open(out_path, "wb") as f:
        for line in ex.map(_work, names, chunksize=16):
            f.write(line)
            f.write(b"\n")
            done += 1
            if b'"error"' in line:
                errors += 1
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(names)}  {rate:.0f} eps/s", flush=True)
    print(f"DONE {done} episodes ({errors} parse errors) in {time.time()-t0:.0f}s -> {out_path}",
          flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()
    run(args.zip, args.out, args.workers)


if __name__ == "__main__":
    main()
