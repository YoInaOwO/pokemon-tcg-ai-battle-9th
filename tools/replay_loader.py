"""Replay loading, deck identity, and archetype classification.

Deck naming convention used across the whole project:
    label = f"{archetype}__{fingerprint}"   e.g. marnie_grimmsnarl__e521a283

- archetype: rule-based on signature card IDs (see ARCHETYPE_RULES / classify_deck).
  Priority-ordered; first match wins; fallback "other".
- fingerprint: content hash of the 60-card multiset (see deck_fingerprint).
  Stable across days / orderings; changing a single card changes it entirely.
"""

from __future__ import annotations

import hashlib
import io
import csv
import zipfile
from collections import Counter

try:
    import orjson as _json

    def _loads(b: bytes):
        return _json.loads(b)
except ImportError:  # pragma: no cover
    import json as _json

    def _loads(b: bytes):
        return _json.loads(b)


# ---------------------------------------------------------------------------
# Deck identity
# ---------------------------------------------------------------------------

def deck_fingerprint(deck: list[int], digits: int = 8) -> str:
    """Content-addressed decklist ID: stable across days, orderings, pools.

    Canonical form is the sorted (card_id, count) multiset, so the same
    60-card list always maps to the same hex ID no matter which day it was
    seen on or how the replay serialized the card order.
    """
    counts = sorted(Counter(int(x) for x in deck).items())
    canon = ",".join(f"{cid}:{n}" for cid, n in counts)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()[: int(digits)]


# Priority-ordered signature rules. Each rule is (archetype, groups) where
# groups is a list of card-ID groups; a deck matches if it contains at least
# one ID from EVERY group. First matching rule wins.
ARCHETYPE_RULES: list[tuple[str, list[list[int]]]] = [
    ("marnie_grimmsnarl", [[648]]),               # Marnie's Grimmsnarl ex
    ("cynthia", [[381]]),                         # Cynthia's Garchomp ex
    ("ns_zoroark", [[293]]),                      # N's Zoroark ex
    ("alakazam", [[245, 743]]),                   # Alakazam
    ("mega_lopunny", [[849]]),                    # Mega Lopunny ex
    ("mega_lucario", [[678]]),                    # Mega Lucario ex
    ("dragapult", [[121]]),                       # Dragapult ex
    ("kangaskhan_crustle", [[756], [345, 533]]),  # Mega Kangaskhan ex + Crustle
    ("kangaskhan_box", [[756]]),                  # Mega Kangaskhan ex, no Crustle
    ("mega_venusaur", [[652]]),                   # Mega Venusaur ex
    ("ogerpon_hydrapple", [[96], [150]]),         # Teal Mask Ogerpon ex + Hydrapple ex
    ("grookey_dipplin", [[90], [93, 347, 921]]),  # Thwackey + Dipplin
    ("ogerpon", [[96]]),                          # Teal Mask Ogerpon ex (mono / misc)
    ("mega_froslass", [[861]]),                   # Mega Froslass ex (non-Lopunny)
]


def classify_deck(deck: list[int]) -> str:
    """Rule-based archetype from signature card IDs; "other" if no rule hits."""
    ids = set(int(x) for x in deck)
    for name, groups in ARCHETYPE_RULES:
        if all(ids & set(g) for g in groups):
            return name
    return "other"


def deck_label(deck: list[int]) -> str:
    return f"{classify_deck(deck)}__{deck_fingerprint(deck)}"


# ---------------------------------------------------------------------------
# Episode parsing
# ---------------------------------------------------------------------------

def parse_episode(raw: bytes, ep_id: str | None = None) -> dict:
    """Extract per-episode metadata needed for meta analysis.

    Cheap fields only: teams, rewards, decks, first player, turns.
    (End-reason logs are absent from all 115k replays -> field removed.)
    """
    d = _loads(raw)
    steps = d.get("steps") or []
    info = d.get("info") or {}
    meta: dict = {
        "ep": ep_id or str(info.get("EpisodeId", "")),
        "teams": info.get("TeamNames") or [None, None],
        "rewards": d.get("rewards") or [None, None],
        "statuses": d.get("statuses") or [None, None],
        "num_steps": len(steps),
        "deck0": None,
        "deck1": None,
        "first_player": None,
        "turns": None,
    }
    try:
        vis = steps[0][0].get("visualize")
        decks = vis[0]["action"]
        if len(decks[0]) == 60 and len(decks[1]) == 60:
            meta["deck0"] = [int(x) for x in decks[0]]
            meta["deck1"] = [int(x) for x in decks[1]]
    except (IndexError, KeyError, TypeError):
        pass
    # first player becomes known within the first few steps
    for step in steps[1:8]:
        try:
            cur = step[0]["observation"]["current"]
            if cur and cur.get("firstPlayer", -1) >= 0:
                meta["first_player"] = cur["firstPlayer"]
                break
        except (IndexError, KeyError, TypeError):
            continue
    if steps:
        # the two players' final observations are not synchronized (the
        # inactive side can lag a turn) -> take the max of both
        turns = []
        for ai in (0, 1):
            try:
                cur = steps[-1][ai]["observation"]["current"]
                if cur and cur.get("turn") is not None:
                    turns.append(int(cur["turn"]))
            except (IndexError, KeyError, TypeError):
                continue
        meta["turns"] = max(turns) if turns else None
    return meta


def load_manifest(zip_path: str) -> dict[str, dict]:
    """manifest.csv inside a daily replay zip -> {episode_id: row dict}."""
    with zipfile.ZipFile(zip_path) as zf:
        if "manifest.csv" not in zf.namelist():
            return {}
        raw = zf.read("manifest.csv").decode("utf-8", "replace")
    rows = {}
    for row in csv.DictReader(io.StringIO(raw)):
        rows[row["episode_id"]] = {
            "avg_score": float(row["avg_score"]),
            "min_score": float(row["min_score"]),
            "create_time": row["create_time"],
        }
    return rows


def list_episode_names(zip_path: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist() if n.endswith(".json")]


# ---------------------------------------------------------------------------
# Slot building (one record per (episode, side)) shared by analysis scripts
# ---------------------------------------------------------------------------

def build_slots(rows: list[dict], manifest: dict[str, dict]) -> tuple[list[dict], int, int]:
    """Meta rows (from etl_day JSONL) -> per-deck-slot records.

    Returns (slots, n_no_deck, n_no_result). Result convention:
    res in {"w","l","d",None}; wr aggregation should score draws as 0.5.
    """
    slots: list[dict] = []
    n_no_deck = n_no_result = 0
    for r in rows:
        if not r["deck0"] or not r["deck1"]:
            n_no_deck += 1
            continue
        rw = r["rewards"]
        score = manifest.get(r["ep"], {}).get("avg_score")
        for side in (0, 1):
            deck = r[f"deck{side}"]
            res = None
            if rw[side] is not None and rw[1 - side] is not None:
                res = "w" if rw[side] > rw[1 - side] else ("l" if rw[side] < rw[1 - side] else "d")
            slots.append({
                "ep": r["ep"], "side": side,
                "team": r["teams"][side] or "(unnamed)",
                "deck": deck,
                "fp": deck_fingerprint(deck),
                "arch": classify_deck(deck),
                "res": res,
                "first": (r["first_player"] == side) if r["first_player"] is not None else None,
                "score": score,
                "turns": r["turns"],
            })
        if rw[0] is None or rw[1] is None:
            n_no_result += 1
    return slots, n_no_deck, n_no_result
