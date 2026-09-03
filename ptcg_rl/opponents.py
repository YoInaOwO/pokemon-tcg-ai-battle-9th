"""Opponent pool for self-play / evaluation.

Config: JSON list of entries with weights. Kinds:
    mirror  - the actor's own current policy (self-play; both seats recorded)
    league  - random snapshot from <league_dir> (recent-biased)
    bc      - fixed checkpoint (BC anchor or other-deck BC agents), needs ckpt+deck
    random  - uniform random legal action
    script  - rule-based Kaggle sample agent; needs module (exposing
              agent(obs)->list[int] and my_deck) + name

Entries whose ckpt/deck files are missing are skipped with a warning.
"""

from __future__ import annotations

import glob
import json
import os
import random

import torch

from .agent import BCAgent


def _read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split()[:60]]


def random_act(obs: dict, rng: random.Random) -> list[int]:
    sel = obs["select"]
    n = len(sel["option"])
    # maxCount may exceed the option count; sample() would raise ValueError
    k = min(int(sel.get("maxCount") or 1), n)
    return rng.sample(range(n), k)


class LeagueAgent:
    """Lazily loads league snapshots, keeps a tiny LRU cache."""

    def __init__(self, league_dir: str, cards_path: str, cache: int = 3,
                 prior_path: str | None = None, deck: list[int] | None = None):
        self.dir = league_dir
        self.cards = cards_path
        self.prior = prior_path
        self.deck = deck  # league snapshots are our own policy -> our deck
        self.cache: dict[str, BCAgent] = {}
        self.cache_n = cache

    def pick(self, rng: random.Random) -> BCAgent | None:
        paths = sorted(glob.glob(os.path.join(self.dir, "*.pt")))
        if not paths:
            return None
        # recent-biased: half the mass on the newest 25%
        if rng.random() < 0.5:
            paths = paths[-max(1, len(paths) // 4):]
        p = rng.choice(paths)
        if p not in self.cache:
            if len(self.cache) >= self.cache_n:
                self.cache.pop(next(iter(self.cache)))
            self.cache[p] = BCAgent(p, self.cards, threads=1,
                                    prior_path=self.prior, deck=self.deck)
        return self.cache[p]


class ScriptAgent:
    """Rule-based opponent: wraps a module exposing agent(obs) and my_deck."""

    def __init__(self, module: str):
        import importlib
        self.mod = importlib.import_module(module)
        self.deck = [int(x) for x in self.mod.my_deck]

    def act(self, obs: dict) -> list[int]:
        return self.mod.agent(obs)


class OpponentPool:
    def __init__(self, config_path: str, our_deck_path: str, cards_path: str,
                 league_dir: str | None = None, quiet: bool = False,
                 prior_path: str | None = None):
        with open(config_path) as f:
            entries = json.load(f)
        self.our_deck = _read_deck(our_deck_path)
        self.cards = cards_path
        self.league = LeagueAgent(league_dir, cards_path, prior_path=prior_path,
                                  deck=self.our_deck) if league_dir else None
        self.entries = []
        # random opponents pilot decks drawn from the replay deck prior when
        # available: exposes training to unseen card sets (off-meta robustness)
        # instead of always mirroring our own list. Decks whose archetype has
        # a BC anchor loaded are piloted by that anchor ("prior:<arch>");
        # only unmapped archetypes fall back to uniform random actions.
        self.random_decks: list[list[int]] = []
        self.random_weights: list[float] = []
        self.random_archs: list[str] = []
        if prior_path and os.path.exists(prior_path):
            try:
                with open(prior_path) as f:
                    prior = json.load(f)
                good = [e for e in prior if len(e.get("cards", [])) == 60]
                self.random_decks = [e["cards"] for e in good]
                self.random_weights = [e.get("w", 1.0) for e in good]
                self.random_archs = [e.get("arch", "other") for e in good]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        agent_cache: dict[str, BCAgent] = {}  # entries sharing a ckpt share one model
        for e in entries:
            kind = e["kind"]
            if kind == "bc":
                if not (os.path.exists(e["ckpt"]) and os.path.exists(e["deck"])):
                    if not quiet:
                        print(f"[pool] skip bc '{e.get('name', '?')}' (missing files)")
                    continue
                e = dict(e)
                key = os.path.abspath(e["ckpt"])
                e["deck_list"] = _read_deck(e["deck"])
                if key not in agent_cache:
                    # deck enables the fwd probe for v5+ anchors (no-op for old)
                    agent_cache[key] = BCAgent(e["ckpt"], cards_path, threads=1,
                                               prior_path=prior_path,
                                               deck=e["deck_list"])
                e["agent"] = agent_cache[key]
            elif kind == "script":
                e = dict(e)
                try:
                    e["agent"] = ScriptAgent(e["module"])
                except Exception as ex:  # noqa: BLE001
                    if not quiet:
                        print(f"[pool] skip script '{e.get('name', '?')}' "
                              f"({type(ex).__name__}: {ex})")
                    continue
                e["deck_list"] = e["agent"].deck
            elif kind == "league" and self.league is None:
                continue
            self.entries.append(e)
        # archetype -> BC anchor, for piloting prior-sampled decks (names from
        # make_env_pool are "<arch>__<fingerprint>")
        self.arch_agents: dict[str, BCAgent] = {}
        for e in self.entries:
            if e["kind"] == "bc" and "__" in e.get("name", ""):
                self.arch_agents.setdefault(e["name"].split("__")[0], e["agent"])
        total = sum(e["weight"] for e in self.entries)
        self.weights = [e["weight"] / total for e in self.entries]
        if not quiet:
            print(f"[pool] {[(e['kind'], e.get('name', ''), round(w, 3)) for e, w in zip(self.entries, self.weights)]}")

    def reload_weights(self, path: str) -> bool:
        """Replace sampling weights from a learner-published JSON map.

        Keys: "bc:<name>" for bc entries, the kind for mirror/random/league.
        Entries absent from the map keep their current weight. Used by the
        adaptive-opponent scheme in selfplay_ppo (low-WR opponents upsampled);
        safe to call every publish, no-op if the file is missing/partial."""
        try:
            with open(path) as f:
                wmap = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        new = []
        for e, w in zip(self.entries, self.weights):
            if e["kind"] in ("bc", "script"):
                key = f"{e['kind']}:{e.get('name', '')}"
            else:
                key = e["kind"]
            new.append(max(0.0, float(wmap.get(key, w))))
        total = sum(new)
        if total <= 0:
            return False
        self.weights = [w / total for w in new]
        return True

    @staticmethod
    def _sync_probe(agent, deck: list[int]):
        """Shared agents pilot different lists; the fwd probe (if any) must
        determinize with the deck this particular game actually uses."""
        probe = getattr(agent, "probe", None)
        if probe is not None:
            probe.set_deck(deck)
        return agent

    def sample(self, rng: random.Random) -> tuple[str, object, list[int]]:
        """-> (name, actor, opp_deck). actor: 'mirror' | 'random' | agent object."""
        e = rng.choices(self.entries, weights=self.weights, k=1)[0]
        kind = e["kind"]
        if kind == "mirror":
            return "mirror", "mirror", self.our_deck
        if kind == "random":
            if self.random_decks:
                i = rng.choices(range(len(self.random_decks)),
                                weights=self.random_weights, k=1)[0]
                deck = list(self.random_decks[i])
                anchor = self.arch_agents.get(self.random_archs[i])
                if anchor is not None:
                    return (f"prior:{self.random_archs[i]}",
                            self._sync_probe(anchor, deck), deck)
                return "random", "random", deck
            return "random", "random", self.our_deck
        if kind == "league":
            ag = self.league.pick(rng)
            if ag is None:  # empty league yet -> mirror
                return "mirror", "mirror", self.our_deck
            return "league", self._sync_probe(ag, self.our_deck), self.our_deck
        if kind == "script":
            return f"script:{e.get('name', '')}", e["agent"], e["deck_list"]
        return (f"bc:{e.get('name', '')}",
                self._sync_probe(e["agent"], e["deck_list"]), e["deck_list"])


@torch.no_grad()
def opponent_act(actor, obs: dict, rng: random.Random) -> list[int]:
    if actor == "random":
        return random_act(obs, rng)
    return actor.act(obs)
