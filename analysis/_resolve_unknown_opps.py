"""Chase down the last unresolved opponents in the Elo bucket.

A submission's decklist is recovered by finding one of its episodes that also
appears in the 0817/0818 replay ETL. The cached ListEpisodes docs we already
have (ep_cache*, gold_eps) cover the busy teams; what is left are low-activity
teams we met once or twice. ListEpisodes returns a submission's most recent
1000 episodes -- for a team that played fewer than that in total, the response
reaches all the way back to 0817/0818 and the intersection lands.

Fetches only the still-unresolved opponent submissions, caches to
build/opp_eps/, and reports which of them the decklist could be pinned down for.

    python build/_resolve_unknown_opps.py
"""
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from replay_loader import classify_deck, deck_fingerprint  # noqa: E402

OUR = [55535786, 55554179]
CACHE = os.path.join(ROOT, "build", "opp_eps")


def call(sub_id):
    path = os.path.join(CACHE, f"{sub_id}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(
        "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
        data=json.dumps({"submissionId": sub_id}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(7):
        try:
            doc = json.loads(urllib.request.urlopen(req, timeout=120).read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 6:
                raise
            wait = 5 * 2 ** attempt
            print(f"    429, waiting {wait}s", flush=True)
            time.sleep(wait)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    time.sleep(2.5)
    return doc


# ---- replay ground truth ----
ep_meta = {}
for day in ("0817", "0818"):
    with open(os.path.join(ROOT, "data", "meta", f"{day}.jsonl"), "rb") as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or not r.get("deck0") or not r.get("deck1"):
                continue
            nm = r["teams"]
            if not nm or None in nm or nm[0] == nm[1]:
                continue
            ep_meta[str(r["ep"])] = (nm, [r["deck0"], r["deck1"]])

# ---- what the existing caches already resolve ----
known, team_name = set(), {}
for d in ("ep_cache", "ep_cache2", "gold_eps", "opp_eps"):
    for p in glob.glob(os.path.join(ROOT, "build", d, "*.json")):
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        for t in doc.get("teams", []):
            team_name[t["id"]] = t["teamName"]
        for e in doc.get("episodes", []):
            if str(e["id"]) in ep_meta:
                for ag in e["agents"]:
                    known.add(ag["submissionId"])

# ---- our Elo-bucket opponents that are still unresolved ----
need = Counter()
opp_team = {}
for sid in OUR:
    with open(os.path.join(ROOT, "build", "gold_eps", f"{sid}.json"),
              encoding="utf-8") as f:
        doc = json.load(f)
    for e in doc.get("episodes", []):
        if e.get("state") != "COMPLETED":
            continue
        me = next((a for a in e["agents"] if a.get("submissionId") == sid), None)
        op = next((a for a in e["agents"] if a.get("submissionId") != sid), None)
        if (me is None or op is None or me.get("initialScore") is None
                or op.get("initialScore") is None):
            continue
        if abs(me["initialScore"] - op["initialScore"]) > 200:
            continue                       # random-matchmaking bucket
        if op["submissionId"] not in known:
            need[op["submissionId"]] += 1
            opp_team[op["submissionId"]] = team_name.get(op["teamId"], "?")
print(f"unresolved opponent submissions: {len(need)} "
      f"({sum(need.values())} games)")

# ---- fetch each and try the intersection ----
found = {}
for i, (sub, n) in enumerate(need.most_common(), 1):
    doc = call(sub)
    for t in doc.get("teams", []):
        team_name[t["id"]] = t["teamName"]
    name = opp_team.get(sub) or "?"
    hit = None
    n_eps = sum(1 for e in doc.get("episodes", []) if e.get("state") == "COMPLETED")
    for e in doc.get("episodes", []):
        m = ep_meta.get(str(e["id"]))
        if not m:
            continue
        nm, decks = m
        ag_team = {a["submissionId"]: team_name.get(a.get("teamId"))
                   for a in e["agents"]}
        side = nm.index(ag_team[sub]) if ag_team.get(sub) in nm else None
        if side is None:
            continue
        hit = decks[side]
        break
    if hit:
        found[sub] = (name, n, deck_fingerprint(hit), classify_deck(hit))
        print(f"{i:3}/{len(need)} {name[:24]:24} sub={sub} {n:2}局 eps={n_eps:4} "
              f"-> {classify_deck(hit)} / {deck_fingerprint(hit)}")
    else:
        print(f"{i:3}/{len(need)} {name[:24]:24} sub={sub} {n:2}局 eps={n_eps:4} "
              f"-> still unresolved (history does not reach 0817/0818)")

print(f"\nresolved {len(found)}/{len(need)} submissions, "
      f"{sum(v[1] for v in found.values())}/{sum(need.values())} games")
by_arch = defaultdict(int)
for _, n, _, arch in found.values():
    by_arch[arch] += n
print("newly attributed games by archetype:",
      dict(sorted(by_arch.items(), key=lambda kv: -kv[1])))
