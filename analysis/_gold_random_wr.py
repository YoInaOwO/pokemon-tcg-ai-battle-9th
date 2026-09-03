"""Random-matchmaking win rate for every team in the gold zone (top 23).

The ladder pairs by Elo 90% of the time and draws a uniformly random
leaderboard opponent the other 10%. Since the leaderboard's median score is
622 and 96.4% of teams sit below 950, a random draw is almost always a much
weaker agent -- so for anyone near the top, the random 10% is a bucket of
free-ish games, and how cleanly a team converts it is a robustness signal
(timeouts, crashes and rule-edge bugs show up here, not against peers).

Rather than threshold on the rating gap (which needs a per-team cut), take
each team's N lowest-rated opponents across both of its final submissions.
N=200 matches the ~10% share of the ~2000 episodes ListEpisodes returns per
team (1000 per submission).

Responses are cached under build/gold_eps/ so re-runs cost no API calls.

    python build/_gold_random_wr.py [--n 200]
"""
import argparse
import csv
import io
import json
import os
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "build", "gold_eps")
LB_ZIP = os.path.join(ROOT, "build", "lb_final", "pokemon-tcg-ai-battle.zip")
OUR_SUBS = [55535786, 55554179]
GOLD = 23


def call(sub_id):
    path = os.path.join(CACHE, f"{sub_id}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(
        "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
        data=json.dumps({"submissionId": sub_id}).encode(),
        headers={"Content-Type": "application/json"})
    # the endpoint rate-limits at roughly one call/second sustained; back off
    # rather than dropping a team out of the comparison
    for attempt in range(7):
        try:
            doc = json.loads(urllib.request.urlopen(req, timeout=120).read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 6:
                raise
            wait = 5 * 2 ** attempt
            print(f"    429 on {sub_id}, waiting {wait}s", flush=True)
            time.sleep(wait)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    time.sleep(2.5)
    return doc


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()
N = args.n

# ---- leaderboard top-N teams ----
with zipfile.ZipFile(LB_ZIP) as z:
    rows = list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode("utf-8"))))
gold = [(int(r[0]), int(r[1]), r[2], float(r[4])) for r in rows[1:GOLD + 1]]

# ---- teamId -> its final submissions (from any response's global list) ----
seed = call(OUR_SUBS[0])
by_team = {}
for s in seed.get("submissions", []):
    by_team.setdefault(s["teamId"], []).append(s)
for sid in OUR_SUBS:                       # the queried submission is omitted
    by_team.setdefault(16425135, [])
    if not any(s["id"] == sid for s in by_team[16425135]):
        by_team[16425135].append({"id": sid, "dateSubmitted": "z"})

out = []
for rank, tid, name, lb_score in gold:
    subs = sorted(by_team.get(tid, []), key=lambda s: s["dateSubmitted"])[-2:]
    games = []          # (opp score, reward, rating delta this game)
    for s in subs:
        doc = call(s["id"])
        for e in doc.get("episodes", []):
            if e.get("state") != "COMPLETED":
                continue
            me = next((a for a in e["agents"] if a.get("submissionId") == s["id"]), None)
            op = next((a for a in e["agents"] if a.get("submissionId") != s["id"]), None)
            if (me is None or op is None or me.get("reward") is None
                    or op.get("initialScore") is None
                    or me.get("initialScore") is None
                    or me.get("updatedScore") is None):
                continue
            games.append((op["initialScore"], me["reward"],
                          me["updatedScore"] - me["initialScore"]))
    if not games:
        print(f"  !! {name}: no episodes")
        continue
    games.sort()
    low = games[:N]
    w = sum(1 for _, r, _x in low if r > 0)
    d = sum(1 for _, r, _x in low if r == 0)
    tot_w = sum(1 for _, r, _x in games if r > 0)
    tot_d = sum(1 for _, r, _x in games if r == 0)
    # fixed-threshold variant: everyone's cut sits at the same absolute score,
    # so a team whose low-N reaches up into its Elo band isn't penalised
    sub950 = [g for g in games if g[0] < 950]
    w9 = sum(1 for _, r, _x in sub950 if r > 0)
    d9 = sum(1 for _, r, _x in sub950 if r == 0)
    net_lo = sum(g[2] for g in sub950)
    net_hi = sum(g[2] for g in games if g[0] >= 950)
    out.append({
        "rank": rank, "team": name, "lb": lb_score, "subs": len(subs),
        "n_all": len(games), "wr_all": round((tot_w + .5 * tot_d) / len(games) * 100, 1),
        "n_low": len(low), "w": w, "d": d, "l": len(low) - w - d,
        "wr_low": round((w + .5 * d) / len(low) * 100, 1),
        "opp_max": round(low[-1][0], 1), "opp_med": round(low[len(low) // 2][0], 1),
        "n_950": len(sub950), "w950": w9, "d950": d9, "l950": len(sub950) - w9 - d9,
        "wr_950": round((w9 + .5 * d9) / len(sub950) * 100, 1) if sub950 else None,
        "net_lo": round(net_lo, 1), "net_hi": round(net_hi, 1),
        "net": round(net_lo + net_hi, 1),
    })
    r = out[-1]
    print(f"{rank:3} {name[:26]:26} lb={lb_score:7.1f}  all {r['n_all']:4} "
          f"{r['wr_all']:5.1f}%   low{N} {r['w']:3}-{r['d']}-{r['l']:3} "
          f"{r['wr_low']:5.1f}%  opp<={r['opp_max']:.0f}   "
          f"<950: {r['n_950']:3} 局 {r['wr_950']:5.1f}% net {r['net_lo']:+7.1f}")

out.sort(key=lambda r: -r["wr_950"])
print(f"\n=== gold zone ranked by win rate vs opponents rated below 950 ===")
print(f"{'#':>3} {'rank':>4} {'team':26} {'lb':>7} | {'n<950':>6} {'win%':>7} "
      f"| {'low'+str(N):>7} {'win%':>7} {'opp<=':>6} | {'overall%':>9}")
for i, r in enumerate(out, 1):
    print(f"{i:3} {r['rank']:4} {r['team'][:26]:26} {r['lb']:7.1f} | "
          f"{r['n_950']:6} {r['wr_950']:6.1f}% | {r['n_low']:7} {r['wr_low']:6.1f}% "
          f"{r['opp_max']:6.0f} | {r['wr_all']:8.1f}%")

# how much of the ladder's overall win rate is the free bucket worth?
print()
lo = [r["wr_950"] for r in out]
print(f"vs <950: best {max(lo)}%  worst {min(lo)}%  spread {max(lo)-min(lo):.1f}pp")
alw = [r["wr_all"] for r in out]
print(f"overall: best {max(alw)}%  worst {min(alw)}%  spread {max(alw)-min(alw):.1f}pp")
n = len(out)
mx = sum(r["lb"] for r in out) / n
my = sum(r["wr_950"] for r in out) / n
cov = sum((r["lb"] - mx) * (r["wr_950"] - my) for r in out)
sx = sum((r["lb"] - mx) ** 2 for r in out) ** .5
sy = sum((r["wr_950"] - my) ** 2 for r in out) ** .5
print(f"corr(final score, win% vs <950) = {cov / (sx * sy):+.2f}")

print("\n=== rating actually banked in each bucket (same window) ===")
print(f"{'rank':>4} {'team':26} {'n<950':>6} {'win%':>7} {'net<950':>9} "
      f"{'n>=950':>7} {'net>=950':>9} {'total':>8}")
for r in sorted(out, key=lambda x: -x["net_lo"]):
    print(f"{r['rank']:4} {r['team'][:26]:26} {r['n_950']:6} {r['wr_950']:6.1f}% "
          f"{r['net_lo']:+9.1f} {r['n_all']-r['n_950']:7} {r['net_hi']:+9.1f} "
          f"{r['net']:+8.1f}")

with open(os.path.join(ROOT, "build", "gold_random_wr.json"), "w",
          encoding="utf-8") as f:
    json.dump({"n": N, "rows": out}, f, ensure_ascii=False)
print("\nwrote build/gold_random_wr.json")
