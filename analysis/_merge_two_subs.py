"""Merge the ladder-score trajectories of both final submissions.

ListEpisodes returns only the most recent 1000 episodes per submission, so the
run's opening days are gone from a post-hoc fetch. Reconstruct the full curve
by unioning (on episode id) every snapshot we captured while the ladder ran:

    build/ep_cache/<sub>.json      08-15 .. 08-21 09:11
    build/two_subs_final.json      (raw fetch, 08-21 21:23 .. 08-31 23:49)

Output: build/two_subs_merged.json -- per submission, the ordered per-episode
updatedScore series plus W/D/L and the size of any residual coverage gap.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
SUBS = {55535786: "A", 55554179: "B"}


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00").split(".")[0]
                                  + "+00:00" if "." in ts
                                  else ts.replace("Z", "+00:00"))


def live_fetch(sub_id):
    req = urllib.request.Request(
        "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
        data=json.dumps({"submissionId": sub_id}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def episodes(doc, sub_id):
    """episode id -> (endTime, createTime, updatedScore, reward)"""
    out = {}
    for e in doc.get("episodes", []):
        if e.get("state") != "COMPLETED":
            continue
        ag = next((a for a in e["agents"] if a.get("submissionId") == sub_id),
                  None)
        if ag is None or ag.get("updatedScore") is None:
            continue
        out[e["id"]] = (e["endTime"], e["createTime"], ag["updatedScore"],
                        ag.get("reward"))
    return out


result = {}
for sid, tag in SUBS.items():
    merged = {}
    src = os.path.join(BUILD, "ep_cache", f"{sid}.json")
    with open(src, encoding="utf-8") as f:
        merged.update(episodes(json.load(f), sid))
    n_cache = len(merged)
    merged.update(episodes(live_fetch(sid), sid))

    pts = sorted(merged.values())
    # residual gap: largest gap in episode end times
    times = [parse(p[0]) for p in pts]
    gaps = sorted(((times[i + 1] - times[i]).total_seconds() / 3600, i)
                  for i in range(len(times) - 1))
    big_h, big_i = gaps[-1]

    wins = sum(1 for p in pts if (p[3] or 0) > 0)
    draws = sum(1 for p in pts if (p[3] or 0) == 0)
    losses = len(pts) - wins - draws
    peak_i = max(range(len(pts)), key=lambda i: pts[i][2])
    result[tag] = {
        "sub_id": sid, "n": len(pts), "n_from_cache": n_cache,
        "w": wins, "d": draws, "l": losses,
        "wr": round(wins / max(1, wins + losses) * 100, 1),
        "start": pts[0][0][:16], "end": pts[-1][0][:16],
        "final": round(pts[-1][2], 1),
        "peak": round(pts[peak_i][2], 1), "peak_at": peak_i,
        "peak_ts": pts[peak_i][0][:16],
        "min": round(min(p[2] for p in pts), 1),
        "gap_hours": round(big_h, 1),
        "gap_at": [pts[big_i][0][:16], pts[big_i + 1][0][:16]],
        "gap_score_jump": round(pts[big_i + 1][2] - pts[big_i][2], 1),
        "series": [[i, round(p[2], 1), p[0][5:16].replace("T", " ")]
                   for i, p in enumerate(pts)],
    }
    r = result[tag]
    print(f"sub {tag} ({sid}): n={r['n']} (cache {n_cache} + live) "
          f"W-D-L {wins}-{draws}-{losses} wr={r['wr']}%")
    print(f"  {r['start']} -> {r['end']}  final={r['final']} "
          f"peak={r['peak']}@ep{r['peak_at']} min={r['min']}")
    print(f"  largest gap {r['gap_hours']}h at {r['gap_at']} "
          f"(score moved {r['gap_score_jump']:+})")

with open(os.path.join(BUILD, "two_subs_merged.json"), "w",
          encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print("wrote build/two_subs_merged.json")
