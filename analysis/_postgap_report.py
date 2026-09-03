"""Post-gap analysis window: the last 1000 ladder episodes of each submission.

Both final submissions ship the same bundle (hydra15038_final.tar.gz), so their
games are pooled. Restricting to what ListEpisodes still returns (the most
recent 1000 per submission, 08-21 20:38 .. 08-31 23:53) buys three things at
once: no placement games, no 08-21 coverage gap, and a fully continuous trace.

Two splits on top of that:

* MATCHMAKING. The ladder pairs by Elo 90% of the time and draws a uniformly
  random leaderboard opponent the other 10%. The rating gap separates them
  cleanly: |our score - opp score| decays geometrically out to ~200 (the Elo
  kernel) and then goes flat all the way past 600 (the uniform draw). The cut
  at 200 is validated three ways -- it lands on exactly 10.0% of games, it
  agrees with an independent "opponent below 950" cut (10.1%), and the random
  bucket's score distribution reproduces the leaderboard's own.

* DECKLIST. Opponent decks are resolved at the SUBMISSION level (a team's two
  submissions often run different decks) by intersecting every cached
  ListEpisodes doc (build/ep_cache*/, per episode both agents' submissionId)
  with data/meta/08{17,18}.jsonl (per episode both sides' team name and
  60-card list). Every archetype is then broken down by exact decklist, so a
  bucket like kangaskhan_box or other is not read as one matchup.

    python build/_postgap_report.py
"""
import glob
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from replay_loader import classify_deck, deck_fingerprint  # noqa: E402

SUBS = {55535786: "A", 55554179: "B"}
MAP_DAYS = ["0817", "0818"]
OUR_TEAM = "Rmy"
RANDOM_GAP = 200.0        # |our score - opp score| above this => random draw
SUBLIST_MIN = 8           # decklists below this stay folded into their archetype


def fetch(sub_id):
    req = urllib.request.Request(
        "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
        data=json.dumps({"submissionId": sub_id}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def P(ts):
    return datetime.fromisoformat(ts.replace("Z", "").split(".")[0]
                                  ).replace(tzinfo=timezone.utc)


# Cyrillic letters that render identically to Latin ones. Teams rename
# themselves mid-competition and some mix the two alphabets, so the episode
# API and the replay's TeamNames can carry different spellings of the same
# name -- "Аzat Аkhtyamоv" (Cyrillic А, о) vs "Azat Akhtyamov" is one team,
# and an exact string match silently dropped all 63 of its games into the
# unresolved bucket.
_HOMOGLYPH = str.maketrans("АВЕКМНОРСТХаеорсух", "ABEKMHOPCTXaeopcyx")


def norm_name(s):
    return (s or "").translate(_HOMOGLYPH).casefold().strip()


CARDS = {c["id"]: c for c in
         json.load(open(os.path.join(ROOT, "data", "card_texts.json"),
                        encoding="utf-8"))["cards"]}


def deck_name(deck):
    """Name a decklist after the ex / Mega-ex Pokemon it runs."""
    cnt = Counter(deck)
    exs = [(n, CARDS[c]["hp"], CARDS[c]["name"], CARDS[c].get("megaEx"))
           for c, n in cnt.items()
           if c in CARDS and CARDS[c]["cardType"] == 0
           and (CARDS[c].get("ex") or CARDS[c].get("megaEx"))]
    if not exs:
        exs = sorted([(n, CARDS[c]["hp"], CARDS[c]["name"], False)
                      for c, n in cnt.items()
                      if c in CARDS and CARDS[c]["cardType"] == 0],
                     key=lambda t: (-t[1], -t[0]))[:1]
    exs.sort(key=lambda t: (-t[0], -t[1]))
    return " + ".join(("Mega " if m and not nm.startswith(("Mega", "M "))
                       else "") + nm for _, _, nm, m in exs[:2]) or "unnamed"


CARD_ID = {c["name"]: c["id"] for c in CARDS.values()}


def has(deck, name):
    return CARD_ID.get(name) in deck


def stadium(deck):
    """The stadium a list leans on: highest count among its stadium cards."""
    st = Counter(c for c in deck if CARDS.get(c, {}).get("cardType") == 4)
    return CARDS[st.most_common(1)[0][0]]["name"] if st else None


# Named sub-archetypes for the two buckets that are too big to read as one
# matchup. Priority-ordered, first match wins -- a secondary attacker line is a
# bigger structural difference than a stadium or an ACE SPEC, so it goes first.
# dragapult: every list runs the same ACE SPEC (Unfair Stamp), so the stadium is
# what separates the builds. kangaskhan_box: the ACE SPEC does, cleanly -- each
# list runs exactly one of Prime Catcher / Secret Box.
VARIANTS = {
    "dragapult": [
        ("dragapult_blaziken", "多龙巴鲁托 + 火焰鸡",
         lambda d: has(d, "Blaziken ex")),
        ("dragapult_dusknoir", "多龙巴鲁托 + 黑夜魔灵",
         lambda d: has(d, "Dusknoir")),
        ("dragapult_jamming", "多龙巴鲁托 · 干扰塔型",
         lambda d: stadium(d) == "Jamming Tower"),
        ("dragapult_ruins", "多龙巴鲁托 · 危险遗迹型",
         lambda d: stadium(d) == "Risky Ruins"),
        ("dragapult_dudunsparce", "多龙巴鲁托 + 土龙节节 ex",
         lambda d: has(d, "Dudunsparce ex")),
        ("dragapult_watchtower", "多龙巴鲁托 · 火箭队瞭望塔型",
         lambda d: stadium(d) == "Team Rocket's Watchtower"),
    ],
    "kangaskhan_box": [
        ("kanga_megabox", "超级混合 box",
         lambda d: has(d, "Mega Heracross ex") or has(d, "Mega Audino ex")),
        ("kanga_ogerpon", "超级袋兽 + 厄诡椪",
         lambda d: has(d, "Teal Mask Ogerpon ex")),
        ("kanga_secretbox", "超级袋兽 box · 秘密盒型",
         lambda d: has(d, "Secret Box")),
        ("kanga_primecatcher", "超级袋兽 box · 究极捕获器型",
         lambda d: has(d, "Prime Catcher")),
        ("kanga_unfairstamp", "超级袋兽 box · 不公印章型",
         lambda d: has(d, "Unfair Stamp")),
    ],
}
VARIANT_CN = {k: cn for rules in VARIANTS.values() for k, cn, _ in rules}


def variant(arch, deck):
    """archetype -> finer named bucket, or the archetype itself."""
    for key, _cn, test in VARIANTS.get(arch, []):
        if test(set(deck)):
            return key
    return arch + "_misc" if arch in VARIANTS else arch


def wilson(w, n, z=1.96):
    if not n:
        return None, None
    p, den = w / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / den
    return round((c - hw) * 100, 1), round((c + hw) * 100, 1)


# ---- 1. replay ground truth ----
ep_meta = {}
for day in MAP_DAYS:
    with open(os.path.join(ROOT, "data", "meta", f"{day}.jsonl"), "rb") as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or not r.get("deck0") or not r.get("deck1"):
                continue
            names = r["teams"]
            if not names or None in names or names[0] == names[1]:
                continue
            ep_meta[str(r["ep"])] = ([norm_name(x) for x in names],
                                     [r["deck0"], r["deck1"]])
print(f"replay meta: {len(ep_meta)} episodes")

# ---- 2. submissionId -> decklist ----
sub_deck, fp_deck, team_name = defaultdict(Counter), {}, {}
CACHE_DIRS = ["ep_cache", "ep_cache2", "gold_eps", "opp_eps"]
for path in sorted(p for d in CACHE_DIRS
                   for p in glob.glob(os.path.join(ROOT, "build", d, "*.json"))):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    for t in doc.get("teams", []):
        team_name[t["id"]] = t["teamName"]
    for e in doc.get("episodes", []):
        hit = ep_meta.get(str(e["id"]))
        if not hit:
            continue
        names, decks = hit
        for ag in e["agents"]:
            nm = norm_name(team_name.get(ag.get("teamId")))
            if nm not in names:
                continue
            deck = decks[names.index(nm)]
            fp = deck_fingerprint(deck)
            fp_deck[fp] = deck
            sub_deck[ag["submissionId"]][fp] += 1

# ---- 3. our post-gap games ----
games, traces, sub_team = [], {}, {}
for sid, tag in SUBS.items():
    doc = fetch(sid)
    for s in doc.get("submissions", []):
        sub_team[s["id"]] = s["teamId"]
    for t in doc.get("teams", []):
        team_name[t["id"]] = t["teamName"]
    pts = []
    for e in doc.get("episodes", []):
        if e.get("state") != "COMPLETED":
            continue
        me = next((a for a in e["agents"] if a.get("submissionId") == sid), None)
        op = next((a for a in e["agents"] if a.get("submissionId") != sid), None)
        if me is None or op is None or me.get("updatedScore") is None:
            continue
        pts.append((P(e["endTime"]), me["updatedScore"], me.get("reward"),
                    op.get("submissionId"), me.get("initialScore"),
                    op.get("initialScore")))
        hit = ep_meta.get(str(e["id"]))
        if hit and norm_name(OUR_TEAM) in hit[0]:
            dk = hit[1][1 - hit[0].index(norm_name(OUR_TEAM))]
            fp_deck[deck_fingerprint(dk)] = dk
            sub_deck[op["submissionId"]][deck_fingerprint(dk)] += 1
    pts.sort()
    traces[tag] = pts
    w = sum(1 for p in pts if (p[2] or 0) > 0)
    dr = sum(1 for p in pts if (p[2] or 0) == 0)
    print(f"seat {tag} ({sid}): {len(pts)} games {pts[0][0]:%m-%d %H:%M} -> "
          f"{pts[-1][0]:%m-%d %H:%M}  W-D-L {w}-{dr}-{len(pts)-w-dr}  "
          f"score {pts[0][1]:.1f} -> {pts[-1][1]:.1f}")
    games.extend((p[3], p[2], p[4], p[5]) for p in pts)

SUB_FP = {s: c.most_common(1)[0][0] for s, c in sub_deck.items()}
print(f"submission->decklist map: {len(SUB_FP)} submissions, "
      f"{len(fp_deck)} distinct lists")

# ---- 4. matchmaking split ----
n = len(games)
rand = [g for g in games if g[2] and g[3] and abs(g[2] - g[3]) > RANDOM_GAP]
elo = [g for g in games if not (g[2] and g[3] and abs(g[2] - g[3]) > RANDOM_GAP)]
below = sum(1 for g in games if g[3] and g[3] < 950)
print(f"\nmatchmaking split at |delta|>{RANDOM_GAP:.0f}: "
      f"random {len(rand)} ({len(rand)/n*100:.1f}%)  elo {len(elo)} "
      f"({len(elo)/n*100:.1f}%)   [cross-check opp<950: {below} "
      f"({below/n*100:.1f}%)]")
for label, sub in (("random 10%", rand), ("elo 90%", elo)):
    w = sum(1 for g in sub if (g[1] or 0) > 0)
    d = sum(1 for g in sub if g[1] == 0)
    lo, hi = wilson(w + .5 * d, len(sub))
    opp = sorted(g[3] for g in sub if g[3])
    print(f"  {label:11} {len(sub):5} games  W-D-L {w}-{d}-{len(sub)-w-d}  "
          f"wr={(w+.5*d)/len(sub)*100:5.1f}% [{lo}, {hi}]  "
          f"opp median {opp[len(opp)//2]:.0f}")

# ---- 5. deck breakdown of the Elo-matched games ----
arch_agg = defaultdict(lambda: [0, 0, 0, 0])
var_agg = defaultdict(lambda: [0, 0, 0, 0])
list_agg = defaultdict(lambda: [0, 0, 0, 0])
var_of = {}
unresolved = 0
for opp_sub, reward, _, _ in elo:
    fp = SUB_FP.get(opp_sub)
    if fp is None:
        arch = var = "unknown"
        unresolved += 1
    else:
        arch = classify_deck(fp_deck[fp])
        var = variant(arch, fp_deck[fp])
        var_of[fp] = var
    slot = 1 if (reward or 0) > 0 else (2 if reward == 0 else 3)
    for c in (arch_agg[arch], var_agg[(arch, var)], list_agg[(arch, fp)]):
        c[0] += 1
        c[slot] += 1

print(f"\nElo-matched games by opponent archetype ({len(elo)} games, "
      f"{unresolved} unresolved):")
print(f"{'archetype':22} {'games':>6} {'W':>5} {'D':>3} {'L':>5} {'win%':>7} "
      f"{'95% CI':>16}")
out_arch, out_var, out_list = [], [], []
for a, (g, w, d, l) in sorted(arch_agg.items(), key=lambda kv: -kv[1][0]):
    lo, hi = wilson(w + .5 * d, g)
    print(f"{a:22} {g:6} {w:5} {d:3} {l:5} {(w+.5*d)/g*100:6.1f}% "
          f"{f'[{lo}, {hi}]':>16}")
    out_arch.append({"arch": a, "g": g, "w": w, "d": d, "l": l,
                     "wr": round((w + .5 * d) / g * 100, 1), "lo": lo, "hi": hi})
    for (aa, vk), (g1, w1, d1, l1) in sorted(
            ((k, v) for k, v in var_agg.items() if k[0] == a),
            key=lambda kv: -kv[1][0]):
        if vk == a:
            continue          # archetype not split into variants
        lo1, hi1 = wilson(w1 + .5 * d1, g1)
        print(f"  ▸ {VARIANT_CN.get(vk, vk):28} {vk[:22]:22} {g1:5} {w1:4} {l1:4} "
              f"{(w1+.5*d1)/g1*100:6.1f}%  [{lo1}, {hi1}]")
        out_var.append({"arch": a, "key": vk, "cn": VARIANT_CN.get(vk, "其他构筑"),
                        "g": g1, "w": w1, "d": d1, "l": l1,
                        "wr": round((w1 + .5 * d1) / g1 * 100, 1),
                        "lo": lo1, "hi": hi1})
    subs = sorted(((k[1], v) for k, v in list_agg.items() if k[0] == a),
                  key=lambda kv: -kv[1][0])
    if a == "unknown" or len(subs) < 2:
        continue
    for fp, (g2, w2, d2, l2) in subs:
        if g2 < SUBLIST_MIN:
            continue
        lo2, hi2 = wilson(w2 + .5 * d2, g2)
        nm = deck_name(fp_deck[fp])
        print(f"      ├ {nm[:30]:30} {fp[:8]} {g2:5} {w2:4} {l2:4} "
              f"{(w2+.5*d2)/g2*100:6.1f}%  [{lo2}, {hi2}]")
        out_list.append({"arch": a, "var": var_of.get(fp, a), "fp": fp, "name": nm,
                         "g": g2, "w": w2, "d": d2, "l": l2,
                         "wr": round((w2 + .5 * d2) / g2 * 100, 1),
                         "lo": lo2, "hi": hi2})
    tail = sum(v[0] for fp, v in subs if v[0] < SUBLIST_MIN)
    if tail:
        print(f"      └ (其余 {sum(1 for fp, v in subs if v[0] < SUBLIST_MIN)} "
              f"份牌表, {tail} 局)")

w = sum(1 for g in elo if (g[1] or 0) > 0)
d = sum(1 for g in elo if g[1] == 0)
lo, hi = wilson(w + .5 * d, len(elo))

T0 = min(traces["A"][0][0], traces["B"][0][0])
out = {
    "t0": T0.isoformat(), "n": n, "random_gap": RANDOM_GAP,
    "split": {"random": len(rand), "elo": len(elo), "below950": below},
    "elo_total": {"g": len(elo), "w": w, "d": d, "l": len(elo) - w - d,
                  "wr": round((w + .5 * d) / len(elo) * 100, 1), "lo": lo, "hi": hi},
    "arch": out_arch, "variants": out_var, "lists": out_list, "seats": {},
}
for label, sub in (("random", rand), ("elo", elo)):
    ww = sum(1 for g in sub if (g[1] or 0) > 0)
    dd = sum(1 for g in sub if g[1] == 0)
    l2, h2 = wilson(ww + .5 * dd, len(sub))
    opp = sorted(g[3] for g in sub if g[3])
    out[label + "_total"] = {"g": len(sub), "w": ww, "d": dd,
                             "l": len(sub) - ww - dd,
                             "wr": round((ww + .5 * dd) / len(sub) * 100, 1),
                             "lo": l2, "hi": h2,
                             "opp_median": round(opp[len(opp) // 2], 1)}
for tag, pts in traces.items():
    ww = sum(1 for p in pts if (p[2] or 0) > 0)
    dd = sum(1 for p in pts if p[2] == 0)
    out["seats"][tag] = {
        "n": len(pts), "w": ww, "d": dd, "l": len(pts) - ww - dd,
        "final": round(pts[-1][1], 1), "first": round(pts[0][1], 1),
        "min": round(min(p[1] for p in pts), 1),
        "max": round(max(p[1] for p in pts), 1),
        "series": [[round((p[0] - T0).total_seconds() / 3600, 3), round(p[1], 1)]
                   for p in pts]}
with open(os.path.join(ROOT, "build", "postgap.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("\nwrote build/postgap.json")
