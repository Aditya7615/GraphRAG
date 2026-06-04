
import json, re, time
import pandas as pd
import requests

BASE = "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/stats"
HEADERS = {"User-Agent": "Mozilla/5.0"}

SEASONS = [
    (10000, 2008), (10001, 2009), (10002, 2010), (10003, 2011),
    (10004, 2012), (10005, 2013), (10006, 2014), (10007, 2015),
    (10008, 2016), (10009, 2017), (10010, 2018), (10011, 2019),
    (10012, 2020), (10013, 2021),
    (60, 2022), (107, 2023), (148, 2024), (203, 2025), (284, 2026),
]

def n(v, t=float):
    if v is None: return 0
    s = str(v).strip()
    if s in ("", "-", "--"): return 0
    try: return t(s)
    except: return 0

def load(url, cb):
    r = requests.get(url, headers=HEADERS, timeout=30)
    m = re.search(rf'{re.escape(cb)}\((.+)\);', r.text, re.DOTALL)
    return json.loads(m.group(1)) if m else None

all_rows = []
for cid, yr in SEASONS:
    print(f"{yr}...", end=" ", flush=True)
    try:
        bat = load(f"{BASE}/{cid}-toprunsscorers.js", "ontoprunsscorers")
        bol = load(f"{BASE}/{cid}-mostwickets.js", "onmostwickets")

        be = (bat.get("toprunsscorers") or bat.get("topRunScorers") or list(bat.values())[0]) if isinstance(bat, dict) else (bat or [])
        bole = (bol.get("mostwickets") or bol.get("mostWickets") or list(bol.values())[0]) if isinstance(bol, dict) else (bol or [])

        bm = {}
        for e in be if isinstance(be, list) else []:
            nm = (e.get("StrikerName") or "").strip()
            if not nm: continue
            bm[nm] = {
                "matches": n(e.get("Matches"), int), "innings": n(e.get("Innings"), int),
                "runs": n(e.get("TotalRuns"), int), "strike_rate": n(e.get("StrikeRate")),
                "fours": n(e.get("Fours"), int), "sixes": n(e.get("Sixes"), int),
                "fifties": n(e.get("FiftyPlusRuns"), int), "hundreds": n(e.get("Centuries"), int),
                "highest_score": e.get("HighestScore", ""), "batting_average": n(e.get("BattingAverage")),
                "catches": n(e.get("Catches"), int),
            }

        wm = {}
        for e in bole if isinstance(bole, list) else []:
            nm = (e.get("BowlerName") or e.get("PlayerName") or "").strip()
            if not nm: continue
            wm[nm] = {
                "wickets": n(e.get("Wickets"), int),
                "bowling_average": n(e.get("BowlingAverage") or e.get("Average")),
                "economy_rate": n(e.get("EconomyRate")),
                "bowling_strike_rate": n(e.get("BowlingSR") or e.get("StrikeRate")),
                "best_bowling": e.get("BBIW") or e.get("BestBowlingFigure") or "",
            }

        seen = set()
        for nm in sorted(set(list(bm) + list(wm))):
            if f"{nm}_{yr}" in seen: continue
            seen.add(f"{nm}_{yr}")
            b, w = bm.get(nm, {}), wm.get(nm, {})
            all_rows.append({
                "player_name": nm, "season": yr,
                "matches": b.get("matches", 0), "innings": b.get("innings", 0),
                "runs": b.get("runs", 0), "batting_average": b.get("batting_average", 0),
                "strike_rate": b.get("strike_rate", 0), "fours": b.get("fours", 0),
                "sixes": b.get("sixes", 0), "fifties": b.get("fifties", 0),
                "hundreds": b.get("hundreds", 0), "highest_score": b.get("highest_score", ""),
                "catches": b.get("catches", 0),
                "wickets": w.get("wickets", 0), "bowling_average": w.get("bowling_average", 0),
                "economy_rate": w.get("economy_rate", 0),
                "bowling_strike_rate": w.get("bowling_strike_rate", 0),
                "best_bowling": w.get("best_bowling", ""),
            })
        print(f"{len(seen)} players")
    except Exception as e:
        print(f"ERR: {e}")
    time.sleep(0.3)

df = pd.DataFrame(all_rows).sort_values(["player_name", "season"]).reset_index(drop=True)
df.to_csv("ipl_batting_fielding_stats.csv", index=False)
print(f"\nDone — {len(df)} rows, {len(df.columns)} columns -> ipl_batting_fielding_stats.csv")
