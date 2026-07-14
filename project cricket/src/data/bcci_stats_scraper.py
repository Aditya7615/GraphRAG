import re, time, os
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.bcci.tv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
}

CATEGORIES = {
    "batting": [
        ("batting_most_runs", "most_runs"),
        ("batting_most_runs_innings", "highest_score"),
        ("batting_highest_average", "highest_average"),
        ("batting_most_run50", "most_fifties"),
        ("batting_most_run100", "most_hundreds"),
        ("batting_highest_strikerate", "highest_strike_rate"),
        ("batting_most_run4", "most_fours"),
        ("batting_most_run6", "most_sixes"),
    ],
    "bowling": [
        ("bowling_top_wicket_takers", "most_wickets"),
        ("bowling_best_averages", "best_average"),
        ("bowling_best_bowling_figures", "best_bowling_figures"),
        ("bowling_best_economy_rates", "best_economy"),
        ("bowling_best_strike_rates", "best_strike_rate"),
    ],
}
FORMATS = ["test", "odi", "t20i"]


def get_csrf():
    html = requests.get(f"{BASE}/international/men/stats/test", headers=HEADERS, timeout=15).text
    m = re.search(r'csrf_token\s*=\s*"([^"]+)"', html)
    return m.group(1) if m else ""


def fetch(fmt, slug, stype, token):
    params = {"platform": "international", "type": "men", "s_type": stype, "slug": slug, "format": fmt, "_token": token}
    h = {**HEADERS, "X-Requested-With": "XMLHttpRequest", "Accept": "application/json"}
    r = requests.get(f"{BASE}/getStats", headers=h, params=params, timeout=15)
    d = r.json()
    return d.get("html") if d.get("status") else None


def parse(html):
    rows = []
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 2:
        return rows
    for tr in tables[1].find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        row = {"rank": cells[0].get_text(strip=True), "player_name": cells[1].get_text(strip=True)}
        for c in cells[2:]:
            h6, sp = c.find("h6"), c.find("span")
            if h6 and sp:
                row[sp.get_text(strip=True).lower()] = h6.get_text(strip=True)
        rows.append(row)
    return rows


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "bcci_stats_rankings_all.csv")

    print("BCCI Stats Scraper")
    print("=" * 35)

    token = get_csrf()
    if not token:
        print("ERROR: could not get CSRF token")
        return

    all_rows = []
    for fmt in FORMATS:
        for stype, cats in CATEGORIES.items():
            for slug, label in cats:
                print(f"  {fmt.upper()} {stype}/{label}...", end=" ", flush=True)
                html = fetch(fmt, slug, stype, token)
                if html:
                    entries = parse(html)
                    for e in entries:
                        row = {
                            "format": fmt.upper(),
                            "stat_category": f"{stype}_{label}",
                            "player_name": e["player_name"],
                            "rank": e.get("rank", ""),
                        }
                        for k, v in e.items():
                            if k not in ("rank", "player_name"):
                                row[f"{stype}_{k}"] = v
                        all_rows.append(row)
                    print(f"{len(entries)} players")
                else:
                    print("no data")
                time.sleep(0.3)

    df = pd.DataFrame(all_rows).fillna("")
    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} — {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
