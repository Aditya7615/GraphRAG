import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from features.data_utils import load_feature_data, classify_player

CLEANED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ipl_cleaned.csv')
RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ipl_batting_fielding_stats.csv')


def build_name_map():
    raw = pd.read_csv(RAW_PATH)
    raw['player_name'] = raw['player_name'].str.strip()
    cleaned = pd.read_csv(CLEANED_PATH)
    name_map = {}
    for pid in cleaned['player_id'].unique():
        c_rows = cleaned[cleaned['player_id'] == pid].reset_index(drop=True)
        r_rows = raw.merge(c_rows[['season', 'matches', 'runs']], on=['season', 'matches', 'runs'], how='inner')
        if len(r_rows) > 0:
            name_map[pid] = r_rows['player_name'].iloc[0]
    return name_map


def lookup(player_id):
    name_map = build_name_map()
    features = load_feature_data()
    player_data = features[features['player_id'] == player_id]

    if player_data.empty:
        print(f"No data found for player_id={player_id}")
        total = features['player_id'].nunique()
        print(f"Valid player IDs: 1 to {total}")
        return

    name = name_map.get(player_id, f"Unknown (ID={player_id})")
    role = classify_player(player_data.iloc[0])

    career = player_data.iloc[0]
    seasons = sorted(player_data['season'].unique())

    sep = "=" * 60
    thin = "-" * 60

    print(f"\n{sep}")
    print(f"  PLAYER: {name}")
    print(f"  ID: {player_id}  |  Role: {role.upper()}")
    print(f"{sep}\n")

    # Career summary
    print(f"  CAREER SUMMARY ({seasons[0]} - {seasons[-1]}, {len(seasons)} seasons)")
    print(f"  {thin}")
    print(f"  {'Matches':<12} {'Innings':<12} {'Runs':<12} {'Wickets':<12}")
    print(f"  {int(career['career_matches']):<12} {int(career['career_innings']):<12} {int(career['career_runs']):<12} {int(career['career_wickets']):<12}")
    print()
    print(f"  {'50s':<12} {'100s':<12} {'Catches':<12} {'Seasons':<12}")
    print(f"  {int(career['career_fifties']):<12} {int(career['career_hundreds']):<12} {int(career['career_catches']):<12} {int(career['seasons_played']):<12}")
    print()

    # Performance scores
    print(f"  PERFORMANCE SCORES")
    print(f"  {thin}")
    print(f"  Batting Impact:       {career['batting_impact']:.4f}")
    print(f"  Bowling Impact:       {career['bowling_impact']:.4f}")
    print(f"  Consistency Score:    {career['consistency_score']:.4f}")
    print(f"  Overall Score:        {career['overall_performance_score']:.4f}")
    print(f"  Category:             {career['performance_category']}")
    print()

    # Season-by-season stats
    print(f"  SEASON-BY-SEASON BREAKDOWN")
    print(f"  {thin}")
    header = f"  {'Season':<8} {'Mat':<5} {'Inn':<5} {'Runs':<7} {'Avg':<8} {'SR':<9} {'4s':<5} {'6s':<5} {'50s':<5} {'100s':<5} {'Wkts':<6} {'Econ':<8}"
    print(header)
    print(f"  {'-' * len(header.strip())}")

    for _, row in player_data.sort_values('season').iterrows():
        hs = row.get('highest_score', '-')
        wkts = int(row['wickets'])
        econ = f"{row['economy_rate']:.2f}" if wkts > 0 else "-"
        bowl_avg = f"{row['bowling_average']:.2f}" if wkts > 0 else "-"
        print(
            f"  {int(row['season']):<8} "
            f"{int(row['matches']):<5} {int(row['innings']):<5} "
            f"{int(row['runs']):<7} {row['batting_average']:<8.2f} {row['strike_rate']:<9.2f} "
            f"{int(row['fours']):<5} {int(row['sixes']):<5} "
            f"{int(row['fifties']):<5} {int(row['hundreds']):<5} "
            f"{wkts:<6} {econ:<8}"
        )

    print(f"\n{sep}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        features = load_feature_data()
        total = features['player_id'].nunique()
        print(f"Usage: python lookup_player.py <player_id>")
        print(f"Player IDs range from 1 to {total}")
        print(f"\nListing all players:")
        name_map = build_name_map()
        for pid in sorted(name_map.keys()):
            print(f"  {pid:>4}  {name_map[pid]}")
        sys.exit(0)

    pid = int(sys.argv[1])
    lookup(pid)
