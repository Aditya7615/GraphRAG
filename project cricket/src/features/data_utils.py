import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ipl_features.csv')

ALL_FEATURES = [
    'runs', 'wickets', 'matches',
    'batting_average', 'strike_rate', 'fours', 'sixes',
    'fifties', 'hundreds', 'catches',
    'bowling_average', 'economy_rate', 'bowling_strike_rate',
    'career_matches', 'seasons_played',
    'career_fifties', 'career_hundreds', 'career_catches',
]

BAT_FEATURES = [
    'runs', 'matches',
    'batting_average', 'strike_rate', 'fours', 'sixes',
    'fifties', 'hundreds', 'catches',
    'career_matches', 'seasons_played',
    'career_fifties', 'career_hundreds', 'career_catches',
]

BOWL_FEATURES = [
    'wickets', 'matches',
    'bowling_average', 'economy_rate', 'bowling_strike_rate',
    'catches', 'career_matches', 'seasons_played',
    'career_catches',
]

AR_FEATURES = ALL_FEATURES


def load_feature_data(path=None):
    if path is None:
        path = FEATURES_PATH
    return pd.read_csv(path)


def get_feature_columns(df):
    exclude = [
        'player_id', 'season', 'highest_score', 'best_bowling',
        'performance_category', '_outlier',
        'total_runs', 'total_wickets', 'batting_impact', 'bowling_impact',
        'consistency_score', 'overall_performance_score',
        'highest_score_num', 'highest_score_notout',
        'best_bowling_wickets', 'best_bowling_runs',
        'innings',
        'career_runs', 'career_wickets',
        'career_innings',
    ]
    return [c for c in df.columns if c not in exclude and df[c].dtype in (np.int64, np.float64)]


def classify_player(row):
    career_runs = row.get('career_runs', row.get('total_runs', 0))
    career_wickets = row.get('career_wickets', row.get('total_wickets', 0))
    if pd.isna(career_runs): career_runs = 0
    if pd.isna(career_wickets): career_wickets = 0
    if career_runs == 0 and career_wickets == 0:
        season_runs = row.get('runs', 0)
        season_wkts = row.get('wickets', 0)
        if pd.isna(season_runs): season_runs = 0
        if pd.isna(season_wkts): season_wkts = 0
        career_runs = season_runs
        career_wickets = season_wkts
    runs_weight = career_runs / (career_runs + career_wickets * 20 + 1)
    wkts_weight = career_wickets * 20 / (career_runs + career_wickets * 20 + 1)
    if runs_weight >= 0.7:
        return 'batsman'
    elif wkts_weight >= 0.7:
        return 'bowler'
    else:
        return 'all_rounder'


FEATURE_SETS = {
    'batsman': BAT_FEATURES,
    'bowler': BOWL_FEATURES,
    'all_rounder': AR_FEATURES,
}


def prepare_data(df, feat_cols=None, test_size=0.2, random_state=42):
    if feat_cols is None:
        feat_cols = get_feature_columns(df)
    X = df[feat_cols].fillna(0)
    y = df['overall_performance_score'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, feat_cols, X.columns.tolist()


def prepare_role_data(df, role, test_size=0.2, random_state=42):
    df_role = df.copy()
    df_role['player_role'] = df_role.apply(classify_player, axis=1)
    df_role = df_role[df_role['player_role'] == role]
    feat_cols = FEATURE_SETS[role]
    if len(df_role) == 0:
        return None, None, None, None, feat_cols
    X = df_role[feat_cols].fillna(0)
    y = df_role['overall_performance_score'].values
    if len(X) < 10:
        return None, None, None, None, feat_cols
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, feat_cols


def get_features_for_role(role):
    return FEATURE_SETS.get(role, ALL_FEATURES)


# ── Feature Engineering ────────────────────────────────────────────

def _log_scale(values, eps=1e-8):
    v = np.maximum(values, 0)
    logged = np.log10(v + 1)
    max_log = logged.max()
    if max_log > 0:
        return np.clip(logged / max_log * 10, 0, 10)
    return np.zeros_like(values)


def compute_batting_impact(df):
    runs = df['runs'].values
    avg = df['batting_average'].values
    sr = df['strike_rate'].values
    raw = runs * np.maximum(avg, 0) * np.maximum(sr, 0) / 10000.0
    return _log_scale(raw)


def compute_bowling_impact(df):
    mask = df['wickets'].values > 0
    impact = np.zeros(len(df), dtype=float)
    if mask.any():
        w = df.loc[mask, 'wickets'].values
        ba = df.loc[mask, 'bowling_average'].values
        er = df.loc[mask, 'economy_rate'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = w / (np.maximum(ba, 0.1) * np.maximum(er, 0.1)) * 100
        impact[mask] = _log_scale(raw)
    return impact


def compute_consistency_score(df):
    player_stats = df.groupby('player_id').agg(
        seasons_active=('season', 'nunique'),
        total_matches=('matches', 'sum'),
        avg_runs_per_season=('runs', 'mean'),
        std_runs=('runs', 'std'),
        avg_sr=('strike_rate', 'mean'),
        std_sr=('strike_rate', 'std'),
        avg_ba=('batting_average', 'mean'),
        std_ba=('batting_average', 'std'),
    ).reset_index()
    player_stats['std_runs'] = player_stats['std_runs'].fillna(0)
    player_stats['std_sr'] = player_stats['std_sr'].fillna(0)
    player_stats['std_ba'] = player_stats['std_ba'].fillna(0)

    eps = 1e-8
    cv_runs = player_stats['std_runs'] / (player_stats['avg_runs_per_season'] + eps)
    cv_sr = player_stats['std_sr'] / (player_stats['avg_sr'] + eps)
    cv_ba = player_stats['std_ba'] / (player_stats['avg_ba'] + eps)

    stability = 1 / (1 + cv_runs + cv_sr + cv_ba)

    season_score = _log_scale(player_stats['seasons_active'].values)
    match_score = _log_scale(player_stats['total_matches'].values)
    stability_score = _log_scale(stability.values)

    player_stats['consistency_score'] = np.clip(
        0.20 * season_score + 0.20 * match_score + 0.60 * stability_score, 0, 10
    )

    df = df.merge(player_stats[['player_id', 'consistency_score']], on='player_id', how='left')
    return df


TARGET_ANCHORS = np.array([
    (0.0, 0.0),
    (0.01, 0.5),
    (0.05, 1.0),
    (0.10, 1.6),
    (0.20, 2.2),
    (0.25, 2.5),
    (0.30, 2.8),
    (0.40, 3.3),
    (0.50, 4.0),
    (0.60, 4.7),
    (0.70, 5.5),
    (0.80, 6.2),
    (0.90, 7.0),
    (0.95, 7.8),
    (0.99, 8.5),
    (0.999, 9.5),
    (1.0, 10.0),
])


def compute_performance_score(df):
    bi = df['batting_impact'].values
    bowi = df['bowling_impact'].values
    cons = df['consistency_score'].values

    raw_score = 0.45 * bi + 0.25 * bowi + 0.30 * cons

    n = len(raw_score)
    ranks = np.argsort(np.argsort(raw_score))
    percentiles = ranks / (n - 1)

    target_scores = np.interp(percentiles, TARGET_ANCHORS[:, 0], TARGET_ANCHORS[:, 1])

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.08, n)
    df['overall_performance_score'] = np.clip(target_scores + noise, 0, 10).round(4)
    return df


def categorize(df):
    def cat(s):
        if s >= 8.5: return 'Excellent'
        elif s >= 7.0: return 'Very Good'
        elif s >= 5.5: return 'Good'
        elif s >= 4.0: return 'Average'
        elif s >= 2.5: return 'Below Average'
        else: return 'Poor'
    df['performance_category'] = df['overall_performance_score'].apply(cat)
    return df


def add_career_features(df):
    career = df.groupby('player_id').agg(
        career_matches=('matches', 'sum'),
        career_innings=('innings', 'sum'),
        career_runs=('runs', 'sum'),
        career_wickets=('wickets', 'sum'),
        career_fifties=('fifties', 'sum'),
        career_hundreds=('hundreds', 'sum'),
        career_catches=('catches', 'sum'),
        seasons_played=('season', 'nunique')
    ).reset_index()
    return df.merge(career, on='player_id', how='left')


def engineer_features(df):
    df = df[df['_outlier'] == 0].copy()
    df['batting_impact'] = compute_batting_impact(df)
    df['bowling_impact'] = compute_bowling_impact(df)
    df = compute_consistency_score(df)
    df = compute_performance_score(df)
    df = categorize(df)
    df = add_career_features(df)
    return df
