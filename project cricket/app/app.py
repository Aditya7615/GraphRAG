import os, sys, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features.data_utils import load_feature_data, get_feature_columns, classify_player
from src.features.data_utils import FEATURE_SETS, ALL_FEATURES
from src.features.data_utils import compute_batting_impact, compute_bowling_impact
from src.features.data_utils import compute_consistency_score, compute_performance_score, categorize, add_career_features

RAW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ipl_batting_fielding_stats.csv'))
CLEANED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'ipl_cleaned.csv'))

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

st.set_page_config(
    page_title="CricPredict — Player Performance Analyzer",
    page_icon=":cricket_game:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    results_path = os.path.join(MODELS_DIR, 'all_models_results.pkl')
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    comp_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    if not os.path.exists(results_path):
        return None, None, None
    results = joblib.load(results_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    comp_df = None
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        if 'Model' not in comp_df.columns and 'Unnamed: 0' in comp_df.columns:
            comp_df.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)
    return results, metadata, comp_df


@st.cache_resource
def load_role_models():
    role_dir = os.path.join(MODELS_DIR, 'role_models')
    role_models = {}
    for role in ['batsman', 'bowler', 'all_rounder']:
        models = {}

        lgb_path = os.path.join(role_dir, role, 'lgb_model.pkl')
        meta_path = os.path.join(role_dir, role, 'metadata.json')
        if os.path.exists(lgb_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            models['LightGBM'] = {
                'model': joblib.load(lgb_path),
                'features': meta['features'],
                'r2_score': meta['r2_score'],
                'mae': meta['mae'],
                'rmse': meta['rmse'],
            }

        ens_path = os.path.join(role_dir, role, 'ensemble_model.pkl')
        ens_meta_path = os.path.join(role_dir, role, 'ensemble_metadata.json')
        if os.path.exists(ens_path) and os.path.exists(ens_meta_path):
            with open(ens_meta_path) as f:
                meta = json.load(f)
            models['Ensemble'] = {
                'model': joblib.load(ens_path),
                'features': meta['features'],
                'r2_score': meta['r2_score'],
                'mae': meta['mae'],
                'rmse': meta['rmse'],
            }

        xgb_path = os.path.join(role_dir, role, 'xgb_model.pkl')
        if os.path.exists(xgb_path):
            with open(meta_path) as f:
                meta = json.load(f)
            models['XGBoost'] = {
                'model': joblib.load(xgb_path),
                'features': meta['features'],
                'r2_score': meta.get('r2_score', 0),
                'mae': meta.get('mae', 0),
                'rmse': meta.get('rmse', 0),
            }

        if models:
            role_models[role] = models
    return role_models


@st.cache_resource
def load_data():
    try:
        return load_feature_data()
    except FileNotFoundError:
        return None


@st.cache_resource
def load_name_map():
    if not os.path.exists(RAW_PATH):
        return {}
    raw = pd.read_csv(RAW_PATH)
    raw['player_name'] = raw['player_name'].str.strip()
    cleaned_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'ipl_cleaned.csv'))
    if not os.path.exists(cleaned_path):
        return {}
    cleaned = pd.read_csv(cleaned_path)
    name_map = {}
    for pid in cleaned['player_id'].unique():
        c_rows = cleaned[cleaned['player_id'] == pid].reset_index(drop=True)
        r_rows = raw.merge(c_rows[['season', 'matches', 'runs']], on=['season', 'matches', 'runs'], how='inner')
        if len(r_rows) > 0:
            name_map[int(pid)] = r_rows['player_name'].iloc[0]
    return name_map


@st.cache_resource
def load_all_players():
    if not os.path.exists(CLEANED_PATH):
        return None
    df = pd.read_csv(CLEANED_PATH)
    df['batting_impact'] = compute_batting_impact(df)
    df['bowling_impact'] = compute_bowling_impact(df)
    df = compute_consistency_score(df)
    df = compute_performance_score(df)
    df = categorize(df)
    df = add_career_features(df)
    return df


results, metadata, comp_df = load_artifacts()
df_full = load_data()
role_models = load_role_models()

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main > div {
        padding: 1rem 1.5rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #0B1120 0%, #162033 50%, #0B1120 100%);
        border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;
        border: 1px solid rgba(56, 189, 248, 0.1);
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ''; position: absolute; top: -60%; left: -30%; width: 160%; height: 200%;
        background: radial-gradient(ellipse at 30% 50%, rgba(56, 189, 248, 0.04) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero h1 {
        color: #F0F6FC; font-size: 1.9rem; font-weight: 700;
        margin: 0 0 0.25rem 0; letter-spacing: -0.3px;
    }
    .hero h1 span { color: #38BDF8; }
    .hero .subtitle {
        color: rgba(240, 246, 252, 0.5); font-size: 0.9rem;
        font-weight: 400; margin: 0; line-height: 1.5;
    }
    .hero .badge {
        display: inline-block; background: rgba(56, 189, 248, 0.08);
        color: #38BDF8; padding: 0.2rem 0.9rem; border-radius: 6px;
        font-size: 0.75rem; font-weight: 500;
        border: 1px solid rgba(56, 189, 248, 0.15); margin-top: 0.7rem;
    }

    /* ── Cards ── */
    .card {
        background: #151B24; border-radius: 10px; padding: 1.25rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.05); margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: rgba(255,255,255,0.08); }
    .card h3 {
        color: #F0F6FC; font-size: 0.8rem; font-weight: 600;
        margin: 0 0 0.9rem 0; letter-spacing: 0.4px;
        text-transform: uppercase; opacity: 0.5;
    }

    /* ── Role Badges ── */
    .role-badge {
        display: inline-block; padding: 0.15rem 0.65rem; border-radius: 4px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.3px;
        margin-left: 0.4rem; vertical-align: middle;
    }
    .badge-batsman { background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.2); }
    .badge-bowler { background: rgba(251, 146, 60, 0.12); color: #FB923C; border: 1px solid rgba(251, 146, 60, 0.2); }
    .badge-allrounder { background: rgba(52, 211, 153, 0.12); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.2); }

    /* ── Buttons ── */
    .stButton button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important; font-weight: 500 !important; border: none !important;
        border-radius: 6px !important; padding: 0.55rem 1.8rem !important;
        font-size: 0.9rem !important; font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important; letter-spacing: 0.2px;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
    }

    /* ── Metrics ── */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important; font-weight: 600 !important;
        color: #F0F6FC !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important; opacity: 0.4 !important;
        font-weight: 500 !important; text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem; background: #151B24; border-radius: 8px;
        padding: 0.25rem; border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important; padding: 0.4rem 1rem !important;
        font-weight: 500 !important; font-size: 0.85rem !important;
        color: rgba(240, 246, 252, 0.5) !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(240, 246, 252, 0.8) !important;
        background: rgba(255,255,255,0.03) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(37, 99, 235, 0.15) !important;
        color: #60A5FA !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0D1117 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.05);
    }

    /* ── Form Inputs ── */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] input {
        background: #0D1117 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 6px !important;
        color: #F0F6FC !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] input:focus {
        border-color: rgba(37, 99, 235, 0.4) !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-size: 0.75rem !important;
        color: rgba(240, 246, 252, 0.5) !important;
        font-weight: 500 !important;
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(37, 99, 235, 0.06);
        border-left: 2.5px solid #3B82F6;
        border-radius: 5px; padding: 0.7rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.8rem; color: rgba(240, 246, 252, 0.6);
        line-height: 1.5;
    }
    .info-box strong { color: rgba(240, 246, 252, 0.8); }

    /* ── Feature Chips ── */
    .feature-chip {
        display: inline-block; background: rgba(255,255,255,0.04);
        border-radius: 3px; padding: 0.1rem 0.45rem; margin: 0.1rem;
        font-size: 0.7rem; font-family: 'SF Mono', 'JetBrains Mono', monospace;
        color: rgba(240, 246, 252, 0.5);
        border: 1px solid rgba(255,255,255,0.04);
    }

    /* ── Leaderboard entry ── */
    .leaderboard-entry {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.35rem 0.6rem; border-radius: 4px;
        transition: background 0.15s; font-size: 0.82rem;
        margin-bottom: 2px;
    }
    .leaderboard-entry:hover { background: rgba(255,255,255,0.02); }
    .leaderboard-entry .lb-model { font-weight: 500; }
    .leaderboard-entry .lb-metrics { opacity: 0.5; font-size: 0.75rem; }
    .leaderboard-entry.active { background: rgba(37, 99, 235, 0.06); }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-size: 0.82rem !important; font-weight: 500 !important;
        color: rgba(240, 246, 252, 0.6) !important;
        background: #151B24 !important;
        border-radius: 6px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* ── DataFrames ── */
    div[data-testid="stDataFrame"] {
        font-size: 0.8rem;
    }
    div[data-testid="stDataFrame"] th {
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.3px; opacity: 0.5;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.04) !important; }

    /* ── Caption ── */
    .stCaption {
        font-size: 0.75rem; opacity: 0.4;
    }

    /* ── Result card ── */
    .result-card {
        border-radius: 10px; padding: 1.8rem;
        display: flex; flex-direction: column; justify-content: center;
        height: 100%; border: 1px solid rgba(255,255,255,0.06);
    }
    .result-card .label {
        color: rgba(240, 246, 252, 0.3);
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 1px; margin: 0;
    }
    .result-card .score {
        font-size: 3rem; font-weight: 700;
        margin: 0.15rem 0; line-height: 1.1;
    }
    .result-card .score .unit {
        font-size: 1rem; font-weight: 400; opacity: 0.35;
    }
    .result-card .category {
        font-size: 1.2rem; font-weight: 600; margin: 0 0 0.5rem 0;
    }
    .result-card .meta {
        margin-top: 0.5rem; font-size: 0.78rem;
    }
    .result-card .meta .meta-label { opacity: 0.4; }
    .result-card .meta .meta-value { font-weight: 500; }
    .result-card .meta .sep { opacity: 0.2; margin: 0 0.4rem; }
</style>
"""


def score_to_category(score):
    score = np.clip(score, 0, 10)
    if score >= 8.5: return "Excellent", "#00D4AA"
    if score >= 7.0: return "Very Good", "#4ADE80"
    if score >= 5.5: return "Good", "#FBBF24"
    if score >= 4.0: return "Average", "#FB923C"
    if score >= 2.5: return "Below Average", "#F97316"
    return "Poor", "#FF6B6B"


def create_gauge(score, color="#00D4AA"):
    score = np.clip(score, 0, 10)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={
            'suffix': " / 10", 'font': {'size': 42, 'color': '#FFFFFF', 'family': 'Inter'},
        },
        delta={'reference': 5, 'increasing': {'color': '#00D4AA'}, 'decreasing': {'color': '#FF6B6B'}},
        gauge={
            'axis': {
                'range': [0, 10], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.3)",
                'tickfont': {'size': 11, 'color': 'rgba(255,255,255,0.5)'},
                'tickvals': [0, 2.5, 4, 5.5, 7, 8.5, 10],
                'ticktext': ['0', 'Poor', 'Below Avg', 'Good', 'Very Good', 'Excellent', '10'],
            },
            'bar': {'color': color, 'thickness': 0.45},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 2.5], 'color': 'rgba(255,107,107,0.12)'},
                {'range': [2.5, 4], 'color': 'rgba(249,115,22,0.12)'},
                {'range': [4, 5.5], 'color': 'rgba(251,146,60,0.12)'},
                {'range': [5.5, 7], 'color': 'rgba(251,191,36,0.12)'},
                {'range': [7, 8.5], 'color': 'rgba(74,222,128,0.12)'},
                {'range': [8.5, 10], 'color': 'rgba(0,212,170,0.12)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.6, 'value': score,
            }
        },
    ))
    fig.update_layout(
        height=300, margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'},
    )
    return fig


def render_role_badge(role):
    cls = {'batsman': 'badge-batsman', 'bowler': 'badge-bowler', 'all_rounder': 'badge-allrounder'}
    labels = {'batsman': 'BAT', 'bowler': 'BOWL', 'all_rounder': 'AR'}
    return f'<span class="role-badge {cls.get(role, "")}">{labels.get(role, role.upper())}</span>'


# ── Load ──
results, metadata, comp_df = load_artifacts()
df_full = load_data()
role_models = load_role_models()
name_map = load_name_map()
all_players = load_all_players()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <h1><span>Cric</span>Predict</h1>
    <p class="subtitle">Role-Aware Performance Analyzer — predicts player performance using only relevant stats</p>
    <span class="badge">XGBoost · LightGBM · Ensemble — Role-Aware</span>
</div>
""", unsafe_allow_html=True)

if results is None:
    st.warning("No trained models found. Run: `python scripts/pipeline.py`")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Player Role")
    st.markdown(
        "<p style='font-size:0.75rem; opacity:0.45; margin-bottom:0.5rem;'>"
        "Determines which stats are used.</p>",
        unsafe_allow_html=True,
    )

    role_descriptions = {
        'batsman': 'Pure batter — no bowling stats, no penalty for zero wickets',
        'all_rounder': 'Bats & bowls — full stats considered',
        'bowler': 'Primary bowler — bowling + fielding, batting excluded',
    }
    selected_role = st.radio(
        "Player Type",
        ['batsman', 'all_rounder', 'bowler'],
        index=0,
        format_func=lambda x: {
            'batsman': 'Batsman',
            'all_rounder': 'All-Rounder',
            'bowler': 'Bowler',
        }[x],
        help=role_descriptions.get('batsman', ''),
    )
    st.caption(role_descriptions[selected_role])

    st.markdown("---")
    st.markdown("### Model")

    role_model_available = selected_role in role_models

    model_options = []
    if role_model_available:
        for sub_model in role_models[selected_role]:
            model_options.append(f"{sub_model} ({selected_role})")
    for gm in sorted(results.keys()):
        model_options.append(f"{gm} (general)")

    selected_model = st.selectbox("Choose model", model_options, index=0)
    is_role_model = role_model_available and f"({selected_role})" in selected_model

    if is_role_model:
        sub_model_name = selected_model.split(" (")[0]
        col1, col2, col3 = st.columns(3)
        rm = role_models[selected_role][sub_model_name]
        col1.metric("R²", f"{rm['r2_score']:.4f}")
        col2.metric("MAE", f"{rm['mae']:.3f}")
        col3.metric("RMSE", f"{rm['rmse']:.3f}")
    else:
        model_key = selected_model.split(" (")[0]
        if model_key in results:
            col1, col2, col3 = st.columns(3)
            md = results[model_key]
            col1.metric("R²", f"{md['metrics']['r2_score']:.4f}")
            col2.metric("MAE", f"{md['metrics']['mae']:.3f}")
            col3.metric("RMSE", f"{md['metrics']['rmse']:.3f}")

    st.markdown("---")
    st.markdown("### :bar_chart: Leaderboard")

    if comp_df is not None:
        sorted_comp = comp_df.sort_values('R²', ascending=False)
        selected_base = selected_model.split(" (")[0]
        for i, (_, row) in enumerate(sorted_comp.iterrows()):
            medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "▫")
            active = " active" if row['Model'] == selected_base else ""
            st.markdown(
                f"<div class='leaderboard-entry{active}'>"
                f"<span class='lb-model'>{medal} {row['Model']}</span>"
                f"<span class='lb-metrics'>R² {row['R²']:.4f} · MAE {row['MAE']:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown(
        "<p style='font-size:0.78rem; opacity:0.55; line-height:1.7;'>"
        "<b>Batsman</b> — 14 batting + career features<br>"
        "<b>Bowler</b> — 9 bowling + career features<br>"
        "<b>All-Rounder</b> — 18 features (bat + bowl + career)<br><br>"
        "No penalty for batsmen with zero bowling stats. "
        "Bowlers who bat get full credit via All-Rounder mode.</p>",
        unsafe_allow_html=True,
    )

# ── Determine features for the selected role ──
role_features = FEATURE_SETS.get(selected_role, ALL_FEATURES)
has_batting = any(f in role_features for f in ['batting_average', 'strike_rate'])
has_bowling = any(f in role_features for f in ['bowling_average', 'economy_rate'])
has_career_bat = any(f in role_features for f in ['career_fifties', 'career_hundreds'])

# ── Feature ranges for defaults ──
feature_ranges = {}
if df_full is not None:
    for f in ALL_FEATURES:
        if f in df_full.columns:
            feature_ranges[f] = {
                'min': float(df_full[f].min()),
                'max': float(df_full[f].max()),
                'mean': float(df_full[f].mean()),
                'median': float(df_full[f].median()),
            }

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Analysis", "Explore", "Player Lookup"])

# ═══════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════
with tab1:
    role_label = {'batsman': 'Batsman', 'all_rounder': 'All-Rounder', 'bowler': 'Bowler'}[selected_role]
    st.markdown(
        f"<div class='card'>"
        f"<h3>Player Lookup</h3>",
        unsafe_allow_html=True,
    )

    player_ids = sorted(name_map.keys()) if name_map else []
    if player_ids:
        lookup_col1, lookup_col2 = st.columns([2, 3])
        with lookup_col1:
            player_options = {pid: f"{pid} — {name_map[pid]}" for pid in player_ids}
            selected_player_id = st.selectbox(
                "Select Player by ID",
                options=player_ids,
                format_func=lambda x: player_options[x],
                index=None,
                placeholder="Search player name or ID...",
            )
        if selected_player_id is not None and all_players is not None:
            pdata = all_players[all_players['player_id'] == selected_player_id].sort_values('season')
            if not pdata.empty:
                player_name = name_map.get(selected_player_id, f"ID {selected_player_id}")
                player_role = classify_player(pdata.iloc[-1])
                seasons = sorted(pdata['season'].unique())
                career = pdata.iloc[-1]

                with lookup_col2:
                    st.markdown(
                        f"<div style='padding-top:0.5rem;'>"
                        f"<span style='font-size:1.1rem;font-weight:600;color:#F0F6FC;'>{player_name}</span>"
                        f" <span class='role-badge badge-{player_role.replace('_','-')}'>"
                        f"{'BAT' if player_role=='batsman' else 'BOWL' if player_role=='bowler' else 'AR'}</span>"
                        f"<br><span style='font-size:0.78rem;opacity:0.45;'>"
                        f"ID: {selected_player_id} &middot; {seasons[0]}-{seasons[-1]} &middot; "
                        f"{int(career['career_matches'])} matches &middot; "
                        f"{int(career['career_runs'])} runs &middot; "
                        f"{int(career['career_wickets'])} wickets &middot; "
                        f"Score: {career['overall_performance_score']:.2f} ({career['performance_category']})"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )

                with st.expander(f"Season-by-season stats for {player_name}", expanded=False):
                    season_df = pdata[['season', 'matches', 'innings', 'runs', 'batting_average',
                                       'strike_rate', 'fours', 'sixes', 'fifties', 'hundreds',
                                       'wickets', 'economy_rate', 'bowling_average',
                                       'overall_performance_score', 'performance_category']].copy()
                    season_df.columns = ['Season', 'Mat', 'Inn', 'Runs', 'Avg', 'SR',
                                         '4s', '6s', '50s', '100s', 'Wkts', 'Econ',
                                         'Bowl Avg', 'Score', 'Category']
                    st.dataframe(season_df, width='stretch', hide_index=True)

                st.session_state['_player_defaults'] = pdata.iloc[-1].to_dict()
            else:
                st.warning(f"No data found for player ID {selected_player_id}")
                st.session_state.pop('_player_defaults', None)
        else:
            st.session_state.pop('_player_defaults', None)
            st.caption("Pick a player to auto-fill stats, or enter values manually below.")
    else:
        st.caption("Player name map unavailable — enter stats manually below.")
        st.session_state.pop('_player_defaults', None)

    st.markdown("</div>", unsafe_allow_html=True)

    # Resolve defaults: player data overrides dataset medians
    _pd = st.session_state.get('_player_defaults', None)
    def _default(feature, fallback_key='median', fallback_val=0, lo=None, hi=None):
        if _pd and feature in _pd:
            v = _pd[feature]
            if pd.notna(v):
                val = v
            else:
                val = feature_ranges.get(feature, {}).get(fallback_key, fallback_val)
        else:
            val = feature_ranges.get(feature, {}).get(fallback_key, fallback_val)
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    st.markdown(
        f"<div class='card'>"
        f"<h3>Player Statistics — {role_label} {render_role_badge(selected_role)}</h3>",
        unsafe_allow_html=True,
    )

    chips = ''.join(f'<span class="feature-chip">{f}</span>' for f in role_features[:8])
    if len(role_features) > 8:
        chips += f'<span class="feature-chip" style="opacity:0.3;">+{len(role_features)-8} more</span>'
    st.markdown(
        f"<div class='info-box'>"
        f"<strong>{len(role_features)} features</strong> &middot; {role_label}: "
        f"{chips}</div>",
        unsafe_allow_html=True,
    )

    input_data = {}

    # Universal: matches available for all roles
    mcols = st.columns([1, 2])
    with mcols[0]:
        input_data['matches'] = st.number_input(
            "Matches", 1, 30,
            value=int(_default('matches', lo=1, hi=30)),
            step=1, key="matches")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**Batting**")
        if has_batting:
            input_data['runs'] = st.number_input(
                "Runs", 0, 2000,
                value=int(_default('runs', lo=0, hi=2000)),
                step=10, key="runs")
            input_data['batting_average'] = st.number_input(
                "Batting Average", 0.0, 100.0,
                value=float(_default('batting_average', lo=0.0, hi=100.0)),
                step=0.5, format="%.1f", key="ba")
            input_data['strike_rate'] = st.number_input(
                "Strike Rate", 0.0, 300.0,
                value=float(_default('strike_rate', lo=0.0, hi=300.0)),
                step=0.5, format="%.1f", key="sr")
            input_data['fours'] = st.number_input(
                "Fours", 0, 100,
                value=int(_default('fours', lo=0, hi=100)),
                step=1, key="fours")
            input_data['sixes'] = st.number_input(
                "Sixes", 0, 100,
                value=int(_default('sixes', lo=0, hi=100)),
                step=1, key="sixes")
        else:
            st.caption("Not used — won't penalize bowlers")

    with col2:
        st.markdown("**Milestones & Fielding**")
        if has_batting:
            input_data['fifties'] = st.number_input(
                "Fifties", 0, 50,
                value=int(_default('fifties', lo=0, hi=50)),
                step=1, key="fifties")
            input_data['hundreds'] = st.number_input(
                "Hundreds", 0, 20,
                value=int(_default('hundreds', lo=0, hi=20)),
                step=1, key="hundreds")
        input_data['catches'] = st.number_input(
            "Catches", 0, 50,
            value=int(_default('catches', lo=0, hi=50)),
            step=1, key="catches")

        st.markdown("**Bowling**")
        if has_bowling:
            input_data['wickets'] = st.number_input(
                "Wickets", 0, 100,
                value=int(_default('wickets', lo=0, hi=100)),
                step=1, key="wickets")
            input_data['bowling_average'] = st.number_input(
                "Bowling Average", 0.0, 100.0,
                value=float(_default('bowling_average', lo=0.0, hi=100.0)),
                step=0.5, format="%.1f", key="bowl_avg")
            input_data['economy_rate'] = st.number_input(
                "Economy Rate", 0.0, 20.0,
                value=float(_default('economy_rate', lo=0.0, hi=20.0)),
                step=0.1, format="%.1f", key="econ")
            input_data['bowling_strike_rate'] = st.number_input(
                "Bowling Strike Rate", 0.0, 50.0,
                value=float(_default('bowling_strike_rate', lo=0.0, hi=50.0)),
                step=0.5, format="%.1f", key="bowl_sr")
        else:
            st.caption("Not used — won't penalize batsmen")

    with col3:
        st.markdown("**Career**")
        input_data['career_matches'] = st.number_input(
            "Career Matches", 0, 500,
            value=int(_default('career_matches', lo=0, hi=500)),
            step=5, key="cmat")
        input_data['seasons_played'] = st.number_input(
            "Seasons Played", 1, 20,
            value=int(_default('seasons_played', lo=1, hi=20)),
            step=1, key="sns")
        if has_career_bat:
            input_data['career_fifties'] = st.number_input(
                "Career Fifties", 0, 100,
                value=int(_default('career_fifties', lo=0, hi=100)),
                step=1, key="c50s")
            input_data['career_hundreds'] = st.number_input(
                "Career Hundreds", 0, 50,
                value=int(_default('career_hundreds', lo=0, hi=50)),
                step=1, key="c100s")
        input_data['career_catches'] = st.number_input(
            "Career Catches", 0, 200,
            value=int(_default('career_catches', lo=0, hi=200)),
            step=1, key="cct")

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_clicked = st.button(
            "PREDICT PERFORMANCE", type="primary", width='stretch'
        )

    if predict_clicked:
        if is_role_model:
            sub_model_name = selected_model.split(" (")[0]
            role_model_data = role_models[selected_role][sub_model_name]
            model = role_model_data['model']
            pred_features = role_model_data['features']
            model_name = selected_model
        else:
            model_key = selected_model.split(" (")[0]
            model_data = results[model_key]
            model = model_data['model']
            pred_features = metadata.get('features', ALL_FEATURES) if metadata else ALL_FEATURES
            model_name = selected_model

        X_input = pd.DataFrame([input_data]).reindex(columns=pred_features, fill_value=0)

        try:
            if not is_role_model and 'X_mean' in model_data:
                X_scaled = (X_input.values - model_data['X_mean']) / model_data['X_std']
                y_pred = model.predict(X_scaled)
            else:
                y_pred = model.predict(X_input.values) if hasattr(model, 'predict') else [0]
            pred_score = float(y_pred[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred_score = 0.0

        category, cat_color = score_to_category(pred_score)

        # Results display
        res_col1, res_col2 = st.columns([1, 1.2])
        with res_col1:
            st.plotly_chart(create_gauge(pred_score, cat_color), width='stretch')

        with res_col2:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {cat_color}08, transparent);
                        border-color: {cat_color}30;">
                <p class="label">Predicted Performance</p>
                <p class="score" style="color: {cat_color};">{pred_score:.2f}<span class="unit"> /10</span></p>
                <p class="category" style="color: #F0F6FC;">{category}</p>
                <div class="meta">
                    <span class="meta-label">Model:</span>
                    <span class="meta-value" style="color: {cat_color};">{model_name}</span>
                    <span class="sep">|</span>
                    <span class="meta-label">Role:</span>
                    <span class="meta-value">{role_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Input summary
        st.markdown(
            "<div class='card' style='margin-top: 1rem;'>"
            "<h3>Input Summary</h3>",
            unsafe_allow_html=True,
        )
        display_df = pd.DataFrame([input_data]).T.rename(columns={0: "Value"})
        display_df.index.name = "Feature"
        st.dataframe(display_df, width='stretch')

        # Feature contribution note
        used = [f for f in pred_features if f in input_data]
        missing = [f for f in pred_features if f not in input_data]
        if missing:
            st.caption(f"Features set to 0 (not provided): {', '.join(missing)}")
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — MODEL ANALYSIS
# ═══════════════════════════════════════════════
with tab2:
    st.markdown(
        f"<div class='card'><h3>Metrics — {selected_model} {render_role_badge(selected_role) if is_role_model else ''}</h3>",
        unsafe_allow_html=True,
    )

    if is_role_model:
        sub_model_name = selected_model.split(" (")[0]
        rm = role_models[selected_role][sub_model_name]
        metrics_data = [
            ("R² Score", f"{rm['r2_score']:.4f}", "#00D4AA"),
            ("MAE", f"{rm['mae']:.4f}", "#4ADE80"),
            ("RMSE", f"{rm['rmse']:.4f}", "#FBBF24"),
            ("Features", str(len(rm['features'])), "#4A9EFF"),
        ]
    else:
        model_key = selected_model.split(" (")[0]
        md = results[model_key]
        metrics_data = [
            ("R² Score", f"{md['metrics']['r2_score']:.4f}", "#00D4AA"),
            ("MAE", f"{md['metrics']['mae']:.4f}", "#4ADE80"),
            ("RMSE", f"{md['metrics']['rmse']:.4f}", "#FBBF24"),
            ("Training Time", f"{md['time']:.1f}s", "#4A9EFF"),
        ]

    cols = st.columns(len(metrics_data))
    for ci, (label, value, color) in enumerate(metrics_data):
        with cols[ci]:
            st.markdown(
                f"<div style='background: rgba(0,0,0,0.2); border-radius: 10px; padding: 1rem; text-align: center;'>"
                f"<p style='color: rgba(255,255,255,0.4); font-size: 0.75rem; margin: 0; text-transform: uppercase;'>{label}</p>"
                f"<p style='color: {color}; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0;'>{value}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Comparison charts
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("<div class='card'><h3>R² Score Comparison</h3>", unsafe_allow_html=True)
        if comp_df is not None:
            all_models_r2 = []
            for _, r in comp_df.iterrows():
                all_models_r2.append({'Model': f"{r['Model']} (general)", 'R²': r['R²'], 'source': 'General'})
            for role, sub_models in role_models.items():
                for sm_name, sm_data in sub_models.items():
                    all_models_r2.append({
                        'Model': f"{sm_name} ({role})",
                        'R²': sm_data['r2_score'],
                        'source': 'Role',
                    })

            r2_df = pd.DataFrame(all_models_r2)
            hl_model = selected_model

            colors = ['#00D4AA' if m == hl_model else
                      'rgba(0,212,170,0.35)' if '(batsman)' in m or '(bowler)' in m or '(all_rounder)' in m else 'rgba(255,255,255,0.2)'
                      for m in r2_df['Model']]

            fig = go.Figure(go.Bar(
                x=r2_df['Model'], y=r2_df['R²'],
                marker_color=colors,
                text=r2_df['R²'].round(3),
                textposition='outside', textfont={'size': 10},
            ))
            fig.update_layout(
                height=350, margin=dict(l=20, r=20, t=10, b=80),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'title': '', 'tickfont': {'color': 'rgba(255,255,255,0.6)', 'size': 10}},
                yaxis={'title': 'R²', 'tickfont': {'color': 'rgba(255,255,255,0.4)'},
                       'gridcolor': 'rgba(255,255,255,0.05)', 'range': [0, 1]},
                showlegend=False,
            )
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'><h3>Error Metrics</h3>", unsafe_allow_html=True)
        if comp_df is not None:
            all_errors = []
            for _, r in comp_df.iterrows():
                all_errors.append({'Model': f"{r['Model']} (general)", 'MAE': r['MAE'], 'RMSE': r['RMSE']})
            for role, sub_models in role_models.items():
                for sm_name, sm_data in sub_models.items():
                    all_errors.append({
                        'Model': f"{sm_name} ({role})",
                        'MAE': sm_data['mae'],
                        'RMSE': sm_data['rmse'],
                    })
            err_df = pd.DataFrame(all_errors)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='MAE', x=err_df['Model'], y=err_df['MAE'],
                                 marker_color='#FB923C', text=err_df['MAE'].round(3),
                                 textposition='outside'))
            fig.add_trace(go.Bar(name='RMSE', x=err_df['Model'], y=err_df['RMSE'],
                                 marker_color='#FF6B6B', text=err_df['RMSE'].round(3),
                                 textposition='outside'))
            fig.update_layout(
                barmode='group', height=350,
                margin=dict(l=20, r=20, t=10, b=100),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'title': '', 'tickfont': {'color': 'rgba(255,255,255,0.6)', 'size': 10}},
                yaxis={'title': 'Error', 'tickfont': {'color': 'rgba(255,255,255,0.4)'},
                       'gridcolor': 'rgba(255,255,255,0.05)'},
                legend={'font': {'color': 'rgba(255,255,255,0.6)'}},
            )
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    # Expanders
    with st.expander("Parameters"):
        if is_role_model:
            sub_model_name = selected_model.split(" (")[0]
            st.json({"model": sub_model_name, "role": selected_role, "features": role_models[selected_role][sub_model_name]['features']})
        else:
            model_key = selected_model.split(" (")[0]
            st.json(results[model_key].get('params', {}))

    with st.expander(f"Features ({len(role_features)})"):
        cols = st.columns(3)
        for i, f in enumerate(role_features):
            cols[i % 3].markdown(f"- `{f}`")

    with st.expander("Role Models"):
        for role in ['batsman', 'bowler', 'all_rounder']:
            if role in role_models:
                st.markdown(f"**{role.title()}**")
                for sm_name, sm_data in role_models[role].items():
                    st.markdown(
                        f"  **{sm_name}**: R²={sm_data['r2_score']:.4f}, MAE={sm_data['mae']:.4f}, "
                        f"{len(sm_data['features'])} features"
                    )

    with st.expander("Dataset"):
        if metadata:
            st.markdown(f"- **Train samples:** {metadata['train_samples']:,}")
            st.markdown(f"- **Test samples:** {metadata['test_samples']:,}")
            st.markdown(f"- **Total features:** {metadata['n_features']}")
            player_count = f"{df_full['player_id'].nunique():,}" if df_full is not None else "N/A"
            st.markdown(f"- **Players:** {player_count}")


# ═══════════════════════════════════════════════
# TAB 3 — EXPLORER
# ═══════════════════════════════════════════════
with tab3:
    if df_full is not None:
        exclude_cols = {'_outlier', 'batting_impact', 'bowling_impact', 'consistency_score',
                        'overall_performance_score', 'highest_score_num', 'highest_score_notout',
                        'best_bowling_wickets', 'best_bowling_runs'}
        num_cols = [c for c in df_full.select_dtypes(include=np.number).columns.tolist()
                    if c not in exclude_cols]
        cats = [c for c in df_full.columns if c not in num_cols
                and c not in exclude_cols and df_full[c].nunique() < 20]

        st.markdown("<div class='card'><h3>Feature Explorer</h3>", unsafe_allow_html=True)

        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])
        with exp_col1:
            col_x = st.selectbox("X-axis", num_cols,
                                 index=num_cols.index('runs') if 'runs' in num_cols else 0)
        with exp_col2:
            col_y = st.selectbox("Y-axis", num_cols,
                                 index=num_cols.index('overall_performance_score') if 'overall_performance_score' in num_cols else 1)
        with exp_col3:
            color_opts = cats + [c for c in num_cols if c not in (col_x, col_y)]
            col_color = st.selectbox("Color by", color_opts, index=0)

        sample_df = df_full.sample(min(1000, len(df_full)), random_state=42)

        fig = px.scatter(
            sample_df, x=col_x, y=col_y, color=col_color,
            title=f"{col_y} vs {col_x}",
            opacity=0.6, height=500,
            template='plotly_dark',
            color_continuous_scale='viridis' if col_color in num_cols else None,
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'rgba(255,255,255,0.7)'},
            xaxis={'gridcolor': 'rgba(255,255,255,0.05)'},
            yaxis={'gridcolor': 'rgba(255,255,255,0.05)'},
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Data Sample</h3>", unsafe_allow_html=True)
        show_cols = [c for c in ALL_FEATURES if c in df_full.columns] + ['performance_category']
        st.dataframe(df_full[show_cols].head(100), width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Feature data not found. Ensure `data/processed/ipl_features.csv` exists.")

# ═══════════════════════════════════════════════
# TAB 4 — PLAYER LOOKUP
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='card'><h3>Player Lookup by ID</h3>", unsafe_allow_html=True)
    st.caption(f"Database contains {len(name_map)} players (IDs 1 – {max(name_map.keys()) if name_map else 0})")

    if all_players is not None and name_map:
        lookup_id = st.number_input(
            "Enter Player ID", min_value=1, max_value=max(name_map.keys()),
            value=1, step=1, key="lookup_id",
        )

        pdata = all_players[all_players['player_id'] == lookup_id].sort_values('season')
        if not pdata.empty:
            player_name = name_map.get(lookup_id, f"ID {lookup_id}")
            p_role = classify_player(pdata.iloc[-1])
            role_cls = {'batsman': 'badge-batsman', 'bowler': 'badge-bowler', 'all_rounder': 'badge-allrounder'}
            role_lbl = {'batsman': 'BATSMAN', 'bowler': 'BOWLER', 'all_rounder': 'ALL-ROUNDER'}

            st.markdown(
                f"<div style='margin-bottom:1rem;'>"
                f"<span style='font-size:1.4rem;font-weight:700;color:#F0F6FC;'>{player_name}</span>"
                f" <span class='role-badge {role_cls.get(p_role, '')}'>{role_lbl.get(p_role, '')}</span>"
                f"<br><span style='font-size:0.82rem;opacity:0.4;'>Player ID: {lookup_id}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            career = pdata.iloc[-1]
            seasons = sorted(pdata['season'].unique())

            # Career summary cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Career Matches", int(career['career_matches']))
            c2.metric("Career Runs", int(career['career_runs']))
            c3.metric("Career Wickets", int(career['career_wickets']))
            c4.metric("Seasons Played", int(career['seasons_played']))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Fifties", int(career['career_fifties']))
            c6.metric("Hundreds", int(career['career_hundreds']))
            c7.metric("Catches", int(career['career_catches']))
            c8.metric("Performance", f"{career['overall_performance_score']:.2f}")

            # Performance breakdown
            st.markdown("<div class='card' style='margin-top:0.5rem;'><h3>Performance Breakdown</h3>", unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Batting Impact", f"{career['batting_impact']:.4f}")
            p2.metric("Bowling Impact", f"{career['bowling_impact']:.4f}")
            p3.metric("Consistency", f"{career['consistency_score']:.4f}")
            p4.metric("Category", career['performance_category'])
            st.markdown("</div>", unsafe_allow_html=True)

            # Season-by-season table
            st.markdown("<div class='card'><h3>Season-by-Season Stats</h3>", unsafe_allow_html=True)
            season_df = pdata[['season', 'matches', 'innings', 'runs', 'batting_average',
                               'strike_rate', 'fours', 'sixes', 'fifties', 'hundreds',
                               'highest_score', 'catches', 'wickets', 'bowling_average',
                               'economy_rate', 'bowling_strike_rate', 'best_bowling',
                               'overall_performance_score', 'performance_category']].copy()
            season_df.columns = ['Season', 'Mat', 'Inn', 'Runs', 'Avg', 'SR',
                                 '4s', '6s', '50s', '100s', 'HS', 'Catches',
                                 'Wkts', 'Bowl Avg', 'Econ', 'Bowl SR',
                                 'Best Bowling', 'Score', 'Category']
            st.dataframe(season_df, width='stretch', hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Career totals chart
            st.markdown("<div class='card'><h3>Career Progression</h3>", unsafe_allow_html=True)
            chart_df = pdata[['season', 'runs', 'wickets', 'overall_performance_score']].copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Runs', x=chart_df['season'], y=chart_df['runs'],
                                 marker_color='#38BDF8', text=chart_df['runs'], textposition='outside'))
            fig.add_trace(go.Bar(name='Wickets', x=chart_df['season'], y=chart_df['wickets'],
                                 marker_color='#FB923C', text=chart_df['wickets'], textposition='outside'))
            fig.add_trace(go.Scatter(name='Score', x=chart_df['season'], y=chart_df['overall_performance_score'],
                                     mode='lines+markers', yaxis='y2',
                                     line=dict(color='#00D4AA', width=2), marker=dict(size=6)))
            fig.update_layout(
                barmode='group', height=380,
                margin=dict(l=20, r=20, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'rgba(255,255,255,0.7)'},
                xaxis={'title': 'Season', 'tickfont': {'color': 'rgba(255,255,255,0.5)'}},
                yaxis={'title': 'Runs / Wickets', 'tickfont': {'color': 'rgba(255,255,255,0.4)'},
                       'gridcolor': 'rgba(255,255,255,0.05)'},
                yaxis2={'title': 'Score', 'overlaying': 'y', 'side': 'right',
                        'tickfont': {'color': 'rgba(0,212,170,0.6)'}},
                legend={'orientation': 'h', 'y': 1.12, 'font': {'color': 'rgba(255,255,255,0.6)'}},
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info(f"No data found for Player ID {lookup_id}.")
    else:
        st.warning("Player data not loaded. Ensure data files exist.")

# ── Footer ──
st.markdown(
    "<div style='text-align: center; padding: 2rem 0 0 0; opacity: 0.15; font-size: 0.7rem;'>"
    "CricPredict — Role-Aware Player Performance Analyzer"
    f" &middot; {datetime.now().strftime('%Y')}</div>",
    unsafe_allow_html=True,
)
