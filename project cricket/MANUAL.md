# Project Manual

## Overview
CricPredict is a machine learning system that predicts cricket player performance scores (0–10) based on IPL season statistics. It supports batsmen, bowlers, and all-rounders with role-aware models.

## Score Categories

| Category | Range | Description |
|----------|-------|-------------|
| Excellent | 8.5–10 | Elite world-class performers |
| Very Good | 7.0–8.49 | Consistent high performers |
| Good | 5.5–6.99 | Solid contributors |
| Average | 4.0–5.49 | Average performers |
| Below Average | 2.5–3.99 | Below par |
| Poor | 0–2.49 | Weak performance |

## Models

### General Models (trained on all players)
- **XGBoost**: Gradient-boosted trees, ~0.985 R²
- **LightGBM**: Lightweight gradient boosting, ~0.986 R²
- **Ensemble**: Blended XGBoost + LightGBM, ~0.985 R²

### Role-Aware Models (trained on role-specific data)
- **Batsman**: LightGBM, ~0.949 R²
- **Bowler**: LightGBM, ~0.632 R² (lower due to sparse bowling features)
- **All-Rounder**: LightGBM, ~0.975 R²

## Feature Engineering

### Batting Impact
Log-scaled combination of runs, batting average, and strike rate:
```
raw = runs * avg * sr / 10000
batting_impact = log10_scale(raw)
```

### Bowling Impact
Log-scaled combination of wickets, bowling average, and economy:
```
raw = wickets / (bowling_avg * economy) * 100
bowling_impact = log10_scale(raw)  [zero for non-bowlers]
```

### Consistency Score
Weighted blend of longevity and stability:
- 20% seasons active (log-scaled)
- 20% career matches (log-scaled)
- 60% stability (low coefficient of variation across seasons)

### Overall Performance Score
Quantile-mapped combination with user-defined distribution:
- 45% batting impact
- 25% bowling impact
- 30% consistency score

The raw weighted score is mapped through percentile anchors to produce the desired category distribution.

## Pipeline

```bash
# Full pipeline (feature engineering + training)
python scripts/pipeline.py

# Feature engineering only
python scripts/build_features.py

# Training only (requires prebuilt features)
python scripts/train.py

# Launch Streamlit app
streamlit run app/app.py
```

## Project Structure

```
├── scripts/
│   ├── pipeline.py       # Full pipeline runner
│   ├── build_features.py # Feature engineering entry point
│   └── train.py          # Training entry point
├── src/
│   ├── features/
│   │   └── data_utils.py # Feature engineering, data loading
│   └── models/
│       ├── train_model.py       # Training orchestration
│       ├── xgboost_model.py     # XGBoost wrapper
│       ├── lightgbm_model.py    # LightGBM wrapper
│       └── ensemble_model.py    # Ensemble blending
├── app/
│   └── app.py            # Streamlit dashboard
├── data/
│   ├── raw/              # Source CSVs
│   ├── processed/        # Cleaned + featured CSVs
│   └── external/
├── models/               # Trained model artifacts
│   └── role_models/      # Role-specific models
└── notebooks/            # Exploration notebooks
```

## Data Flow

1. Raw IPL stats → `data/processed/ipl_cleaned.csv` (pre-cleaned)
2. `build_features.py` → computes batting/bowling impact, consistency, target score
3. `train.py` → trains XGBoost, LightGBM, Ensemble models
4. `app.py` → loads models for interactive prediction

## Category Thresholds (App)

The app uses these thresholds for gauge visualization:
- Excellent: 8.5–10
- Very Good: 7.0–8.5
- Good: 5.5–7.0
- Average: 4.0–5.5
- Below Average: 2.5–4.0
- Poor: 0–2.5
