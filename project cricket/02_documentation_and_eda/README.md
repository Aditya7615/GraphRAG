# Player Performance Prediction

**Group 1 - Team Project (Week 4)**

Data pipeline for cricket player performance using IPL and international statistics. Players identified by unique `player_id` — no player names in processed data.

## Project Structure

```
project cricket/
├── 01_raw_data_and_scrapers/     # Raw data + scraper scripts
│   ├── ipl_batting_fielding_stats.csv
│   ├── bcci_stats_rankings_all.csv
│   ├── generate_ipl_data.py         # IPL stats scraper
│   └── bcci_stats_scraper.py        # BCCI international stats scraper
│
├── 02_documentation_and_eda/     # Documentation + full EDA
│   ├── README.md
│   ├── requirements.txt
│   ├── docs/
│   │   ├── Project_Plan.md
│   │   └── Test_Plan.md
│   └── notebooks/
│       └── 01_eda.ipynb             # Full EDA (raw → cleaned → feature → split)
│
└── 03_eda_cleaned/               # Pre-processed data + cleaned EDA
    ├── ipl_cleaned.csv              # Cleaned IPL data (no player_name)
    ├── ipl_features.csv             # Feature-engineered IPL data
    └── 01_eda_cleaned.ipynb         # EDA on pre-cleaned data
```

## Setup

### 1. Install dependencies

```bash
pip install -r "02_documentation_and_eda/requirements.txt"
```

### 2. Data Collection (if re-scraping needed)

```bash
# IPL data scraper
python "01_raw_data_and_scrapers/generate_ipl_data.py"

# BCCI data scraper
python "01_raw_data_and_scrapers/bcci_stats_scraper.py"
```

Existing CSVs are already provided in `01_raw_data_and_scrapers/`.

### 3. Run EDA Notebooks

Open the notebooks in Jupyter:

```bash
jupyter notebook "02_documentation_and_eda/notebooks/01_eda.ipynb"
jupyter notebook "03_eda_cleaned/01_eda_cleaned.ipynb"
```

The notebooks contain all preprocessing, cleaning, and feature engineering code inline.

## Pipeline Steps (within EDA notebooks)

| Step | Description |
|------|-------------|
| 1 | Data Loading & Overview — IPL + BCCI |
| 2 | Data Quality Assessment — missing values, duplicates, anomalies |
| 3 | Univariate Analysis — distributions with summary statistics |
| 4 | Bivariate & Multivariate — pairplots, correlation heatmaps |
| 5 | Data Cleaning — IPL preprocessing (dtypes, nulls, dedup, outliers) |
| 6 | Data Cleaning — BCCI preprocessing (column normalize, pivot) |
| 7 | Feature Engineering — batting/bowling impact, consistency, target |
| 8 | Engineered Feature Analysis — correlations, categories, top performers |
| 9 | Train-Test Split — 80:20 split for Phase 2 modeling |

## Player Identity

- `player_name` is replaced with unique `player_id` (integer) during cleaning
- `player_name` column is dropped — not present in cleaned/featured data
- Both IPL and BCCI datasets get independent `player_id` mappings

## Engineered Features

| Feature | Used For |
|---------|----------|
| Total Runs | Aggregate batting contribution |
| Total Wickets | Aggregate bowling contribution |
| Batting Impact | Composite (runs × avg × SR) normalized [0-10] |
| Bowling Impact | Composite (wickets / (avg × econ)) normalized [0-10] |
| Consistency Score | Multi-season consistency [0-10] |
| Overall Performance Score | Target variable [0-10] |
| Performance Category | Poor / Average / Good / Very Good / Excellent |

## Data Sources

- **IPL Stats:** Scraped from IPL Sports Mechanic API (2008–2026), ~800 unique players
- **BCCI Stats:** Scraped from BCCI.tv (TEST, ODI, T20I formats), ~340 unique players

## Phase 2 (Future)

Model training, evaluation, and prediction using the feature-engineered dataset.
