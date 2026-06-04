# Project Plan - Player Performance Prediction

**Project:** Group 1 - Player Performance Prediction  
**Version:** 1.0 (Week 4)  
**Date:** June 4, 2026  

---

## 1. Project Overview

Build a data pipeline that processes cricket player performance data using IPL and international statistics. The pipeline handles data collection, cleaning, feature engineering, EDA, and train-test splitting. Players are identified by unique `player_id` instead of names.

## 2. Team Roles

| Role | Name | Responsibilities |
|------|------|-----------------|
| Team Leader | Aditya | Timeline, coordination, deliverables |
| Data Engineer | Aditya | Data scraping, cleaning, preprocessing |
| ML Engineer | Aditya | Model training, tuning, evaluation (Phase 2) |
| Test Plan Manager | Aditya | Test plans, quality assurance |
| Documentation | Aditya | Project documentation, reports |

## 3. Scope

### In-Scope
- IPL player data (2008–2026)
- International cricket stats (TEST, ODI, T20I)
- Feature engineering (Batting/Bowling Impact, Consistency Score)
- EDA & visualization
- Train-test split (80:20)
- Player identification via unique `player_id`

### Out-of-Scope
- Model training (Phase 2)
- Real-time match prediction
- Ball-by-ball analysis
- Player injury prediction
- Auction price prediction

## 4. Timeline (Week 4)

| Phase | Task | Status | Delivery |
|-------|------|--------|----------|
| 1 | Data Collection (IPL + BCCI scrapers) | ✅ Complete | `01_raw_data_and_scrapers/generate_ipl_data.py`, `01_raw_data_and_scrapers/bcci_stats_scraper.py` |
| 2 | Data Preprocessing & Cleaning | ✅ Complete | Inline in `02_documentation_and_eda/notebooks/01_eda.ipynb` |
| 3 | Feature Engineering | ✅ Complete | Inline in `02_documentation_and_eda/notebooks/01_eda.ipynb` |
| 4 | EDA & Visualization | ✅ Complete | `02_documentation_and_eda/notebooks/01_eda.ipynb`, `03_eda_cleaned/01_eda_cleaned.ipynb` |
| 5 | Train-Test Split | ✅ Complete | Inline in EDA notebooks |
| 6 | Documentation | ✅ Complete | `02_documentation_and_eda/docs/`, `02_documentation_and_eda/README.md` |

## 5. Technical Architecture

```
Data Sources (IPL API, BCCI Website)
        ↓
Data Scrapers (generate_ipl_data.py, bcci_stats_scraper.py)
        ↓
Raw CSV Data (01_raw_data_and_scrapers/)
        ↓
Preprocessing (missing values, dtypes, dedup, outlier flagging)
        ↓
Feature Engineering (batting_impact, bowling_impact,
                     consistency_score, overall_performance_score)
        ↓
Train-Test Split (80:20)
        ↓
Ready for Model Training (Phase 2)
```

## 6. Feature Definitions

| Feature | Formula | Purpose |
|---------|---------|---------|
| Total Runs | Σ(runs) per player | Overall batting contribution |
| Total Wickets | Σ(wickets) per player | Overall bowling contribution |
| Batting Impact | (runs × avg × SR) / 10000, normalized [0-10] | Composite batting measure |
| Bowling Impact | (wickets / (bowl_avg × econ)) × 100, normalized [0-10] | Composite bowling measure |
| Consistency Score | Weighted mix of seasons_active, total_matches, CV of runs [0-10] | Performance consistency |
| Overall Performance Score | 0.40×BatImpact + 0.30×BowlImpact + 0.30×Consistency + noise [0-10] | Target prediction variable |
| Player ID | Integer encoding of player_name | Replaces player_name everywhere |

## 7. Feature Columns (for future model training)

| Feature | Type | Category |
|---------|------|----------|
| batting_average | float | Batting efficiency |
| strike_rate | float | Batting efficiency |
| bowling_average | float | Bowling efficiency |
| economy_rate | float | Bowling efficiency |
| bowling_strike_rate | float | Bowling efficiency |
| fours, sixes | int | Batting skill indicators |
| fifties, hundreds | int | Batting milestones |
| catches | int | Fielding contribution |
| career_fifties, career_hundreds, career_catches | int | Career milestones |

## 8. Risk Management

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing data in scraped stats | Medium | Imputation with 0/mean values |
| API changes in data sources | High | Modular scrapers with error handling |

## 9. Quality Gates

| Gate | Status |
|------|--------|
| Code runs without errors | ✅ Verified |
| EDA covers distributions, correlations, outliers | ✅ Verified |
| Feature engineering produces expected columns | ✅ Verified |
| Train-test split correctly sized | ✅ Verified |
| No `player_name` in cleaned/featured data | ✅ Verified |
| BCCI smart quotes in column names normalized | ✅ Fixed |

---

**Version History**

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2026-06-04 | Initial project plan — data pipeline, EDA, feature engineering, train-test split |
