# Test Plan - Player Performance Prediction

**Project:** Group 1 - Player Performance Prediction  
**Version:** 1.0 (Week 4)  
**Date:** June 4, 2026  
**Test Plan Manager:** Aditya

---

## 1. Test Strategy

Verification of data integrity, cleaning correctness, and pipeline completeness. All processing code is implemented inline within the EDA notebooks and verified through manual execution and visual inspection of outputs.

## 2. Test Scope

### In-Scope
- Data loading and type conversion
- Missing value handling (highest_score, best_bowling)
- Outlier detection (extreme SR, inconsistent matches)
- Player ID assignment (no player names in processed data)
- Duplicate removal (player-season pairs)
- BCCI column name normalization (smart quotes, spaces)
- Feature engineering correctness
- Train-test split correctness

### Out-of-Scope
- Model training tests (Phase 2)
- Data scraper tests (external API dependency)
- Performance/load testing
- Security testing

## 3. Test Environment

| Component | Specification |
|-----------|---------------|
| Python | 3.9+ |
| OS | macOS / Linux / Windows |
| Dependencies | See `02_documentation_and_eda/requirements.txt` |
| Notebook Runner | Jupyter Notebook or JupyterLab |

## 4. Test Cases

### TC-DATA-000: Player ID Assignment
- **Description:** Verify player_name is replaced with unique player_id and player_name is dropped
- **Precondition:** IPL data loaded in notebook
- **Steps:** Execute `add_player_id()` cell in notebook
- **Expected:** `player_id` column created; `player_name` column dropped; mapping dict contains all unique players with int IDs
- **Status:** ✅ Verified

### TC-DATA-001: Data Type Conversion
- **Description:** Verify numeric columns are converted to correct types
- **Precondition:** IPL CSV loaded
- **Steps:** Execute `fix_dtypes()` cell
- **Expected:** All numeric columns are `int64` or `float64`
- **Status:** ✅ Verified

### TC-DATA-002: Missing Value Handling
- **Description:** Verify highest_score and best_bowling are handled
- **Precondition:** Data with None values
- **Steps:** Execute `clean_highest_score()`, `parse_best_bowling()` cells
- **Expected:** highest_score extracted as numeric + not_out flag; best_bowling parsed into wickets/runs
- **Status:** ✅ Verified

### TC-DATA-003: Duplicate Removal
- **Description:** Verify duplicate player-season rows removed using player_id
- **Precondition:** Data with duplicates
- **Steps:** Execute `remove_duplicates()` cell
- **Expected:** Row count decreases; no duplicate player_id-season pairs
- **Status:** ✅ Verified

### TC-DATA-004: Outlier Detection
- **Description:** Verify extreme values flagged
- **Precondition:** Data with SR > 350 or inconsistent matches/runs
- **Steps:** Execute `flag_outliers()` cell
- **Expected:** Outlier flag set to 1 for anomalous rows
- **Status:** ✅ Verified

### TC-DATA-005: BCCI Column Name Normalization
- **Description:** Verify BCCI smart quotes (U+2019) and spaces in column names are normalized
- **Precondition:** BCCI raw data loaded
- **Steps:** Execute `clean_bcci()` cell
- **Expected:** Columns like `batting_4's`, `batting_6's` become `batting_4s`, `batting_6s`; spaces become underscores
- **Status:** ✅ Verified

### TC-FEAT-001: Batting Impact Computation
- **Description:** Verify batting_impact is in [0, 10] range
- **Precondition:** FeatureEngineer with valid data
- **Steps:** Execute `add_batting_impact()` cell
- **Expected:** `batting_impact` column created, all values between 0 and 10
- **Status:** ✅ Verified

### TC-FEAT-002: Bowling Impact Computation
- **Description:** Verify bowling_impact is in [0, 10] range
- **Precondition:** FeatureEngineer with valid data
- **Steps:** Execute `add_bowling_impact()` cell
- **Expected:** `bowling_impact` column created, all values between 0 and 10
- **Status:** ✅ Verified

### TC-FEAT-003: Consistency Score Computation
- **Description:** Verify consistency_score is in [0, 10] range
- **Precondition:** FeatureEngineer with multi-season data
- **Steps:** Execute `add_consistency_score()` cell
- **Expected:** `consistency_score` column created, all values between 0 and 10
- **Status:** ✅ Verified

### TC-FEAT-004: Overall Performance Score
- **Description:** Verify target variable computed correctly
- **Precondition:** All feature columns present
- **Steps:** Execute `add_performance_score()`, `categorize()` cells
- **Expected:** `overall_performance_score` in [0, 10]; `performance_category` with valid labels
- **Status:** ✅ Verified

### TC-SPLIT-001: Train-Test Split
- **Description:** Verify 80:20 train-test split
- **Precondition:** Feature-engineered DataFrame
- **Steps:** Execute train-test split cell
- **Expected:** 80% train, 20% test split; feature columns exclude player_id, target, and engineered composites
- **Status:** ✅ Verified

### TC-SPLIT-002: Player Name Not in Features
- **Description:** Verify player_name is NOT in training features
- **Precondition:** Prepared data split
- **Steps:** Check feature columns output
- **Expected:** `player_name` not in feature list
- **Status:** ✅ Verified

### TC-INT-001: Full EDA Notebook Execution
- **Description:** Verify end-to-end notebook execution without errors
- **Precondition:** All CSV files present
- **Steps:** Run all cells in `02_documentation_and_eda/notebooks/01_eda.ipynb`
- **Expected:** All cells execute without errors; final output shows split summary
- **Status:** ✅ Verified

### TC-INT-002: Cleaned EDA Notebook Execution
- **Description:** Verify cleaned data notebook executes correctly
- **Precondition:** Pre-cleaned CSV files present
- **Steps:** Run all cells in `03_eda_cleaned/01_eda_cleaned.ipynb`
- **Expected:** All cells execute without errors; BCCI data loads and cleans correctly
- **Status:** ✅ Verified

## 5. Test Execution

```bash
# Run the full EDA notebook (interactive)
jupyter notebook "02_documentation_and_eda/notebooks/01_eda.ipynb"

# Run the cleaned data EDA notebook
jupyter notebook "03_eda_cleaned/01_eda_cleaned.ipynb"
```

## 6. Pass/Fail Criteria

| Criterion | Threshold |
|-----------|-----------|
| Notebook cells execute successfully | 100% |
| No `player_name` in cleaned data | Confirmed |
| All visualizations render | Confirmed |
| Train-test split ratios correct | Confirmed |

## 7. Test Deliverables

- ✅ `02_documentation_and_eda/notebooks/01_eda.ipynb` — Full EDA with inline processing
- ✅ `03_eda_cleaned/01_eda_cleaned.ipynb` — Cleaned data EDA
- ✅ This Test Plan document

---

**Version History**

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2026-06-04 | Data pipeline test plan |
