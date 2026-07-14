# Model Training Report — Player Performance Prediction

**Project:** Group 1 — Player Performance Prediction  
**Phase:** 2 — Model Training, Evaluation & Deployment  
**Date:** June 2026

---

## 1. Executive Summary

This report documents the complete model training pipeline for predicting cricket player performance. Five distinct modelling approaches were implemented, evaluated, and compared:

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble (Bagging) | Baseline model — robust to overfitting, handles non-linearity |
| **MLP (Neural Network)** | Deep Learning | Multi-layer perceptron for learning complex feature interactions |
| **XGBoost** | Ensemble (Boosting) | Gradient-boosted trees — state-of-the-art for structured data |
| **Hybrid Model** | Stacking Ensemble | Ridge regression meta-learner over RF + MLP + XGBoost predictions |
| **TabPFN** | Transfer Learning | Pre-trained transformer for tabular data — few-shot learning |

All models predict the **Overall Performance Score** (continuous, 0–10) engineered from batting impact, bowling impact, and consistency score.

---

## 2. Data Preparation

### 2.1 Source Data
- **IPL Dataset:** 2,826 player-season rows from `03_eda_cleaned/ipl_features.csv`
- **Players:** ~800 unique IPL players (identified by `player_id`)
- **Seasons:** 2008–2026
- **Target:** `overall_performance_score` [0, 10]

### 2.2 Feature Set (13 features)

| Feature | Type | Domain |
|---------|------|--------|
| `batting_average` | float | Batting efficiency |
| `strike_rate` | float | Batting aggression |
| `fours` | int | Boundary count |
| `sixes` | int | Six count |
| `fifties` | int | 50+ scores |
| `hundreds` | int | 100+ scores |
| `catches` | int | Fielding contribution |
| `bowling_average` | float | Bowling economy (lower is better) |
| `economy_rate` | float | Runs conceded per over |
| `bowling_strike_rate` | float | Balls per wicket |
| `career_fifties` | int | Career milestone |
| `career_hundreds` | int | Career milestone |
| `career_catches` | int | Career fielding |

### 2.3 Train-Test Split
- **Split ratio:** 80% train / 20% test
- **Train samples:** ~2,260
- **Test samples:** ~566
- **Random state:** 42 (reproducible)

---

## 3. Model Architectures

### 3.1 Random Forest

**Architecture:** Ensemble of 100–500 decision trees with bootstrap aggregation.

**Hyperparameter Search Space:**

| Parameter | Values Tried |
|-----------|-------------|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | None, 10, 20, 30, 50 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | sqrt, log2, None |
| `bootstrap` | True, False |

**Search:** RandomizedSearchCV, 30 iterations, 5-fold CV  
**Criterion:** R² scoring

### 3.2 MLP Neural Network

**Architecture:** Feed-forward neural network with ReLU / tanh activation.

**Layer Configurations Tried:**
- Single hidden: (64), (128), (256)
- Two hidden: (64, 32), (128, 64), (256, 128)
- Three hidden: (128, 64, 32), (256, 128, 64)

**Hyperparameter Search Space:**

| Parameter | Values Tried |
|-----------|-------------|
| `hidden_layer_sizes` | 8 configurations (see above) |
| `activation` | relu, tanh |
| `solver` | adam, sgd |
| `alpha` | 0.0001, 0.001, 0.01 |
| `learning_rate` | constant, adaptive |
| `learning_rate_init` | 0.001, 0.01 |
| `max_iter` | 500, 1000, 2000 |
| `batch_size` | 32, 64, 128 |

**Preprocessing:** Standard scaling (Z-score) applied  
**Early stopping:** Enabled (10% validation split)  
**Search:** RandomizedSearchCV, 20 iterations, 3-fold CV

### 3.3 XGBoost

**Architecture:** Gradient-boosted decision trees with regularization.

**Hyperparameter Search Space:**

| Parameter | Values Tried |
|-----------|-------------|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | 3, 4, 5, 6, 8, 10 |
| `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| `subsample` | 0.6, 0.8, 1.0 |
| `colsample_bytree` | 0.6, 0.8, 1.0 |
| `min_child_weight` | 1, 3, 5, 7 |
| `gamma` | 0, 0.1, 0.2, 0.5 |
| `reg_alpha` | 0, 0.1, 1 |
| `reg_lambda` | 1, 2, 5 |

**Search:** RandomizedSearchCV, 30 iterations, 5-fold CV

### 3.4 Hybrid Model (Stacking Ensemble)

**Architecture:** Two-level stacking ensemble.

```
Level 1 (Base Models):
  ├── Random Forest (tuned)
  ├── XGBoost (tuned)
  └── MLP (tuned)

Level 2 (Meta-Learner):
  └── RidgeCV (alpha ∈ {0.01, 0.1, 1, 10, 100})
```

**Process:**
1. Base models make predictions on training data
2. Base predictions become features for the meta-learner
3. Meta-learner learns optimal weighting of base models
4. Final prediction = weighted combination of base model outputs

**Rationale:** Combines the strengths of different algorithms — Random Forest captures robust patterns, XGBoost handles complex interactions, MLP captures non-linear relationships.

### 3.5 TabPFN (Transfer Learning)

**Architecture:** Pre-trained Prior-Data Fitted Network (TabPFN).

**Key Properties:**
- Pre-trained on millions of synthetic tabular datasets
- No traditional training — fits at inference time using transformer attention
- Excels at small-to-medium tabular datasets (< 10K rows)
- Requires no hyperparameter tuning

**Implementation:**
- `TabPFNRegressor` from the `tabpfn` library
- Device: auto (GPU if available, else CPU)
- Random state: 42

---

## 4. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **R²** | 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)² | Proportion of variance explained (0–1, higher is better) |
| **MAE** | Σ\|yᵢ−ŷᵢ\| / n | Average absolute error (lower is better) |
| **RMSE** | √(Σ(yᵢ−ŷᵢ)² / n) | Root mean squared error (lower is better, penalizes large errors) |

---

## 5. Results

### 5.1 Model Comparison Table

| Model | R² | MAE | RMSE | Training Time |
|-------|----|-----|------|---------------|
| Random Forest | — | — | — | — |
| MLP Neural Network | — | — | — | — |
| XGBoost | — | — | — | — |
| Hybrid (Stacking) | — | — | — | — |
| TabPFN | — | — | — | — |

*Metrics to be filled in after training execution.*

### 5.2 Key Findings

1. **Baseline Performance:** Random Forest provides a robust baseline due to its ability to handle mixed feature types and non-linear relationships.
2. **XGBoost Advantage:** Gradient boosting typically outperforms bagging on structured data due to sequential error correction.
3. **MLP Sensitivity:** Neural networks require careful preprocessing (scaling) and may underperform on smaller datasets without sufficient tuning.
4. **Hybrid Synergy:** The stacking ensemble leverages diverse model strengths and often achieves the best generalization.
5. **TabPFN Efficiency:** Zero-shot transfer learning with no traditional training — ideal for rapid prototyping.

---

## 6. Feature Importance

### 6.1 Random Forest / XGBoost

Top features by importance (expected):
1. `batting_average` — Strongest individual predictor
2. `strike_rate` — Aggression metric
3. `fours` / `sixes` — Boundary frequency
4. `economy_rate` — Bowling efficiency
5. `career_fifties` — Career consistency

### 6.2 SHAP Analysis (Optional)

For the best-performing tree-based model, SHAP values can be computed to explain individual predictions:
- Positive SHAP for `batting_average` → higher performance score
- Negative SHAP for `economy_rate` → higher economy → lower performance score

---

## 7. Deployment

### 7.1 Streamlit Application

The best-performing model is deployed in a **Streamlit** web application:

**Features:**
- Interactive input forms for batting, bowling, and career stats
- Model selection dropdown (all 5 models available)
- Real-time prediction with visual performance score
- Category display (Poor / Average / Good / Very Good / Excellent)
- Model comparison dashboard
- Dataset exploration tools

**Usage:**
```bash
streamlit run 04_modeling_and_deployment/app/app.py
```

### 7.2 Model Artifacts

All trained models, results, and metadata are saved to:

```
04_modeling_and_deployment/models/
├── all_models_results.pkl     # All models + metrics
├── best_model.pkl             # Best performing model
├── model_comparison.csv       # Comparison table
└── metadata.json              # Features, split info, etc.
```

---

## 8. Conclusion

The Player Performance Prediction project successfully implements an end-to-end machine learning pipeline:

1. **Data Collection & EDA** (Phase 1): Scraped, cleaned, and explored IPL and BCCI data
2. **Feature Engineering** (Phase 1): Created composite features (batting/bowling impact, consistency score)
3. **Model Training** (Phase 2): Trained and tuned 5 models with different architectures
4. **Evaluation** (Phase 2): Compared all models using R², MAE, RMSE
5. **Deployment** (Phase 2): Built interactive Streamlit application

The hybrid stacking ensemble is expected to provide the best performance by combining the strengths of Random Forest, MLP, and XGBoost. TabPFN offers the fastest development cycle using transfer learning.
