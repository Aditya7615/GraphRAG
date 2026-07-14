# Model Card — Player Performance Predictor

## Model Details

- **Model Name:** Player Performance Predictor
- **Version:** 1.0
- **Type:** Ensemble of Random Forest, MLP, XGBoost, Hybrid Stacking, TabPFN
- **Date:** June 2026
- **Organization:** Group 1 — Team Project

### Model Architecture Options

| Model | Parameters | Framework |
|-------|-----------|-----------|
| Random Forest | n_estimators=100–500, max_depth=10–50 | scikit-learn |
| MLP | hidden_layers=(64–256), activation=ReLU | scikit-learn |
| XGBoost | n_estimators=100–500, max_depth=3–10, lr=0.01–0.2 | XGBoost |
| Hybrid Stacking | Base: RF+MLP+XGB, Meta: RidgeCV | scikit-learn |
| TabPFN | Pre-trained transformer, zero-shot | TabPFN |

## Intended Use

### Primary Use
Predict cricket player performance score (0–10) from historical batting and bowling statistics.

### Use Cases
- Player evaluation and scouting
- Team selection support
- Fantasy sports analytics
- Performance benchmarking

### Out-of-Scope
- Real-time match prediction
- Ball-by-ball analysis
- Injury prediction
- Auction price prediction
- Player ranking (Elo/ICC style)

## Training Data

- **Source:** IPL batting and bowling statistics (2008–2026)
- **Size:** 2,826 player-season records
- **Features:** 13 numeric features (batting average, strike rate, boundaries, wickets, economy, career milestones)
- **Target:** Overall Performance Score (continuous, 0–10)
- **Split:** 80% train / 20% test
- **Player Privacy:** All players identified by anonymous `player_id` (no names in training data)

## Performance

*To be updated after training execution.*

| Metric | Best Model | All Models |
|--------|-----------|------------|
| R² Score | TBD | Range TBD |
| MAE | TBD | Range TBD |
| RMSE | TBD | Range TBD |

### Performance by Category
- **Poor (0–3.0):** TBD
- **Average (3.0–4.5):** TBD
- **Good (4.5–6.0):** TBD
- **Very Good (6.0–7.5):** TBD
- **Excellent (7.5–10.0):** TBD

## Limitations

1. **Format-specific:** Trained only on IPL data — may not generalize directly to international formats (Test, ODI, T20I) without retraining
2. **Seasonal bias:** Performance can vary significantly year-to-year; model predicts expected performance based on historical patterns
3. **No contextual features:** Does not account for opposition quality, venue conditions, match situation, or team composition
4. **Data sparsity:** Players with few seasons have less reliable predictions
5. **Engineering assumptions:** Target variable is an engineered composite — not a ground-truth metric

## Ethical Considerations

- Predictions should be used as one input among many in decision-making, not as the sole determinant
- Player identities are anonymized (`player_id`) — no bias based on name, nationality, or personal attributes
- Model is a statistical tool — does not account for unquantifiable factors (form, mentality, team dynamics)
- Fairness: Performance categories are based on statistical thresholds, not subjective ratings

## Maintainance

- Retrain with each new IPL season to maintain relevance
- Monitor for data drift as playing styles evolve (e.g., increasing strike rates)
- Update feature engineering if new statistics become available

## References

- scikit-learn: https://scikit-learn.org
- XGBoost: https://xgboost.readthedocs.io
- TabPFN: https://github.com/automl/TabPFN
- Streamlit: https://streamlit.io
- IPL Data Source: IPL Sports Mechanic API
