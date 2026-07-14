# Test Plan — Phase 2: Model Training & Deployment

**Project:** Group 1 — Player Performance Prediction  
**Version:** 2.0 (Phase 2)  
**Date:** June 2026

---

## 1. Test Strategy

Verification of model training, evaluation, and deployment pipeline correctness.

## 2. Test Scope

### In-Scope
- Data loading and feature selection
- Train-test split correctness
- Model training (all 5 architectures)
- Hyperparameter tuning execution
- Model evaluation metrics computation
- Model persistence (save/load)
- Streamlit app functionality

### Out-of-Scope
- Real-time performance testing
- Production deployment testing
- Cross-validation consistency (Phase 3)

## 3. Test Environment

| Component | Specification |
|-----------|---------------|
| Python | 3.9+ |
| Dependencies | `04_modeling_and_deployment/requirements.txt` |
| Data | `03_eda_cleaned/ipl_features.csv` |

## 4. Test Cases

### TC-MODEL-001: Random Forest Training
- **Description:** Verify Random Forest trains and produces predictions
- **Steps:** Run `train_all_models.py` → check `Random Forest` in results
- **Expected:** Model trains without error; R² > 0
- **Status:** ⏳ Pending

### TC-MODEL-002: MLP Training
- **Description:** Verify MLP Neural Network trains
- **Steps:** Run `train_all_models.py` → check `MLP` in results
- **Expected:** Model trains without error; R² > 0
- **Status:** ⏳ Pending

### TC-MODEL-003: XGBoost Training
- **Description:** Verify XGBoost trains
- **Steps:** Run `train_all_models.py` → check `XGBoost` in results
- **Expected:** Model trains without error; R² > 0
- **Status:** ⏳ Pending

### TC-MODEL-004: Hybrid Model Training
- **Description:** Verify Hybrid stacking ensemble trains
- **Steps:** Run `train_all_models.py` → check `Hybrid` in results
- **Expected:** Model trains without error; predictions valid
- **Status:** ⏳ Pending

### TC-MODEL-005: TabPFN Training
- **Description:** Verify TabPFN transfer learning model runs
- **Steps:** Run `train_all_models.py` → check `TabPFN` in results
- **Expected:** Model loads/prepares without error; predictions valid
- **Status:** ⏳ Pending

### TC-MODEL-006: Metrics Computation
- **Description:** Verify all metrics computed correctly
- **Steps:** Check output of each model's `evaluate_model()`
- **Expected:** R² in [-∞, 1.0], MAE ≥ 0, RMSE ≥ 0
- **Status:** ⏳ Pending

### TC-MODEL-007: Model Persistence
- **Description:** Verify models save and load correctly
- **Steps:**
  1. Run training → check `04_modeling_and_deployment/models/`
  2. Load `all_models_results.pkl` with `joblib.load()`
- **Expected:** All 4 files present; loaded object contains all models
- **Status:** ⏳ Pending

### TC-DEPLOY-001: Streamlit App Launches
- **Description:** Verify Streamlit app starts
- **Steps:** `streamlit run 04_modeling_and_deployment/app/app.py`
- **Expected:** App opens in browser; model comparison visible
- **Status:** ⏳ Pending

### TC-DEPLOY-002: Prediction Workflow
- **Description:** Verify end-to-end prediction in Streamlit
- **Steps:**
  1. Select model
  2. Input stats
  3. Click "Predict Performance"
- **Expected:** Score and category displayed
- **Status:** ⏳ Pending

### TC-DEPLOY-003: Model Selection
- **Description:** Verify all 5 models selectable in app
- **Steps:** Cycle through each model in dropdown → predict
- **Expected:** Each model produces valid prediction
- **Status:** ⏳ Pending

## 5. Test Execution

```bash
# Train models
python "04_modeling_and_deployment/scripts/train_all_models.py"

# Launch app
streamlit run "04_modeling_and_deployment/app/app.py"
```

## 6. Pass/Fail Criteria

| Criterion | Threshold |
|-----------|-----------|
| All 5 models train successfully | 100% |
| All metrics in valid ranges | 100% |
| Prediction within [0, 10] | 100% |
| Streamlit app launches | ✅ |
| All model comparison renders | ✅ |

## 7. Test Deliverables

- `04_modeling_and_deployment/scripts/train_all_models.py` — Training orchestrator
- `04_modeling_and_deployment/models/all_models_results.pkl` — Trained artifacts
- `04_modeling_and_deployment/models/model_comparison.csv` — Results table
- `04_modeling_and_deployment/app/app.py` — Deployment app
- This test plan document
