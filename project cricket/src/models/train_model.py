import os, sys, json, joblib, time, warnings
import numpy as np
import pandas as pd

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.data_utils import load_feature_data, prepare_data, get_feature_columns
from src.features.data_utils import FEATURE_SETS, classify_player
from sklearn.model_selection import train_test_split
from src.models.xgboost_model import train_xgboost, evaluate_model as eval_xgb
from src.models.lightgbm_model import train_lightgbm, evaluate_model as eval_lgb
from src.models.ensemble_model import train_ensemble, evaluate_ensemble, EnsembleRegressor

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
ROLE_DIR = os.path.join(MODELS_DIR, 'role_models')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ROLE_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("PLAYER PERFORMANCE PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    df = load_feature_data()
    print(f"\nData: {df.shape[0]:,} rows, {df.shape[1]} cols, {df['player_id'].nunique()} players")

    X_full, X_test, y_train, y_test, feat_cols, _ = prepare_data(df)
    X_train, X_val, y_tr, y_val = train_test_split(
        X_full, y_train, test_size=0.2, random_state=42,
    )
    print(f"    Features ({len(feat_cols)}): {feat_cols}")
    print(f"    Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── 1. General XGBoost ──
    print("\n[1] Training XGBoost (general)...")
    t0 = time.time()
    xgb_model, xgb_params = train_xgboost(X_full, y_train, tune=True)
    xgb_metrics = eval_xgb(xgb_model, X_test, y_test)
    xgb_time = time.time() - t0
    print(f"    R²={xgb_metrics['r2_score']:.4f}  MAE={xgb_metrics['mae']:.4f}  ({xgb_time:.1f}s)")

    # ── 2. General LightGBM ──
    print("\n[2] Training LightGBM (general)...")
    t0 = time.time()
    lgb_model, lgb_params = train_lightgbm(X_full.values, y_train, tune=True)
    lgb_metrics = eval_lgb(lgb_model, X_test.values, y_test)
    lgb_time = time.time() - t0
    print(f"    R²={lgb_metrics['r2_score']:.4f}  MAE={lgb_metrics['mae']:.4f}  ({lgb_time:.1f}s)")

    # ── 3. General Ensemble (XGBoost + LightGBM) ──
    print("\n[3] Training Ensemble (XGBoost + LightGBM)...")
    t0 = time.time()
    ensemble_model, ensemble_info = train_ensemble(
        X_full.values, y_train,
        lambda x, y: train_xgboost(
            pd.DataFrame(x, columns=feat_cols), y, tune=False,
        ),
        lambda x, y: train_lightgbm(x, y, tune=False),
        val_split=0.2, random_state=42,
    )
    ens_metrics = evaluate_ensemble(ensemble_model, X_test.values, y_test)
    ens_time = time.time() - t0
    print(f"    R²={ens_metrics['r2_score']:.4f}  MAE={ens_metrics['mae']:.4f}  "
          f"Blend={ensemble_info['weight']:.3f}  ({ens_time:.1f}s)")

    # ── 4. Role-aware LightGBM + Role Ensembles ──
    print("\n[4] Training role-aware models...")
    df_copy = df.copy()
    df_copy['player_role'] = df_copy.apply(classify_player, axis=1)
    role_counts = df_copy['player_role'].value_counts()
    print(f"    Player breakdown: {role_counts.to_dict()}")

    role_results = {}
    role_ensemble_results = {}

    for role in ['batsman', 'bowler', 'all_rounder']:
        print(f"\n    [{role}]...", end=" ", flush=True)
        feat_cols_role = FEATURE_SETS[role]
        role_df = df_copy[df_copy['player_role'] == role]

        if len(role_df) < 10:
            print(f"skipped ({len(role_df)} rows)")
            continue

        X_r = role_df[feat_cols_role].fillna(0)
        y_r = role_df['overall_performance_score'].values
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

        # Role-specific LightGBM
        t0 = time.time()
        role_lgb, _ = train_lightgbm(Xr_tr.values, yr_tr, tune=False, random_state=42)
        role_metrics = eval_lgb(role_lgb, Xr_te.values, yr_te)
        rlgb_time = time.time() - t0
        print(f"LGB R²={role_metrics['r2_score']:.4f}  MAE={role_metrics['mae']:.4f}  ({rlgb_time:.1f}s)")

        role_results[role] = {
            'model': role_lgb,
            'metrics': role_metrics,
            'features': feat_cols_role,
            'time': rlgb_time,
            'train_samples': len(Xr_tr),
            'test_samples': len(Xr_te),
        }

        role_path = os.path.join(ROLE_DIR, role)
        os.makedirs(role_path, exist_ok=True)
        joblib.dump(role_lgb, os.path.join(role_path, 'lgb_model.pkl'))
        meta = {
            'model': 'LightGBM',
            'features': feat_cols_role,
            'r2_score': role_metrics['r2_score'],
            'mae': role_metrics['mae'],
            'rmse': role_metrics['rmse'],
            'train_samples': len(Xr_tr),
            'test_samples': len(Xr_te),
        }
        with open(os.path.join(role_path, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Role-specific XGBoost
        t0 = time.time()
        role_xgb, _ = train_xgboost(Xr_tr, yr_tr, tune=False)
        rxgb_metrics = eval_xgb(role_xgb, Xr_te, yr_te)
        rxgb_time = time.time() - t0
        print(f"           XGB R²={rxgb_metrics['r2_score']:.4f}  MAE={rxgb_metrics['mae']:.4f}  ({rxgb_time:.1f}s)")
        joblib.dump(role_xgb, os.path.join(role_path, 'xgb_model.pkl'))

        # Role-specific Ensemble (XGBoost + LightGBM)
        t0 = time.time()
        role_ensemble = EnsembleRegressor(role_xgb, role_lgb, weight=0.5)
        # Find optimal blend weight on val split of role data
        Xr_tr2, Xr_v, yr_tr2, yr_v = train_test_split(
            Xr_tr, yr_tr, test_size=0.2, random_state=42,
        )
        role_xgb_v, _ = train_xgboost(Xr_tr2, yr_tr2, tune=False)
        role_lgb_v, _ = train_lightgbm(Xr_tr2.values, yr_tr2, tune=False, random_state=42)
        p_xgb_v = role_xgb_v.predict(Xr_v)
        p_lgb_v = role_lgb_v.predict(Xr_v.values)
        best_w = 0.5
        best_r2 = -np.inf
        for w in np.linspace(0, 1, 51):
            blended = w * p_xgb_v + (1 - w) * p_lgb_v
            from sklearn.metrics import r2_score
            s = r2_score(yr_v, blended)
            if s > best_r2:
                best_r2 = s
                best_w = w
        role_ensemble = EnsembleRegressor(role_xgb, role_lgb, weight=best_w)
        rens_metrics = evaluate_ensemble(role_ensemble, Xr_te.values, yr_te)
        rens_time = time.time() - t0
        print(f"           ENS R²={rens_metrics['r2_score']:.4f}  MAE={rens_metrics['mae']:.4f}  "
              f"Blend={best_w:.3f}  ({rens_time:.1f}s)")

        role_ensemble_results[role] = {
            'model': role_ensemble,
            'metrics': rens_metrics,
            'features': feat_cols_role,
            'weight': best_w,
            'time': rens_time,
        }
        joblib.dump(role_ensemble, os.path.join(role_path, 'ensemble_model.pkl'))
        ens_meta = {
            'model': 'Ensemble',
            'features': feat_cols_role,
            'r2_score': rens_metrics['r2_score'],
            'mae': rens_metrics['mae'],
            'rmse': rens_metrics['rmse'],
            'blend_weight': best_w,
            'train_samples': len(Xr_tr),
            'test_samples': len(Xr_te),
        }
        with open(os.path.join(role_path, 'ensemble_metadata.json'), 'w') as f:
            json.dump(ens_meta, f, indent=2)

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    comparison = []
    for model_name, metrics, t in [
        ('XGBoost', xgb_metrics, xgb_time),
        ('LightGBM', lgb_metrics, lgb_time),
        ('Ensemble', ens_metrics, ens_time),
    ]:
        comparison.append({
            'Model': model_name,
            'R²': round(metrics['r2_score'], 4),
            'MAE': round(metrics['mae'], 4),
            'RMSE': round(metrics['rmse'], 4),
            'Time (s)': round(t, 1),
        })
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    print("\nRole-aware models:")
    role_meta = {}
    for role, data in role_results.items():
        m = data['metrics']
        print(f"  {role:12s} LGB R²={m['r2_score']:.4f}  MAE={m['mae']:.4f}")
        role_meta[role] = {
            'lgb': {
                'r2_score': m['r2_score'],
                'mae': m['mae'],
                'rmse': m['rmse'],
                'features': data['features'],
            }
        }
    for role, data in role_ensemble_results.items():
        m = data['metrics']
        print(f"  {'':12s}  ENS R²={m['r2_score']:.4f}  MAE={m['mae']:.4f}  (w={data['weight']:.2f})")
        role_meta[role]['ensemble'] = {
            'r2_score': m['r2_score'],
            'mae': m['mae'],
            'rmse': m['rmse'],
            'weight': data['weight'],
            'features': data['features'],
        }

    # ── Save ──
    print("\n[5] Saving...")
    best_model_key = 'Ensemble' if ens_metrics['r2_score'] >= max(
        xgb_metrics['r2_score'], lgb_metrics['r2_score'],
    ) else ('XGBoost' if xgb_metrics['r2_score'] >= lgb_metrics['r2_score'] else 'LightGBM')

    results = {
        'XGBoost': {'model': xgb_model, 'params': xgb_params, 'metrics': xgb_metrics, 'time': xgb_time},
        'LightGBM': {'model': lgb_model, 'params': lgb_params, 'metrics': lgb_metrics, 'time': lgb_time},
        'Ensemble': {'model': ensemble_model, 'params': ensemble_info, 'metrics': ens_metrics, 'time': ens_time},
    }
    joblib.dump(results, os.path.join(MODELS_DIR, 'all_models_results.pkl'))
    comp_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)
    best_model = results[best_model_key]['model']
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))

    with open(os.path.join(ROLE_DIR, 'role_models_metadata.json'), 'w') as f:
        json.dump(role_meta, f, indent=2)

    metadata = {
        'best_model_name': best_model_key,
        'features': feat_cols,
        'train_samples': len(X_full),
        'test_samples': len(X_test),
        'n_features': len(feat_cols),
        'comparison': comparison,
        'role_lgb': {r: role_meta[r]['lgb'] for r in role_meta},
        'role_ensemble': {r: role_meta[r]['ensemble'] for r in role_meta if 'ensemble' in role_meta[r]},
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"    Saved to {MODELS_DIR}")
    print(f"    Best model: {best_model_key}")
    print("\nDone!")


if __name__ == '__main__':
    main()
