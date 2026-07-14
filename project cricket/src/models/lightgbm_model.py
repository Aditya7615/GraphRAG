import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

LGB_PARAM_GRID = {
    'n_estimators': [100, 200, 500, 800, 1000, 1500],
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [-1, 4, 6, 8, 10, 12],
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [5, 10, 20, 30, 50],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.001, 0.01, 0.1, 0.5, 1],
}


def train_lightgbm(X_train, y_train, tune=True, random_state=42):
    import lightgbm as lgb

    if tune:
        base = lgb.LGBMRegressor(
            objective='regression', metric='rmse',
            boosting_type='gbdt', random_state=random_state,
            n_jobs=-1, verbose=-1,
        )
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=LGB_PARAM_GRID,
            n_iter=15,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': random_state,
            'n_jobs': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
        }
        model = lgb.LGBMRegressor(**params, n_estimators=2000, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_params = params

    return model, best_params


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }
