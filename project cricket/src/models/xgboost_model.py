import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
    'colsample_bynode': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.05, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 2, 5],
}


def train_xgboost(X_train, y_train, tune=True, n_iter=20, cv=3, random_state=42):
    if tune:
        search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(random_state=random_state, verbosity=0, n_jobs=1),
            param_distributions=XGB_PARAM_GRID,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=1,
            random_state=random_state,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbosity=0, n_jobs=1,
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()

    return model, best_params


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }
