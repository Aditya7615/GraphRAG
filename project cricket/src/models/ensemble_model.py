import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def find_blend_weight(y_val, pred1, pred2, metric='r2'):
    best_w = 0.5
    best_score = -np.inf
    for w in np.linspace(0, 1, 51):
        blended = w * pred1 + (1 - w) * pred2
        if metric == 'r2':
            score = r2_score(y_val, blended)
        else:
            score = -mean_absolute_error(y_val, blended)
        if score > best_score:
            best_score = score
            best_w = w
    return best_w, best_score


class EnsembleRegressor:
    def __init__(self, model1, model2, weight=0.5):
        self.model1 = model1
        self.model2 = model2
        self.weight = weight

    def predict(self, X):
        p1 = self.model1.predict(X)
        p2 = self.model2.predict(X)
        return self.weight * p1 + (1 - self.weight) * p2


def train_ensemble(X_train, y_train, train_fn1, train_fn2, val_split=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    n = len(X_train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=random_state,
    )
    model1, _ = train_fn1(X_tr, y_tr)
    model2, _ = train_fn2(X_tr, y_tr)
    p1_val = model1.predict(X_val)
    p2_val = model2.predict(X_val)
    w, _ = find_blend_weight(y_val, p1_val, p2_val)
    model1_full, _ = train_fn1(X_train, y_train)
    model2_full, _ = train_fn2(X_train, y_train)
    ensemble = EnsembleRegressor(model1_full, model2_full, weight=w)
    return ensemble, {'weight': w, 'model1': type(model1).__name__, 'model2': type(model2).__name__}


def evaluate_ensemble(ensemble, X_test, y_test):
    y_pred = ensemble.predict(X_test)
    return {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }
