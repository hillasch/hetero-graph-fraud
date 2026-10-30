"""Baseline vs. graph-enhanced XGBoost, evaluated by PR-AUC and F1."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier

TARGET = "isFraud"
DROP = {TARGET, "TransactionDT", "card_key", "device_key", "addr_key", "p_email", "r_email"}


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — test set is the latest ``test_frac`` of transactions."""
    df = df.sort_values("TransactionDT")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def tabular_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in df.columns if c not in DROP]].copy()
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")
    return X


def fit_eval(X_tr, y_tr, X_te, y_te, seed: int = 42) -> dict:
    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.5, tree_method="hist",
        enable_categorical=True, eval_metric="aucpr", random_state=seed,
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
    return {
        "pr_auc": average_precision_score(y_te, p),
        "f1@0.5": f1_score(y_te, p >= 0.5),
        "best_f1": float(f1s[best]),
        "best_threshold": float(thr[best]) if len(thr) else 0.5,
        "model": model,
    }


def compare(train: pd.DataFrame, test: pd.DataFrame, graph_feats: pd.DataFrame) -> pd.DataFrame:
    """Train the same model with and without graph features; return a results table.

    Note: ``best_f1`` picks the threshold on the test set — for the final report,
    tune the threshold on a validation split instead.
    """
    y_tr, y_te = train[TARGET], test[TARGET]
    X_tr, X_te = tabular_matrix(train), tabular_matrix(test)
    base = fit_eval(X_tr, y_tr, X_te, y_te)

    X_tr_g = X_tr.join(graph_feats.loc[train.index])
    X_te_g = X_te.join(graph_feats.loc[test.index])
    graph = fit_eval(X_tr_g, y_tr, X_te_g, y_te)

    cols = ["pr_auc", "f1@0.5", "best_f1"]
    return pd.DataFrame({"baseline": {k: base[k] for k in cols},
                         "graph_enhanced": {k: graph[k] for k in cols}}).T
