"""Run a single model in its own process (memory isolation), save OOF + test preds.

usage: python run_one.py <model_name>
"""
import sys
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from pipeline import get_matrices

SEED = 42
NFOLD = 10
NAME = sys.argv[1]

Xtr_t, Xte_t, Xtr_l, Xte_l, y, test_ids, cat_cols = get_matrices()
folds = list(KFold(n_splits=NFOLD, shuffle=True, random_state=SEED).split(Xtr_t))
n = len(y)

if NAME in ("lasso", "ridge", "enet"):
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    makers = {
        "lasso": lambda: make_pipeline(RobustScaler(), LassoCV(
            alphas=np.logspace(-5, -1, 40), cv=5, selection="random",
            max_iter=10000, random_state=SEED)),
        "ridge": lambda: make_pipeline(RobustScaler(), RidgeCV(
            alphas=np.logspace(-2, 3, 40))),
        "enet": lambda: make_pipeline(RobustScaler(), ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 1.0], alphas=np.logspace(-5, -1, 25), cv=4,
            selection="random", max_iter=10000, random_state=SEED)),
    }
    X, Xte = Xtr_l, Xte_l
    t0 = time.time()
    o = np.zeros(n)
    t = np.zeros(len(Xte))
    for tr, va in folds:
        est = makers[NAME]()
        est.fit(X.iloc[tr], y[tr])
        o[va] = est.predict(X.iloc[va])
        t += est.predict(Xte) / NFOLD

elif NAME == "lgbm":
    import lightgbm as lgb
    p = dict(learning_rate=0.02, num_leaves=8, max_depth=4, min_child_samples=5,
             subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
             reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=1, verbosity=-1)
    t0 = time.time()
    o = np.zeros(n)
    t = np.zeros(len(Xte_t))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**p, n_estimators=4000)
        m.fit(Xtr_t.iloc[tr], y[tr], eval_set=[(Xtr_t.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        o[va] = m.predict(Xtr_t.iloc[va])
        t += m.predict(Xte_t) / NFOLD

elif NAME == "xgb":
    import xgboost as xgb
    p = dict(learning_rate=0.02, max_depth=3, min_child_weight=2, subsample=0.7,
             colsample_bytree=0.5, reg_lambda=1.0, reg_alpha=0.0, random_state=SEED,
             n_jobs=1, tree_method="hist", enable_categorical=True)
    t0 = time.time()
    o = np.zeros(n)
    t = np.zeros(len(Xte_t))
    for tr, va in folds:
        m = xgb.XGBRegressor(**p, n_estimators=4000, early_stopping_rounds=150,
                             eval_metric="rmse")
        m.fit(Xtr_t.iloc[tr], y[tr], eval_set=[(Xtr_t.iloc[va], y[va])], verbose=False)
        o[va] = m.predict(Xtr_t.iloc[va])
        t += m.predict(Xte_t) / NFOLD

elif NAME == "cat":
    from catboost import CatBoostRegressor
    Xc = Xtr_t.copy()
    Xtc = Xte_t.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str)
        Xtc[c] = Xtc[c].astype(str)
    cat_idx = [Xc.columns.get_loc(c) for c in cat_cols]
    t0 = time.time()
    o = np.zeros(n)
    t = np.zeros(len(Xtc))
    for tr, va in folds:
        m = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=5,
                              l2_leaf_reg=5.0, loss_function="RMSE", random_seed=SEED,
                              thread_count=1, verbose=0, early_stopping_rounds=150)
        m.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx, eval_set=(Xc.iloc[va], y[va]))
        o[va] = m.predict(Xc.iloc[va])
        t += m.predict(Xtc) / NFOLD
else:
    raise SystemExit(f"unknown model {NAME}")

rmse = float(np.sqrt(mean_squared_error(y, o)))
fs = [float(np.sqrt(mean_squared_error(y[va], o[va]))) for _, va in folds]
np.savez(f"preds/{NAME}.npz", oof=o, tst=t, rmse=rmse, fold_std=float(np.std(fs)))
print(f"{NAME:6s} OOF={rmse:.6f}  fold_std={np.std(fs):.6f}  took {time.time()-t0:.0f}s", flush=True)
