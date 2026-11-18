"""修正 test 中"Edwards 区异常低价豪宅"这一类样本。

主模型用 remove_outliers=True 训练（整体 CV 更好），但它会把
GrLivArea>4000 的样本外推到 80 万量级。
这里用 remove_outliers=False 训练的树模型（能学到 Edwards 交互）单独预测这类样本，
其余样本保持主模型预测不变。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb

from pipeline import get_matrices

SEED = 42
NFOLD = 10

Xtr, Xte, _, _, y, test_ids, cat_cols = get_matrices(remove_outliers=False)
folds = list(KFold(n_splits=NFOLD, shuffle=True, random_state=SEED).split(Xtr))
n = len(y)
print(f"keep-outlier training rows = {n}", flush=True)

preds = []

xp = dict(learning_rate=0.02, max_depth=3, min_child_weight=2, subsample=0.7,
          colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=1,
          tree_method="hist", enable_categorical=True)
tp = np.zeros(len(Xte))
for tr, va in folds:
    m = xgb.XGBRegressor(**xp, n_estimators=4000, early_stopping_rounds=150, eval_metric="rmse")
    m.fit(Xtr.iloc[tr], y[tr], eval_set=[(Xtr.iloc[va], y[va])], verbose=False)
    tp += m.predict(Xte) / NFOLD
preds.append(tp)
print("xgb(keep) done", flush=True)

lp = dict(learning_rate=0.02, num_leaves=8, max_depth=4, min_child_samples=5,
          subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
          reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=1, verbosity=-1)
tp = np.zeros(len(Xte))
for tr, va in folds:
    m = lgb.LGBMRegressor(**lp, n_estimators=4000)
    m.fit(Xtr.iloc[tr], y[tr], eval_set=[(Xtr.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    tp += m.predict(Xte) / NFOLD
preds.append(tp)
print("lgbm(keep) done", flush=True)

keep_log = np.mean(preds, axis=0)

# 主预测（remove_outliers=True 的 blend）
sub = pd.read_csv("submission_blend.csv")
main_log = np.log1p(sub["SalePrice"].values)

test = pd.read_csv("data/test.csv")
target = test[(test.Neighborhood == "Edwards") & (test.GrLivArea > 4000)]
idx = [list(sub["Id"]).index(i) for i in target["Id"]]
print(f"\n修正样本: {target[['Id','GrLivArea','OverallQual']].to_dict('records')}")
for j in idx:
    print(f"  Id {sub['Id'].iloc[j]}: main={sub['SalePrice'].iloc[j]:,.0f} -> keep-outlier model={np.expm1(keep_log[j]):,.0f}")

fixed = sub.copy()
fixed.loc[idx, "SalePrice"] = np.expm1(keep_log[idx])
fixed.to_csv("submission_v2.csv", index=False)
print(f"\nsubmission_v2.csv written, changed {len(idx)} row(s)")
print(f"price range {fixed.SalePrice.min():,.0f} - {fixed.SalePrice.max():,.0f}")
