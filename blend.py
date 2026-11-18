"""Blend model predictions with non-negative weights fitted on OOF.
Also reports an honest K-fold estimate of the blend's generalization error.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from pipeline import get_matrices

_, _, _, _, y, test_ids, _ = get_matrices()

names, oofs, tsts = [], [], []
for f in sorted(glob.glob("preds/*.npz")):
    d = np.load(f, allow_pickle=True)
    names.append(os.path.basename(f)[:-4])
    oofs.append(d["oof"])
    tsts.append(d["tst"])

O = np.column_stack(oofs)   # (n, m)
T = np.column_stack(tsts)
m = O.shape[1]
print("models:", names)
for i, nm in enumerate(names):
    print(f"  {nm:6s} {np.sqrt(mean_squared_error(y, O[:, i])):.6f}")

print("\nOOF corr matrix:")
print(pd.DataFrame(np.corrcoef(O.T), index=names, columns=names).round(3).to_string())

# --- non-negative weights, no intercept ---
w, _ = nnls(O, y)
print("\nnnls weights:", dict(zip(names, np.round(w, 4))))
blend_oof = O @ w
rmse = float(np.sqrt(mean_squared_error(y, blend_oof)))
print(f"blend OOF RMSE = {rmse:.6f}  (sum w = {w.sum():.3f})")

# --- equal-weight reference ---
eq = O.mean(axis=1)
print(f"equal-weight OOF RMSE = {np.sqrt(mean_squared_error(y, eq)):.6f}")

# --- honest estimate: fit weights on 9/10, score on held-out 1/10 ---
kf = KFold(n_splits=10, shuffle=True, random_state=7)
cv_pred = np.zeros(len(y))
for tr, va in kf.split(O):
    ww, _ = nnls(O[tr], y[tr])
    cv_pred[va] = O[va] @ ww
print(f"nested-weight OOF RMSE = {np.sqrt(mean_squared_error(y, cv_pred)):.6f}")

# --- submission ---
blend_test = T @ w
pred = np.expm1(blend_test)
pred = np.clip(pred, 30000, 800000)
sub = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
sub.to_csv("submission_blend.csv", index=False)
print(f"\nsubmission_blend.csv  rows={len(sub)}  price range {pred.min():.0f}-{pred.max():.0f}")

# also save single best model submission for comparison
best_i = int(np.argmin([np.sqrt(mean_squared_error(y, O[:, i])) for i in range(m)]))
sub_b = pd.DataFrame({"Id": test_ids, "SalePrice": np.clip(np.expm1(T[:, best_i]), 30000, 800000)})
sub_b.to_csv(f"submission_{names[best_i]}.csv", index=False)
print(f"single-best = {names[best_i]}")

json.dump({"names": names, "weights": w.tolist(), "blend_oof_rmse": rmse,
           "nested_oof_rmse": float(np.sqrt(mean_squared_error(y, cv_pred)))},
          open("blend_weights.json", "w"), indent=2)
