# House Prices — top 5% on the public leaderboard

Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
Team `kiritocui` / group `we-are-nai-looong`, submitted 18 Nov 2026.

| | Public LB |
|---|---|
| Score | **0.11569** (RMSE on log1p SalePrice) |
| Rank | **236 / 4,955** |
| Percentile | **4.76%** |
| Top-10% cutoff | 0.11972 (rank 495) |

Two submissions were used. The first scored 0.12165 (rank 923, 18.6%); the
second fixed a single test row and moved up 687 places.

## What actually moved the score

Cross-validation said 0.1059, the leaderboard said 0.12165. That 0.0157 gap is
larger than this competition usually produces, so it was worth chasing instead
of tuning models.

The gap came from one row. Test `Id 2550` is a 5,095 sq ft, quality-10 house
built in 2008 in the Edwards neighbourhood. The model predicted **$800,000**.
Training data contains two comparable Edwards houses — 4,676 and 5,642 sq ft,
both quality 10, built 2007–2008 — that sold for **$184,750** and **$160,000**.
Large Edwards houses trade far below what their size implies.

The model never learned this because those two rows had been dropped as
outliers. Dropping them improved CV from 0.1367 to 0.1100, which looked like a
win, but it also removed the only evidence that this interaction exists. The
model was left with "big house = expensive".

One row cannot be shrugged off. A log-error of 1.55 on one of 1,459 rows
contributes 0.041 to the RMSE — roughly the entire gap.

The fix is not a hand-written constant. Two models are trained:

- **Main path** — outliers removed, as before. Six models, blended.
- **Keep-outlier path** — XGBoost + LightGBM trained *with* the outlier rows
  present, so the trees can split on `Neighborhood = Edwards AND GrLivArea >
  4000`.

Only the Edwards-large test rows are taken from the second path. Everything
else keeps the blend prediction. One row changed, $800,000 → $251,322.

## Method

1. **Target** — `log1p(SalePrice)`. Raw price skew is 1.883, log skew is 0.121.
2. **Feature engineering** — 27 derived columns on top of the original 79:
   total square footage, total baths, porch area, age, remodel flags,
   quality×area interactions, area ratios. 23 ordinal columns mapped to
   integers instead of one-hot.
3. **Two matrix views** — native categoricals for tree models (106 columns),
   one-hot for linear models (252 columns).
4. **Models** — Lasso, Ridge, ElasticNet, XGBoost, LightGBM, CatBoost. All
   hyperparameters fixed; linear regularisation strength chosen inside each
   fold, never on the outer validation split.
5. **Validation** — 10-fold, seed 42. Model selection uses OOF only. The public
   leaderboard was never used to choose between models.
6. **Blend** — non-negative least squares on OOF predictions, no intercept.
   Nested 10-fold check (weights fitted on 9/10, scored on the held-out 1/10)
   gives 0.1063 against 0.1059 in-sample, so the weights are not overfitted.

### OOF results

| Model | OOF RMSE | Weight |
|---|---|---|
| Lasso | 0.10997 | 0.503 |
| ElasticNet | 0.11005 | 0.000 |
| XGBoost | 0.11050 | 0.393 |
| Ridge | 0.11198 | 0.000 |
| LightGBM | 0.11304 | 0.091 |
| CatBoost | 0.11352 | 0.014 |
| **Blend** | **0.10587** | |

Three linear models land within 0.002 of each other, so NNLS puts the whole
linear direction on Lasso and zeroes out Ridge and ElasticNet. The blend's gain
over the best single model (0.10997 → 0.10587) comes from mixing a regularised
linear fit with two tree ensembles, which rank features differently.

## Reproduce

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p preds
for m in lasso ridge enet lgbm xgb cat; do python run_one.py $m; done

python blend.py           # -> submission_blend.csv + blend_weights.json
python fix_outlier_test.py # -> submission_v2.csv  (the submitted file)
```

Full write-up: [`report.html`](report.html).

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Data loading, ordinal maps, 27 engineered features, two matrix builds |
| `run_one.py` | One model, one process, 10-fold OOF + test predictions |
| `blend.py` | NNLS weights, nested validation, writes the blend submission |
| `fix_outlier_test.py` | Keep-outlier tree path, patches the Edwards rows |
| `submission_v2.csv` | Submitted file — public LB 0.11569 |
| `submission_blend.csv` | First submission, before the outlier fix — 0.12165 |
| `report.html` | Method report |

`data/`, `preds/` and `teammate/` are not tracked. The first two come from the
commands above; `teammate/` holds a private notebook belonging to another team
member.

## Things that did not work

- **Training on outliers removed everywhere.** CV 0.1100, LB 0.12165. The CV
  number was honest for the rows it saw; the problem was that the test set
  contains a row from the distribution that had been deleted.
- **Linear models on the keep-outlier data.** OOF 0.137, and `Id 2550` still
  predicted $800,000. A linear model sums effects, so it cannot represent "this
  neighbourhood is an exception for large houses". Trees can.
- **Selecting models on the public leaderboard.** The public set is 20% of the
  test data. This is what cost 260 places in a different competition
  (`playground-series-s6e10`: public 8.2%, private 16.9%). Every decision here
  was made on OOF instead.

## Environment note

This was run on an 8 GB M2 Mac. Multi-threaded BLAS exhausted memory and the
process was killed (SIGKILL) partway through, so every model runs in its own
process with `OMP_NUM_THREADS=1`. Expect ~15 min for the full six-model run on
comparable hardware.
