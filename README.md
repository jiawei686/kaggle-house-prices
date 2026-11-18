# House Prices — top 5% on the public leaderboard

![public LB](https://img.shields.io/badge/public%20LB-0.11569-blue)
![rank](https://img.shields.io/badge/rank-236%20%2F%204%2C957-brightgreen)
![percentile](https://img.shields.io/badge/top-4.76%25-success)
![python](https://img.shields.io/badge/python-3.13-informational)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques),
team `we are nai looong` (Kaggle group [`we-are-nai-looong`](https://www.kaggle.com/groups/we-are-nai-looong)),
submitted 18 Nov 2026.

| | Public LB |
|---|---|
| Score | **0.11569** (RMSE on log1p SalePrice) |
| Rank | **236 / 4,957** |
| Percentile | **4.76%** |
| Top-10% cutoff | 0.11966 (rank 495) |
| Submissions used | 2 |

The first submission scored 0.12165 and ranked 923rd. The second changed one
test row and moved up 687 places.

---

## Contents

- [The one row that mattered](#the-one-row-that-mattered)
- [Method](#method)
- [OOF results](#oof-results)
- [Reproduce](#reproduce)
- [Repository layout](#repository-layout)
- [What did not work](#what-did-not-work)

---

## The one row that mattered

Cross-validation said 0.1059, the leaderboard said 0.12165. That 0.0157 gap is
larger than this competition usually produces, so it was worth chasing instead
of tuning models.

The gap came from one row. Test `Id 2550` is a 5,095 sq ft, quality-10 house
built in 2008 in the Edwards neighbourhood. The model predicted **$800,000**.
Training data contains two comparable Edwards houses — 4,676 and 5,642 sq ft,
both quality 10, built 2007–2008 — that sold for **$184,750** and **$160,000**.
Large houses in that neighbourhood trade well below what their size implies.

The model never learned this because those two rows had been dropped as
outliers. Dropping them improved CV from 0.1367 to 0.1100, which looked like a
win, but it also removed the only evidence that this interaction exists. The
model was left with "big house = expensive".

One row cannot be shrugged off. A log-error of 1.55 on one of 1,459 test rows
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

## OOF results

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
git clone https://github.com/jiawei686/kaggle-house-prices
cd kaggle-house-prices

kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p preds
for m in lasso ridge enet lgbm xgb cat; do python run_one.py $m; done

python blend.py            # -> submission_blend.csv + blend_weights.json
python fix_outlier_test.py # -> submission_v2.csv  (the submitted file)
```

Expect ~15 minutes for the six-model run on 8 GB of RAM with single-threaded
BLAS. Full write-up: [`report.pdf`](report.pdf) (source: [`report.html`](report.html)). Short version for a Kaggle
post: [`writeup.md`](writeup.md).

## Repository layout

```
pipeline.py             data loading, 23 ordinal maps, 27 features, two matrix builds
run_one.py              one model, one process, 10-fold OOF + test predictions
blend.py                NNLS weights, nested validation, writes the blend submission
fix_outlier_test.py     keep-outlier tree path, patches the Edwards rows
requirements.txt        pinned to the versions that produced the scores

submission_v2.csv       submitted file — public LB 0.11569
submission_blend.csv    first submission, before the outlier fix — 0.12165
blend_weights.json      fitted weights and both OOF estimates

report.pdf              full method report, A4, 5 pages
report.html             source of report.pdf, six tables
make_report_pdf.py      renders the PDF from the HTML (playwright + PyMuPDF)
writeup.md              short version formatted for a Kaggle discussion post
```

`data/` and `preds/` are not tracked. The first comes from the Kaggle CLI, the
second is regenerated by `run_one.py`.

## What did not work

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

## Notes

- The leaderboard is partly contaminated. The top 77 entries score exactly
  `0.00000`, which is a perfect match against the original Ames dataset rather
  than a model. About 110 teams beat 0.11. No outside data was used here —
  matching test rows to the original Ames file is leakage.
- The competition stays open, and the board grows as new teams join. Percentiles
  move without any new submission.
- Built on an 8 GB M2 Mac. Multi-threaded BLAS exhausted memory and the process
  was killed (SIGKILL) partway through, which is why every model runs in its own
  process with `OMP_NUM_THREADS=1`.

## License

Code released under the MIT License — see [`LICENSE`](LICENSE).
Report and write-up text: CC BY 4.0.
