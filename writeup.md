# House Prices: top 5% (0.11569), and the one row that mattered

*Team `kiritocui` — Kaggle group [`we-are-nai-looong`](https://www.kaggle.com/groups/we-are-nai-looong)*

Public LB **0.11569**, rank **236 / 4,955** (4.76%). Took two submissions.

| Model | OOF RMSE | Weight |
|---|---|---|
| Lasso | 0.10997 | 0.503 |
| XGBoost | 0.11050 | 0.393 |
| LightGBM | 0.11304 | 0.091 |
| CatBoost | 0.11352 | 0.014 |
| Ridge / ElasticNet | 0.11198 / 0.11005 | 0.000 |
| **Blend (NNLS)** | **0.10587** | |

The first submission scored 0.12165 and ranked 923rd. Cross-validation had
said 0.1059, so the 0.0157 gap was worth chasing rather than tuning.

It came from a single test row. `Id 2550` is a 5,095 sq ft, quality-10 house in
Edwards. The model said $800,000. Training data has two comparable Edwards
houses — 4,676 and 5,642 sq ft, same quality, built 2007–2008 — that sold for
$184,750 and $160,000. Large houses in that neighbourhood trade well below what
their size implies.

The model missed this because those two rows had been dropped as outliers.
Dropping them improved CV (0.1367 → 0.1100), which looked like a win, but it
also deleted the only evidence that the interaction exists. One log-error of
1.55 on 1,459 rows contributes 0.041 to RMSE — about the whole gap.

Fix: keep the outlier-aware path separate. Train XGBoost and LightGBM *with*
those rows so the trees can split on `Neighborhood = Edwards AND GrLivArea >
4000`, then use them only for the Edwards-large rows. Main predictions
untouched. One row changed, $800,000 → $251,322. Rank moved 923 → 236.

Code, report and both submissions: [github.com/jiawei686/kaggle-house-prices](https://github.com/jiawei686/kaggle-house-prices)

No model was selected on the public leaderboard. Public is 20% of the test set
here; picking the best public score across 14 submissions is how a top-10%
public result becomes a 16.9% private one, which happened to this team in
`playground-series-s6e10`.
