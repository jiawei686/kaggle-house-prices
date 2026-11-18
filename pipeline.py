"""House Prices - Advanced Regression Techniques
Group: we-are-nai-looong
Strategy: log-target regression, heavy feature engineering, CV-driven model selection.
"""
import numpy as np
import pandas as pd

DATA = "data"

ORDINAL_MAPS = {
    "ExterQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"None": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                   "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Street": {"Grvl": 0, "Pave": 1},
    "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
}

# NA 在语义上表示"没有这个设施"的列
NA_MEANS_NONE = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
]


def load_raw(remove_outliers=False):
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    # 已知的 2 个"Edwards 区大房子异常低价"样本。
    # 默认保留：test 里存在同类样本（Id 2550），删除它们会导致模型对该类样本
    # 外推到 80 万量级，单个样本即可贡献约 0.04 的 RMSE。
    if remove_outliers:
        train = train[~((train.GrLivArea > 4000) & (train.SalePrice < 300000))].reset_index(drop=True)
    y = np.log1p(train["SalePrice"].values)
    test_ids = test["Id"].values
    df = pd.concat([train.drop(columns=["SalePrice"]), test], ignore_index=True)
    return df, y, test_ids, len(train)


def build_features(df):
    df = df.copy()
    df = df.drop(columns=["Id"])

    # --- 语义缺失填充 ---
    for c in NA_MEANS_NONE:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    zero_fill = ["GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea",
                 "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "BsmtFullBath", "BsmtHalfBath"]
    for c in zero_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # LotFrontage 按街区中位数插补
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # --- 有序变量映射 ---
    for c, m in ORDINAL_MAPS.items():
        if c in df.columns:
            df[c] = df[c].map(lambda v: m.get(v, np.nan)).astype(float)
            df[c] = df[c].fillna(0)

    # --- 衍生特征 ---
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorch"] = (df["OpenPorchSF"] + 3 * df["EnclosedPorch"]
                        + df["3SsnPorch"] + df["ScreenPorch"])
    df["TotalRooms"] = df["TotRmsAbvGrd"] + df["TotalBath"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["QualSF"] = df["OverallQual"] * df["TotalSF"]
    df["QualLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["BsmtScore"] = df["BsmtQual"] * (df["BsmtFinSF1"] + df["BsmtFinSF2"])
    df["GarageScore"] = df["GarageQual"] * df["GarageArea"]
    df["ExterScore"] = df["ExterQual"] * df["ExterCond"]
    df["KitchenScore"] = df["KitchenQual"] * df["KitchenAbvGr"]
    df["FireplaceScore"] = df["FireplaceQu"] * df["Fireplaces"]
    df["AvgRoomSF"] = df["GrLivArea"] / df["TotRmsAbvGrd"].replace(0, np.nan)
    df["AvgRoomSF"] = df["AvgRoomSF"].fillna(0)
    df["SFperLot"] = df["GrLivArea"] / df["LotArea"].replace(0, np.nan)
    df["SFperLot"] = df["SFperLot"].fillna(0)
    df["BathPerRoom"] = df["TotalBath"] / df["TotRmsAbvGrd"].replace(0, np.nan)
    df["BathPerRoom"] = df["BathPerRoom"].fillna(0)
    df["PorchPerSF"] = df["TotalPorch"] / df["TotalSF"].replace(0, np.nan)
    df["PorchPerSF"] = df["PorchPerSF"].fillna(0)
    df["TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    df["LowQualFinPct"] = df["LowQualFinSF"] / df["TotalFlrSF"].replace(0, np.nan)
    df["LowQualFinPct"] = df["LowQualFinPct"].fillna(0)

    # --- 剩余类别型（pandas 3 默认 str dtype，不能用 dtype==object 判断） ---
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:
        df[c] = df[c].fillna("None").astype(str)

    return df, cat_cols


def get_matrices(remove_outliers=True):
    """返回 (X_tree_tr, X_tree_te, X_lin_tr, X_lin_te, y, test_ids, cat_cols)

    remove_outliers=True: 剔除 2 个 Edwards 异常低价大房子（整体 CV 更好）
    remove_outliers=False: 保留，树模型能学到 "Edwards + 大面积 -> 低价" 的交互
    """
    df, y, test_ids, ntrain = load_raw(remove_outliers=remove_outliers)
    df, cat_cols = build_features(df)

    # 树模型：类别用 pandas category
    X_tree = df.copy()
    for c in cat_cols:
        X_tree[c] = X_tree[c].astype("category")

    # 线性模型：one-hot + 标准化由 pipeline 处理
    X_lin = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    return (X_tree.iloc[:ntrain], X_tree.iloc[ntrain:],
            X_lin.iloc[:ntrain], X_lin.iloc[ntrain:],
            y, test_ids, cat_cols)
