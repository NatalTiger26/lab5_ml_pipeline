"""Leak-free preprocessing, feature engineering, and model pipelines for Lab 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Each hypothesis is stated before the code that builds it.

    Hypotheses (stated first, then tested via CV in the Day-2 notebook):
    1. family_size / is_alone: traveling with family changes survival odds
       (women/children first; large groups harder to coordinate).
    2. title extracted from Name: social status / age / sex proxy beyond raw Sex.
    3. has_cabin: presence of a recorded cabin is a proxy for class/wealth and
       itself predicts survival (see Day-1 missingness check).
    """
    out = df.copy()

    # Hypothesis 1: family size captures traveling-party structure better than
    # SibSp or Parch alone; is_alone flags solo passengers.
    out["family_size"] = out["SibSp"] + out["Parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)

    # Hypothesis 2: title from Name encodes age band and social status
    # (Master/Miss vs Mr/Mrs, rare titles like Dr/Rev/Col).
    if "Name" in out.columns:
        titles = out["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
        titles = titles.str.strip()
        # Collapse rare titles into a single bucket so one-hot stays stable
        common = {"Mr", "Mrs", "Miss", "Master"}
        out["title"] = titles.where(titles.isin(common), other="Rare")
    else:
        out["title"] = "Unknown"

    # Hypothesis 3: the *fact* of having a cabin recorded is informative even
    # when the specific cabin string is mostly missing / high-cardinality.
    if "Cabin" in out.columns:
        out["has_cabin"] = out["Cabin"].notna().astype(int)
    else:
        out["has_cabin"] = 0

    return out


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Numeric: median impute + scale. Categorical: most-frequent impute + one-hot.

    handle_unknown='ignore' is required so unseen categories at predict time
    do not raise.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    model,
) -> Pipeline:
    """Full sklearn Pipeline: preprocessor then model. Fit only on train folds."""
    return Pipeline(
        steps=[
            ("pre", build_preprocessor(num_cols, cat_cols)),
            ("model", model),
        ]
    )


def get_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric vs categorical, with Pclass forced categorical.

    Call this *after* engineer() so family_size / is_alone / has_cabin / title
    land in the right groups.
    """
    # Explicit overrides: Pclass is ordinal but not continuous; treat as category.
    force_cat = {"Pclass", "title", "Sex", "Embarked", "is_alone", "has_cabin"}
    force_num = {"Age", "SibSp", "Parch", "Fare", "family_size"}

    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in X.columns:
        if col in force_cat:
            cat_cols.append(col)
        elif col in force_num:
            num_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols
