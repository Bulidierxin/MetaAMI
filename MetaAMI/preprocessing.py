from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    """Median-impute and z-score numeric features."""

    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        Xn = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.feature_cols = list(Xn.columns)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        Xi = self.imputer.fit_transform(Xn)
        self.scaler.fit(Xi)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.scaler is None or self.feature_cols is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        Xn = X[self.feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return self.scaler.transform(self.imputer.transform(Xn))

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def load_ami_data(
    data_path: str | Path,
    label_path: str | Path | None = None,
    id_col: str = "hadm_id",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load feature matrix and labels.

    If label_path is provided, it is merged by id_col. Otherwise data_path must contain label_col.
    Returns X_df, y, ids.
    """
    data = pd.read_csv(data_path)
    data.columns = [str(c).strip() for c in data.columns]

    if label_path is not None:
        labels = pd.read_csv(label_path)
        labels.columns = [str(c).strip() for c in labels.columns]
        df = data.merge(labels[[id_col, label_col]], on=id_col, how="inner")
    else:
        df = data.copy()

    missing = [c for c in (id_col, label_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    ids = df[id_col].copy()
    y = df[label_col].astype(int).copy()
    X = df.drop(columns=[id_col, label_col])
    X = X.loc[:, X.columns.notna()]
    return X, y, ids
