"""Carga, limpeza, split e pipeline de pré-processamento — NYC Airbnb 2019.

Uso a partir da raiz do repositório:

    from docs.projects.eda.code.pipeline import load_split, build_preprocessor
    X_train, X_test, y_train, y_test = load_split()
    pre = build_preprocessor().fit(X_train)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

SEED = 42
TEST_SIZE = 0.2
N_PRICE_BINS = 5            # estratos de log(price) para estratificar o split
MAX_MINIMUM_NIGHTS = 365    # mais de um ano de estadia mínima é inconsistente
MIN_NEIGHBOURHOOD_FREQ = 50  # bairros mais raros viram "infrequent"
CLIP_QUANTILE = 0.99

DATA_PATH = Path(__file__).resolve().parents[4] / "data" / "AB_NYC_2019.csv"

TARGET = "price"
DROP_COLS = ["id", "name", "host_id", "host_name", "last_review"]
CAT_COLS = ["neighbourhood_group", "neighbourhood", "room_type"]
GEO_COLS = ["latitude", "longitude"]
# Contagens com cauda longa: recebem clip no p99 do treino + log1p
SKEWED_COLS = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
]
BOUNDED_COLS = ["availability_365"]
NUM_COLS = GEO_COLS + SKEWED_COLS + BOUNDED_COLS


def load_raw(path=DATA_PATH):
    """Lê o CSV original, sem nenhuma alteração."""
    return pd.read_csv(path)


def clean(df):
    """Remove linhas com alvo impossível (price == 0) e colunas descartadas.

    Não usa estatística nenhuma do conjunto — é seguro antes do split.
    """
    kept = df.loc[df[TARGET] > 0]
    capped = kept.assign(
        minimum_nights=kept["minimum_nights"].clip(upper=MAX_MINIMUM_NIGHTS)
    )
    return capped.drop(columns=DROP_COLS)


def split(df):
    """Split 80/20 estratificado por quintil de log(price)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    strata = pd.qcut(np.log1p(y), q=N_PRICE_BINS, labels=False, duplicates="drop")
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=strata
    )


def load_split(path=DATA_PATH):
    """Atalho: carga + limpeza + split. Retorna X_train, X_test, y_train, y_test."""
    return split(clean(load_raw(path)))


class QuantileClipper:
    """Corta cada coluna no quantil superior aprendido no fit (só no treino)."""

    def __init__(self, q=CLIP_QUANTILE):
        self.q = q

    def fit(self, X, y=None):
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

    def get_params(self, deep=True):
        return {"q": self.q}

    def set_params(self, **params):
        self.q = params.get("q", self.q)
        return self


def build_preprocessor():
    """ColumnTransformer com todas as decisões da Etapa 4A."""
    skewed = Pipeline([
        # reviews_per_month é NaN exatamente quando number_of_reviews == 0
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("clip", QuantileClipper()),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_NEIGHBOURHOOD_FREQ,
        sparse_output=False,
    )
    return ColumnTransformer([
        ("geo", StandardScaler(), GEO_COLS),
        ("skewed", skewed, SKEWED_COLS),
        ("bounded", StandardScaler(), BOUNDED_COLS),
        ("cat", categorical, CAT_COLS),
    ], verbose_feature_names_out=False)


def transform_target(y):
    """Alvo do modelo: log1p(price)."""
    return np.log1p(y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    pre = build_preprocessor().fit(X_train)
    Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)
    print("train:", Xt_train.shape, "test:", Xt_test.shape)
    print("NaN train/test:", np.isnan(Xt_train).sum(), np.isnan(Xt_test).sum())
    print("features:", list(pre.get_feature_names_out()))
