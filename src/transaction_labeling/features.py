import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = ["amount_value", "fees"]
CATEGORICAL_COLUMNS = ["card_name", "digitalization_type", "merchant_pan_entry_method"]
TEXT_COLUMN = "title"
DATETIME_COLUMN = "created_at"


def signed_log(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def positive_log(values):
    values = np.asarray(values, dtype=float)
    return np.log1p(values)


def calendar_features(values):
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    dates = pd.to_datetime(pd.Series(values).reset_index(drop=True), utc=True, errors="coerce")
    hour = dates.dt.hour + dates.dt.minute / 60
    weekday = dates.dt.dayofweek + hour / 24
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
    ])


def build_preprocessor():
    amount_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("signed_log", FunctionTransformer(signed_log, validate=False)),
        ("scale", StandardScaler()),
    ])
    fees_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", FunctionTransformer(positive_log, validate=False)),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("one_hot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=10,
        )),
    ])

    return ColumnTransformer([
        ("amount", amount_pipeline, ["amount_value"]),
        ("fees", fees_pipeline, ["fees"]),
        ("calendar", FunctionTransformer(calendar_features, validate=False), [DATETIME_COLUMN]),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ("title", TfidfVectorizer(
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
        ), TEXT_COLUMN),
    ], remainder="drop")

\n