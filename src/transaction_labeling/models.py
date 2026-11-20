import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.transaction_labeling.features import build_preprocessor


RANDOM_STATE = 42


def build_logistic_classifier(class_weight=None):
    return LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2_000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )


def build_xgboost_classifier():
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
        reg_lambda=1.0,
    )


def build_model(model_family, class_weight=None):
    if model_family == "logistic":
        classifier = build_logistic_classifier(class_weight=class_weight)
    elif model_family == "xgboost":
        classifier = build_xgboost_classifier()
    else:
        raise ValueError(f"Famille de modèle inconnue : {model_family}.")

    return Pipeline([
        ("features", build_preprocessor()),
        ("classifier", classifier),
    ])


def predict_minor(bundle, rows):
    predictions = bundle["model"].predict(rows)
    if bundle["label_encoding"] == "integer":
        classes = bundle["classes"]
        decoded = []
        for prediction in predictions:
            class_index = int(prediction)
            if class_index < 0 or class_index >= len(classes):
                raise ValueError(f"Indice de classe prédit invalide : {class_index}.")
            decoded.append(classes[class_index])
        predictions = decoded
    elif bundle["label_encoding"] != "identity":
        raise ValueError("Encodage des catégories absent ou inconnu dans l’artefact.")

    return pd.Series(np.asarray(predictions), index=rows.index, name="minor_category")
