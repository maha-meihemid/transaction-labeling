import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.transaction_labeling.features import build_preprocessor


RANDOM_STATE = 42

MODEL_CANDIDATES = [
    {
        "name": "logistic_word_unweighted_c1",
        "family": "logistic",
        "text_variant": "word",
        "weighting": "unweighted",
        "c": 1.0,
    },
    {
        "name": "logistic_word_balanced_c1",
        "family": "logistic",
        "text_variant": "word",
        "weighting": "balanced",
        "c": 1.0,
    },
    {
        "name": "xgboost_word_unweighted",
        "family": "xgboost",
        "text_variant": "word",
        "weighting": "unweighted",
        "c": None,
    },
    {
        "name": "xgboost_word_balanced",
        "family": "xgboost",
        "text_variant": "word",
        "weighting": "balanced",
        "c": None,
    },
    *[
        {
            "name": f"logistic_word_char_{weighting}_c{c:g}",
            "family": "logistic",
            "text_variant": "word_char",
            "weighting": weighting,
            "c": c,
        }
        for weighting in ["unweighted", "balanced"]
        for c in [1.0, 4.0, 10.0]
    ],
]

MODEL_CANDIDATES_BY_NAME = {
    candidate["name"]: candidate for candidate in MODEL_CANDIDATES
}


def build_logistic_classifier(class_weight=None, c=1.0):
    return LogisticRegression(
        solver="lbfgs",
        C=c,
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


def build_candidate_classifier(candidate):
    if candidate["family"] == "logistic":
        class_weight = "balanced" if candidate["weighting"] == "balanced" else None
        return build_logistic_classifier(class_weight=class_weight, c=candidate["c"])
    if candidate["family"] == "xgboost":
        classifier = build_xgboost_classifier()
        return classifier
    raise ValueError(f"Famille de modèle inconnue : {candidate['family']}.")


def build_model(candidate_name):
    try:
        candidate = MODEL_CANDIDATES_BY_NAME[candidate_name]
    except KeyError as error:
        raise ValueError(f"Configuration inconnue : {candidate_name}.") from error

    return Pipeline([
        ("features", build_preprocessor(candidate["text_variant"])),
        ("classifier", build_candidate_classifier(candidate)),
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
