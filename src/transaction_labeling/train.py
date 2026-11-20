import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost

from src.transaction_labeling.features import (
    CATEGORICAL_COLUMNS,
    DATETIME_COLUMN,
    NUMERIC_COLUMNS,
    TEXT_COLUMN,
)
from src.transaction_labeling.models import (
    MODEL_CANDIDATES,
    MODEL_CANDIDATES_BY_NAME,
    build_model,
)
from src.transaction_labeling.targets import MINOR_TARGET, build_major_category_mapping


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "split_summary.json"
COMPARISON_SUMMARY_PATH = ROOT / "reports" / "tables" / "model_comparison.csv"
COMPARISON_PAIRED_PATH = ROOT / "reports" / "tables" / "model_comparison_paired.csv"
COMPARISON_MANIFEST_PATH = ROOT / "reports" / "tables" / "model_comparison_manifest.json"
MODEL_PATH = ROOT / "models"
MODEL_FILE = MODEL_PATH / "transaction_labeler.joblib"
MANIFEST_PATH = MODEL_PATH / "model_manifest.json"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_variant(comparison_summary):
    expected_models = set(MODEL_CANDIDATES_BY_NAME)
    available_models = set(comparison_summary["model"])
    if not expected_models.issubset(available_models):
        missing = sorted(expected_models - available_models)
        raise ValueError(f"Configurations absentes du résumé CV : {missing}.")

    candidates = comparison_summary.loc[
        comparison_summary["model"].isin(expected_models)
    ].copy()
    candidates["selection_order"] = candidates["model"].map({
        candidate["name"]: index
        for index, candidate in enumerate(MODEL_CANDIDATES)
    })
    return candidates.sort_values(
        ["macro_f1_mean", "selection_order"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]["model"]


def main():
    for path in [
        TRAIN_PATH, SUMMARY_PATH, COMPARISON_SUMMARY_PATH,
        COMPARISON_PAIRED_PATH, COMPARISON_MANIFEST_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Fichier requis introuvable : {path}")

    train = pd.read_csv(TRAIN_PATH)
    split_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if file_sha256(TRAIN_PATH) != split_summary["splits"]["train"]["sha256"]:
        raise ValueError("train.csv ne correspond pas à l’empreinte du résumé des splits.")

    comparison_manifest = json.loads(COMPARISON_MANIFEST_PATH.read_text(encoding="utf-8"))
    if comparison_manifest["train_sha256"] != file_sha256(TRAIN_PATH):
        raise ValueError("La comparaison a été calculée sur une autre version de train.csv.")
    if comparison_manifest["split_summary_sha256"] != file_sha256(SUMMARY_PATH):
        raise ValueError("La comparaison ne correspond pas au résumé courant des splits.")
    if comparison_manifest["summary_sha256"] != file_sha256(COMPARISON_SUMMARY_PATH):
        raise ValueError("Le résumé de comparaison ne correspond pas à son manifeste.")
    if comparison_manifest["paired_summary_sha256"] != file_sha256(COMPARISON_PAIRED_PATH):
        raise ValueError("Les écarts appariés ne correspondent pas au manifeste de comparaison.")

    comparison_summary = pd.read_csv(COMPARISON_SUMMARY_PATH)
    selected_variant = select_variant(comparison_summary)
    candidate = MODEL_CANDIDATES_BY_NAME[selected_variant]
    model_family = candidate["family"]
    weighting = candidate["weighting"]
    labeled = train.loc[train[MINOR_TARGET].notna()].copy()
    mapping = build_major_category_mapping(labeled)
    model = build_model(selected_variant)
    classes = sorted(labeled[MINOR_TARGET].unique().tolist())
    label_encoding = "identity"

    if model_family == "xgboost":
        class_to_id = {label: index for index, label in enumerate(classes)}
        y = labeled[MINOR_TARGET].map(class_to_id).to_numpy()
        if weighting == "balanced":
            class_counts = pd.Series(y).value_counts().sort_index().to_numpy()
            sample_weight = len(y) / (len(classes) * class_counts[y])
            model.fit(labeled, y, classifier__sample_weight=sample_weight)
        else:
            model.fit(labeled, y)
        label_encoding = "integer"
    else:
        model.fit(labeled, labeled[MINOR_TARGET])

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "major_category_mapping": mapping,
        "label_encoding": label_encoding,
        "classes": classes,
    }
    joblib.dump(bundle, MODEL_FILE, compress=3)

    manifest = {
        "model_type": f"{selected_variant}_with_derived_major_category",
        "selected_variant": selected_variant,
        "selected_candidate": candidate,
        "selection_metric": "minor_category_macro_f1_mean",
        "selection_source": "temporal_cross_validation_on_train_only",
        "classifier_parameters": ({
            "solver": "lbfgs",
            "C": candidate["c"],
            "max_iter": 2_000,
            "random_state": 42,
            "class_weight": "balanced" if weighting == "balanced" else None,
        } if model_family == "logistic" else {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "n_estimators": 80,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": 4,
            "random_state": 42,
            "reg_lambda": 1.0,
            "class_weighting": weighting,
        }),
        "text_parameters": {
            "variant": candidate["text_variant"],
            "word": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_features": 20_000,
                "sublinear_tf": True,
            },
            "character": ({
                "analyzer": "char_wb",
                "ngram_range": [2, 5],
                "min_df": 2,
                "max_features": 50_000,
                "sublinear_tf": True,
            } if candidate["text_variant"] == "word_char" else None),
        },
        "categorical_parameters": {
            "min_frequency": 10,
            "handle_unknown": "infrequent_if_exist",
        },
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "xgboost_version": xgboost.__version__,
        "training_data_sha256": file_sha256(TRAIN_PATH),
        "model_comparison_summary_sha256": file_sha256(COMPARISON_SUMMARY_PATH),
        "model_comparison_manifest_sha256": file_sha256(COMPARISON_MANIFEST_PATH),
        "model_file": MODEL_FILE.name,
        "model_sha256": file_sha256(MODEL_FILE),
        "training_rows": int(len(labeled)),
        "target": MINOR_TARGET,
        "classes": classes,
        "label_encoding": label_encoding,
        "features": {
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
            "text": TEXT_COLUMN,
            "datetime": DATETIME_COLUMN,
            "text_variant": candidate["text_variant"],
        },
        "major_category_mapping_policy": "learned_from_full_train_labels; refund maps to null",
        "major_category_mapping": mapping,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Variante sélectionnée par macro-F1 CV : {selected_variant}. "
        f"Modèle réentraîné sur les {len(labeled)} lignes de train."
    )
    print(f"Modèle : {MODEL_FILE}")
    print(f"Manifeste : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
