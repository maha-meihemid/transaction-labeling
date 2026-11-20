import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from src.transaction_labeling.targets import (
    MAJOR_TARGET,
    MINOR_TARGET,
    NULL_MAJOR_PREDICTION,
    map_minor_to_major,
)
from src.transaction_labeling.models import predict_minor


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "split_summary.json"
MODEL_PATH = ROOT / "models"
MODEL_FILE = MODEL_PATH / "transaction_labeler.joblib"
MANIFEST_PATH = MODEL_PATH / "model_manifest.json"
TABLES_PATH = ROOT / "reports" / "tables"
SLICE_PATH = TABLES_PATH / "final_test_slice_metrics.csv"
FIGURES_PATH = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "final_test_report.md"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_predictions(target, model_name, y_true, predictions, labels):
    _, recall, class_f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=labels, zero_division=0
    )
    return {
        "target": target,
        "model": model_name,
        "evaluated_rows": int(len(y_true)),
        "macro_f1": float(np.mean(class_f1)),
        "weighted_f1": float(np.average(class_f1, weights=support)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(np.mean(recall)),
        "null_major_predictions": (
            int(pd.Series(predictions).eq(NULL_MAJOR_PREDICTION).sum())
            if target == MAJOR_TARGET else 0
        ),
    }


def save_model_details(target, model_name, labels, y_true, predictions):
    report = classification_report(
        y_true, predictions, labels=labels, target_names=labels,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        TABLES_PATH / f"{model_name}_{target}_classification_report.csv",
        index_label="class", encoding="utf-8-sig",
    )

    matrix_labels = labels.copy()
    if pd.Series(predictions).eq(NULL_MAJOR_PREDICTION).any():
        matrix_labels.append(NULL_MAJOR_PREDICTION)
    matrix = confusion_matrix(y_true, predictions, labels=matrix_labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=matrix_labels)
    matrix_frame.index.name = "true_label"
    matrix_frame.to_csv(
        TABLES_PATH / f"{model_name}_{target}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Part des transactions de la classe réelle")
    axis.set_xticks(range(len(matrix_labels)), labels=matrix_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Catégorie prédite")
    axis.set_ylabel("Catégorie réelle")
    axis.set_title(f"Matrice de confusion normalisée — {target} ({model_name})")
    figure.tight_layout()
    figure.savefig(FIGURES_PATH / f"{model_name}_{target}_confusion_matrix.png", dpi=160)
    plt.close(figure)


def evaluate_history_slices(train, test, predictions):
    rows = []
    for feature in ["title", "merchant_id"]:
        known_values = set(train[feature].dropna())
        is_known = test[feature].isin(known_values)
        for cohort, mask in [("known_in_train", is_known), ("unseen_in_train", ~is_known)]:
            subset = test.loc[mask]
            if subset.empty:
                continue
            labels = sorted(subset[MINOR_TARGET].unique().tolist())
            metrics = score_predictions(
                MINOR_TARGET,
                "selected_model",
                subset[MINOR_TARGET],
                predictions.loc[subset.index],
                labels,
            )
            rows.append({
                "feature": feature,
                "cohort": cohort,
                "evaluated_rows": metrics["evaluated_rows"],
                "classes_present": len(labels),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            })
    return pd.DataFrame(rows)


def main():
    for path in [TRAIN_PATH, TEST_PATH, SUMMARY_PATH, MODEL_FILE, MANIFEST_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Fichier requis introuvable : {path}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    split_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if file_sha256(TRAIN_PATH) != split_summary["splits"]["train"]["sha256"]:
        raise ValueError("train.csv ne correspond pas à l’empreinte du résumé des splits.")
    if file_sha256(TRAIN_PATH) != manifest["training_data_sha256"]:
        raise ValueError("Le modèle a été entraîné sur une autre version du train.")
    if file_sha256(TEST_PATH) != split_summary["splits"]["test"]["sha256"]:
        raise ValueError("test.csv ne correspond pas à l’empreinte du résumé des splits.")
    if file_sha256(MODEL_FILE) != manifest["model_sha256"]:
        raise ValueError("L’artefact du modèle ne correspond pas au manifeste.")

    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    test_labeled = test.loc[test[MINOR_TARGET].notna()].copy()
    test_major = test.loc[test[MAJOR_TARGET].notna()].copy()
    train_labeled = train.loc[train[MINOR_TARGET].notna()]
    minor_labels = sorted(train_labeled[MINOR_TARGET].unique().tolist())
    major_labels = sorted(train[MAJOR_TARGET].dropna().unique().tolist())

    bundle = joblib.load(MODEL_FILE)
    minor_predictions = predict_minor(bundle, test_labeled)
    major_predictions = map_minor_to_major(
        minor_predictions, bundle["major_category_mapping"]
    )
    slice_results = evaluate_history_slices(train_labeled, test_labeled, minor_predictions)

    majority_label = train_labeled[MINOR_TARGET].value_counts().idxmax()
    majority_minor_predictions = pd.Series(majority_label, index=test_labeled.index)
    majority_major_predictions = map_minor_to_major(
        majority_minor_predictions, bundle["major_category_mapping"]
    )

    model_name = manifest["selected_variant"]
    rows = []
    evaluations = [
        (MINOR_TARGET, "majority_baseline", test_labeled[MINOR_TARGET], majority_minor_predictions, minor_labels),
        (MINOR_TARGET, model_name, test_labeled[MINOR_TARGET], minor_predictions, minor_labels),
        (MAJOR_TARGET, "majority_baseline_mapped", test_major[MAJOR_TARGET], majority_major_predictions.loc[test_major.index], major_labels),
        (MAJOR_TARGET, f"{model_name}_derived", test_major[MAJOR_TARGET], major_predictions.loc[test_major.index].fillna(NULL_MAJOR_PREDICTION), major_labels),
    ]
    for target, name, y_true, predictions, labels in evaluations:
        rows.append(score_predictions(target, name, y_true, predictions, labels))
        save_model_details(target, name, labels, y_true, predictions)

    results = pd.DataFrame(rows)
    results.to_csv(TABLES_PATH / "final_test_metrics.csv", index=False, encoding="utf-8-sig")
    slice_results.to_csv(SLICE_PATH, index=False, encoding="utf-8-sig")
    lines = [
        "# Évaluation finale sur le test",
        "",
        "Le modèle a été sélectionné par macro-F1 de `minor_category` en validation croisée temporelle sur `train`, puis réentraîné sur tout `train`. Le test a été évalué après ce choix et ne sert pas à régler le modèle.",
        "",
        f"Période test UTC : {split_summary['splits']['test']['date_min_utc']} à {split_summary['splits']['test']['date_max_utc']} ({len(test)} lignes).",
        f"Modèle évalué : `{model_name}`. La catégorie principale est dérivée ; les remboursements sans annotation principale sont exclus de son score ({int(test[MAJOR_TARGET].isna().sum())} lignes).",
        f"Empreintes SHA-256 — test : `{file_sha256(TEST_PATH)}` ; modèle : `{manifest['model_sha256']}` ; train : `{manifest['training_data_sha256']}`.",
        "",
        "| Cible | Référence | Lignes évaluées | Macro-F1 | Balanced accuracy | F1 pondéré | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target']}` | `{row['model']}` | {row['evaluated_rows']} | "
            f"{row['macro_f1']:.3f} | {row['balanced_accuracy']:.3f} | "
            f"{row['weighted_f1']:.3f} | {row['accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "Les résultats par classe et les matrices de confusion sont disponibles dans `reports/tables/` et `reports/figures/`. Les métriques de `major_category` portent uniquement sur les lignes annotées. Le jeu étant synthétique et court, le score test reste une mesure indicative de généralisation temporelle.",
        "",
        "## Résultats par historique de libellé et de marchand",
        "",
        "Ces sous-groupes décrivent le même test final ; ils n’ont pas servi à sélectionner le modèle. Le macro-F1 est calculé sur les classes présentes dans chaque sous-groupe.",
        "",
        "| Champ | Cohorte | Lignes | Classes présentes | Macro-F1 | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in slice_results.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | `{row.cohort}` | {row.evaluated_rows} | "
            f"{row.classes_present} | {row.macro_f1:.3f} | {row.accuracy:.3f} |"
        )
    lines.extend([
        "",
        "Les détails sont dans `reports/tables/final_test_slice_metrics.csv`. La cohorte de `merchant_id` inconnu peut encore contenir un `title` connu, puisque le modèle s’appuie sur le texte.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Évaluation finale terminée sur {len(test)} lignes de test.")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
