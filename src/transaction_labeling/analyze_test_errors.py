import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.transaction_labeling.baselines import fit_title_history_rule, normalize_title
from src.transaction_labeling.targets import MINOR_TARGET, map_minor_to_major
from src.transaction_labeling.models import predict_minor


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "split_summary.json"
MODEL_FILE = ROOT / "models" / "transaction_labeler.joblib"
MANIFEST_PATH = ROOT / "models" / "model_manifest.json"
TABLES_PATH = ROOT / "reports" / "tables"
ERRORS_PATH = TABLES_PATH / "final_test_errors.csv"
CONFUSIONS_PATH = TABLES_PATH / "final_test_top_confusions.csv"
REPORT_PATH = ROOT / "reports" / "final_test_error_analysis.md"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if file_sha256(TEST_PATH) != split_summary["splits"]["test"]["sha256"]:
        raise ValueError("test.csv ne correspond pas à l’empreinte du résumé des splits.")
    if file_sha256(TRAIN_PATH) != manifest["training_data_sha256"]:
        raise ValueError("Le modèle a été entraîné sur une autre version du train.")
    if file_sha256(MODEL_FILE) != manifest["model_sha256"]:
        raise ValueError("L’artefact du modèle ne correspond pas au manifeste.")

    test = test.loc[test[MINOR_TARGET].notna()].copy()
    bundle = joblib.load(MODEL_FILE)
    test["predicted_minor_category"] = predict_minor(bundle, test)
    test["predicted_major_category"] = map_minor_to_major(
        test["predicted_minor_category"], bundle["major_category_mapping"]
    )
    test["is_error"] = test[MINOR_TARGET].ne(test["predicted_minor_category"])

    labels = sorted(train[MINOR_TARGET].dropna().unique().tolist())
    report = classification_report(
        test[MINOR_TARGET], test["predicted_minor_category"],
        labels=labels, target_names=labels, output_dict=True, zero_division=0,
    )
    class_metrics = pd.DataFrame(report).T.loc[labels].reset_index(names="category")
    class_metrics["support"] = class_metrics["support"].astype(int)
    class_metrics["errors"] = class_metrics["support"] - class_metrics["recall"].mul(class_metrics["support"]).round().astype(int)
    class_metrics.to_csv(TABLES_PATH / "final_test_class_metrics.csv", index=False, encoding="utf-8-sig")

    matrix = confusion_matrix(test[MINOR_TARGET], test["predicted_minor_category"], labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.index.name = "true_category"
    confusion.columns.name = "predicted_minor_category"
    confusion = confusion.stack().rename("count").reset_index()
    confusion = confusion.loc[
        confusion["true_category"].ne(confusion["predicted_minor_category"])
        & confusion["count"].gt(0)
    ].sort_values(["count", "true_category", "predicted_minor_category"], ascending=[False, True, True])
    confusion.to_csv(CONFUSIONS_PATH, index=False, encoding="utf-8-sig")

    error_columns = [
        "created_at", "title", "amount_value", "fees",
        "major_category", "predicted_major_category", MINOR_TARGET,
        "predicted_minor_category",
    ]
    errors = test.loc[test["is_error"], error_columns].copy()
    title_modes = fit_title_history_rule(train)["title_modes"]
    errors["train_title_mode"] = errors["title"].map(normalize_title).map(title_modes)
    errors["error_context"] = "label_matches_train_title_mode"
    errors.loc[errors["train_title_mode"].isna(), "error_context"] = "no_nonrefund_title_history"
    errors.loc[
        errors["train_title_mode"].notna()
        & errors[MINOR_TARGET].ne(errors["train_title_mode"]),
        "error_context",
    ] = "label_differs_from_train_title_mode"
    errors.to_csv(ERRORS_PATH, index=False, encoding="utf-8-sig")
    error_context = (
        errors["error_context"].value_counts().rename_axis("error_context")
        .rename("transactions").reset_index()
    )
    extreme_errors = errors.loc[errors["amount_value"].abs().gt(1_000)].sort_values(
        "amount_value", key=lambda values: values.abs(), ascending=False
    )

    lines = [
        "# Analyse des erreurs sur le test final",
        "",
        "Cette analyse compare les prédictions aux annotations du CSV. Elle ne permet pas d’établir que ces annotations sont la vérité métier.",
        "",
        f"Le modèle diffère des annotations sur {len(errors)} transactions sur {len(test)} ({len(errors) / len(test):.1%}). La cible analysée est `{MINOR_TARGET}` ; `major_category` est dérivée de la catégorie fine.",
        "",
        "## Contexte des divergences selon l’historique du titre",
        "",
        "Le mode historique est calculé dans `train` en excluant `refund`. Ces groupes décrivent des désaccords avec l’historique ; ils ne prouvent pas un bruit d’annotation ni une erreur métier.",
        "",
        "| Situation | Transactions |",
        "|---|---:|",
    ]
    context_labels = {
        "no_nonrefund_title_history": "Aucun historique non-refund pour ce titre",
        "label_differs_from_train_title_mode": "Annotation différente du mode historique du titre",
        "label_matches_train_title_mode": "Annotation conforme au mode historique du titre",
    }
    for row in error_context.itertuples(index=False):
        lines.append(f"| {context_labels[row.error_context]} | {row.transactions} |")

    lines.extend([
        "",
        "## Catégories avec le plus d’erreurs",
        "",
        "| Catégorie réelle | Transactions test | Erreurs | Rappel | F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    worst_classes = class_metrics.sort_values(["recall", "support"], ascending=[True, False]).head(5)
    for row in worst_classes.itertuples(index=False):
        lines.append(f"| `{row.category}` | {row.support} | {row.errors} | {row.recall:.3f} | {row._3:.3f} |")

    lines.extend(["", "## Confusions les plus fréquentes", "", "| Catégorie réelle | Prédite à tort comme | Nombre | Exemples de titre |", "|---|---|---:|---|"])
    for row in confusion.head(8).itertuples(index=False):
        examples = errors.loc[
            errors[MINOR_TARGET].eq(row.true_category)
            & errors["predicted_minor_category"].eq(row.predicted_minor_category), "title"
        ].dropna().head(3).tolist()
        title_examples = ", ".join(f"`{title}`" for title in examples) or "—"
        lines.append(f"| `{row.true_category}` | `{row.predicted_minor_category}` | {row.count} | {title_examples} |")

    lines.extend([
        "",
        "## Montants atypiques parmi les erreurs",
        "",
        f"{len(extreme_errors)} erreurs sur {len(errors)} concernent un montant absolu supérieur à 1 000 €. Ce seuil sert ici à repérer des cas à relire, pas à définir une règle métier.",
        "",
        "| Titre | Montant | Catégorie réelle | Catégorie prédite |",
        "|---|---:|---|---|",
    ])
    for row in extreme_errors.itertuples(index=False):
        lines.append(f"| `{row.title}` | {row.amount_value:.2f} | `{row.minor_category}` | `{row.predicted_minor_category}` |")

    lines.extend([
        "",
        "## Lecture",
        "",
        "Les principales confusions sont des écarts par rapport aux annotations disponibles. Un même intitulé peut être ambigu selon le contexte, et les exemples ci-dessus ne suffisent pas à établir la cause. Les catégories peu représentées dans le test ont des estimations de rappel plus incertaines. Les erreurs avec des montants extrêmes méritent une vérification des données ; cette analyse ne permet pas de conclure s’il s’agit d’anomalies métier ou d’erreurs de génération.",
        "",
        "Les remboursements sont correctement classés en `minor_category` dans ce test, mais le signe du montant les distingue parfaitement dans ce CSV synthétique ; ce résultat ne démontre donc pas une capacité robuste à reconnaître les remboursements en production.",
        "",
        "La confrontation des exemples au CSV brut et l’audit des annotations associées aux mêmes titres sont documentés dans `reports/final_test_case_audit.md`. Les exemples exportés ne contiennent aucun identifiant de membre. Les fichiers détaillés sont dans `reports/tables/final_test_errors.csv`, `reports/tables/final_test_top_confusions.csv` et `reports/tables/final_test_class_metrics.csv`.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Erreurs du test : {len(errors)} / {len(test)} ({len(errors) / len(test):.1%})")
    print("Confusions les plus fréquentes :")
    print(confusion.head(8).to_string(index=False))
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
