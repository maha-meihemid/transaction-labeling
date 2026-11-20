import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from src.transaction_labeling.baselines import fit_title_history_rule, predict_title_history_rule
from src.transaction_labeling.features import build_preprocessor
from src.transaction_labeling.models import (
    MODEL_CANDIDATES,
    MODEL_CANDIDATES_BY_NAME,
    build_candidate_classifier,
)
from src.transaction_labeling.targets import MINOR_TARGET


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
SPLIT_SUMMARY_PATH = ROOT / "data" / "processed" / "split_summary.json"
TABLES_PATH = ROOT / "reports" / "tables"
SUMMARY_PATH = TABLES_PATH / "model_comparison.csv"
FOLD_PATH = TABLES_PATH / "model_comparison_folds.csv"
CLASS_PATH = TABLES_PATH / "model_comparison_classes.csv"
PAIRED_PATH = TABLES_PATH / "model_comparison_paired.csv"
MANIFEST_PATH = TABLES_PATH / "model_comparison_manifest.json"
REPORT_PATH = ROOT / "reports" / "model_comparison_report.md"
RANDOM_STATE = 42
INITIAL_TRAIN_DAYS = 28
VALIDATION_WINDOW_DAYS = 7
BASELINE_NAME = "title_history_rule"
REFERENCE_MODEL = "logistic_word_unweighted_c1"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_folds(data):
    dates = pd.to_datetime(data["created_at"], utc=True).dt.floor("D")
    first_validation_day = dates.min() + pd.Timedelta(days=INITIAL_TRAIN_DAYS)
    last_day = dates.max()
    folds = []

    while first_validation_day <= last_day:
        last_validation_day = min(
            first_validation_day + pd.Timedelta(days=VALIDATION_WINDOW_DAYS - 1),
            last_day,
        )
        folds.append((first_validation_day, last_validation_day))
        first_validation_day += pd.Timedelta(days=VALIDATION_WINDOW_DAYS)

    if len(folds) < 2:
        raise ValueError("Pas assez de données pour créer au moins deux fenêtres temporelles.")
    return folds


def main():
    for path in [TRAIN_PATH, SPLIT_SUMMARY_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Fichier requis introuvable : {path}")

    train = pd.read_csv(TRAIN_PATH)
    split_summary = json.loads(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    train_hash = file_sha256(TRAIN_PATH)
    if train_hash != split_summary["splits"]["train"]["sha256"]:
        raise ValueError("train.csv ne correspond pas à l’empreinte du résumé des splits.")

    labeled = train.loc[train[MINOR_TARGET].notna()].copy()
    labels = sorted(labeled[MINOR_TARGET].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    y = labeled[MINOR_TARGET].map(label_to_id).to_numpy()
    dates = pd.to_datetime(labeled["created_at"], utc=True).dt.floor("D")
    fold_windows = make_folds(labeled)
    fold_rows = []
    class_rows = []

    for fold_number, (start_day, end_day) in enumerate(fold_windows, start=1):
        train_mask = (dates < start_day).to_numpy()
        validation_mask = dates.between(start_day, end_day).to_numpy()
        fold_train = labeled.loc[train_mask]
        fold_validation = labeled.loc[validation_mask]
        y_train = y[train_mask]
        y_validation = y[validation_mask]

        missing_train = set(np.unique(y)) - set(np.unique(y_train))
        missing_validation = set(np.unique(y)) - set(np.unique(y_validation))
        if missing_train or missing_validation:
            raise ValueError(
                f"Fold {fold_number}: classes absentes — train={sorted(missing_train)}, "
                f"validation={sorted(missing_validation)}."
            )

        class_counts = np.bincount(y_train, minlength=len(labels))
        sample_weights = len(y_train) / (len(labels) * class_counts[y_train])

        transformed_by_variant = {}
        for text_variant in {candidate["text_variant"] for candidate in MODEL_CANDIDATES}:
            preprocessor = build_preprocessor(text_variant)
            transformed_by_variant[text_variant] = (
                preprocessor.fit_transform(fold_train, y_train),
                preprocessor.transform(fold_validation),
            )

        for candidate in MODEL_CANDIDATES:
            model_name = candidate["name"]
            x_train, x_validation = transformed_by_variant[candidate["text_variant"]]
            model = build_candidate_classifier(candidate)
            use_sample_weights = (
                candidate["family"] == "xgboost"
                and candidate["weighting"] == "balanced"
            )
            fit_parameters = {"sample_weight": sample_weights} if use_sample_weights else {}
            model.fit(x_train, y_train, **fit_parameters)
            predictions = model.predict(x_validation)
            precision, recall, class_f1, support = precision_recall_fscore_support(
                y_validation,
                predictions,
                labels=np.arange(len(labels)),
                zero_division=0,
            )
            fold_rows.append({
                "model": model_name,
                "fold": fold_number,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "validation_start_utc": start_day.date().isoformat(),
                "validation_end_utc": end_day.date().isoformat(),
                "macro_f1": f1_score(
                    y_validation, predictions, labels=np.arange(len(labels)),
                    average="macro", zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(y_validation, predictions),
                "weighted_f1": f1_score(
                    y_validation, predictions, average="weighted", zero_division=0,
                ),
                "accuracy": accuracy_score(y_validation, predictions),
            })
            class_rows.extend({
                "model": model_name,
                "fold": fold_number,
                "class": labels[index],
                "precision": precision[index],
                "recall": recall[index],
                "f1": class_f1[index],
                "support": int(support[index]),
            } for index in range(len(labels)))

        title_rule = fit_title_history_rule(fold_train)
        baseline_predictions = predict_title_history_rule(
            title_rule, fold_validation
        ).map(label_to_id).to_numpy()
        precision, recall, class_f1, support = precision_recall_fscore_support(
            y_validation,
            baseline_predictions,
            labels=np.arange(len(labels)),
            zero_division=0,
        )
        fold_rows.append({
            "model": BASELINE_NAME,
            "fold": fold_number,
            "train_rows": len(fold_train),
            "validation_rows": len(fold_validation),
            "validation_start_utc": start_day.date().isoformat(),
            "validation_end_utc": end_day.date().isoformat(),
            "macro_f1": f1_score(
                y_validation, baseline_predictions, labels=np.arange(len(labels)),
                average="macro", zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(y_validation, baseline_predictions),
            "weighted_f1": f1_score(
                y_validation, baseline_predictions, average="weighted", zero_division=0,
            ),
            "accuracy": accuracy_score(y_validation, baseline_predictions),
        })
        class_rows.extend({
            "model": BASELINE_NAME,
            "fold": fold_number,
            "class": labels[index],
            "precision": precision[index],
            "recall": recall[index],
            "f1": class_f1[index],
            "support": int(support[index]),
        } for index in range(len(labels)))

        print(
            f"Fold {fold_number}/{len(fold_windows)} terminé "
            f"({len(fold_train)} train, {len(fold_validation)} validation).",
            flush=True,
        )

    fold_results = pd.DataFrame(fold_rows)
    class_results = pd.DataFrame(class_rows)
    reference_scores = (
        fold_results.loc[fold_results["model"].eq(REFERENCE_MODEL), ["fold", "macro_f1"]]
        .rename(columns={"macro_f1": "reference_macro_f1"})
    )
    paired_results = fold_results.merge(reference_scores, on="fold", validate="many_to_one")
    paired_results["macro_f1_delta"] = paired_results["macro_f1"] - paired_results["reference_macro_f1"]
    paired_summary = paired_results.groupby("model").agg(
        macro_f1_delta_mean=("macro_f1_delta", "mean"),
        macro_f1_delta_std=("macro_f1_delta", "std"),
        folds_better_than_reference=("macro_f1_delta", lambda values: int(values.gt(0).sum())),
    ).reset_index()

    metric_columns = ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"]
    summary = fold_results.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    selection_order = {
        candidate["name"]: index for index, candidate in enumerate(MODEL_CANDIDATES)
    }
    selection_order[BASELINE_NAME] = len(MODEL_CANDIDATES)
    summary = summary.reset_index()
    summary["selection_order"] = summary["model"].map(selection_order)
    summary = summary.sort_values(
        ["macro_f1_mean", "selection_order"],
        ascending=[False, True],
        kind="stable",
    ).drop(columns="selection_order")

    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    class_results.to_csv(CLASS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    paired_summary.to_csv(PAIRED_PATH, index=False, encoding="utf-8-sig")
    manifest = {
        "train_sha256": train_hash,
        "split_summary_sha256": file_sha256(SPLIT_SUMMARY_PATH),
        "summary_sha256": file_sha256(SUMMARY_PATH),
        "paired_summary_sha256": file_sha256(PAIRED_PATH),
        "initial_train_days": INITIAL_TRAIN_DAYS,
        "validation_window_days": VALIDATION_WINDOW_DAYS,
        "fold_count": len(fold_windows),
        "models": [candidate["name"] for candidate in MODEL_CANDIDATES],
        "baseline": BASELINE_NAME,
        "selection_target": MINOR_TARGET,
        "selection_metric": "macro_f1_mean",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    class_summary = (
        class_results.groupby(["model", "class"])[["precision", "recall", "f1", "support"]]
        .mean()
        .reset_index()
    )
    lines = [
        "# Comparaison des modèles par validation temporelle",
        "",
        "## Protocole",
        "",
        f"Les {len(MODEL_CANDIDATES)} configurations modèle et la baseline sont évaluées sur les mêmes "
        f"{len(fold_windows)} fenêtres temporelles, "
        f"créées uniquement dans `train`. Chaque fenêtre valide sur au plus {VALIDATION_WINDOW_DAYS} jours "
        f"après un historique initial de {INITIAL_TRAIN_DAYS} jours. À chaque fold, le prétraitement est "
        "ajusté uniquement sur les dates d’entraînement. Le choix du modèle repose sur ces folds. Le test "
        "a déjà été consulté dans un cycle exploratoire antérieur et ne constitue plus une estimation indépendante.",
        "",
        f"La cible de sélection est `{MINOR_TARGET}` et le critère principal est le macro-F1 moyen. "
        "La baseline `title_history_rule` utilise uniquement l’historique antérieur à chaque fenêtre : "
        "catégorie non-refund la plus fréquente par titre, avec repli sur la classe globale majoritaire. "
        "Le signal montant positif → refund est activé seulement si cette relation est parfaite dans le train du fold. "
        "Les modèles comparent les deux pondérations ; la logistique est évaluée en TF-IDF mots et mots+caractères "
        "avec C = 1, 4 ou 10. XGBoost garde des paramètres fixes (80 arbres, profondeur 4, taux d’apprentissage 0,05). "
        "La baseline ne participe pas à la sélection du pipeline de modèle.",
        "",
        f"Empreinte SHA-256 de `train.csv` : `{train_hash}`.",
        "",
        "## Résultats par configuration",
        "",
        "| Configuration | Macro-F1 moyen ± écart-type | Balanced accuracy moyenne ± écart-type | F1 pondéré moyen ± écart-type | Accuracy moyenne ± écart-type |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['model']}` | {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} "
            f"| {row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f} "
            f"| {row['weighted_f1_mean']:.3f} ± {row['weighted_f1_std']:.3f} "
            f"| {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f} |"
        )

    lines.extend([
        "",
        f"Les écarts ci-dessous sont calculés fenêtre par fenêtre par rapport à `{REFERENCE_MODEL}`. "
        "Les folds sont temporels et leurs historiques d’entraînement se recouvrent ; ces écarts-types "
        "décrivent la variabilité observée et ne sont pas des intervalles de confiance.",
        "",
        "| Configuration | Écart moyen de macro-F1 | Écart-type des écarts | Fenêtres gagnées sur la référence |",
        "|---|---:|---:|---:|",
    ])
    for row in paired_summary.itertuples(index=False):
        lines.append(
            f"| `{row.model}` | {row.macro_f1_delta_mean:+.3f} | "
            f"{row.macro_f1_delta_std:.3f} | "
            f"{row.folds_better_than_reference}/{len(fold_windows)} |"
        )

    selected = summary.loc[summary["model"].isin(MODEL_CANDIDATES_BY_NAME)].iloc[0]
    lines.extend([
        "",
        f"La configuration modèle au meilleur macro-F1 moyen est `{selected['model']}` "
        f"({selected['macro_f1_mean']:.3f} ± {selected['macro_f1_std']:.3f}). "
        "Cette comparaison sélectionne la configuration selon la validation temporelle. Le score test est "
        "rétrospectif, car le jeu a déjà été exploré.",
        "",
        "## Résultats moyens par catégorie",
        "",
        "| Configuration | Catégorie | Précision | Rappel | F1 | Support moyen |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in class_summary.sort_values(["class", "model"]).iterrows():
        lines.append(
            f"| `{row['model']}` | `{row['class']}` | {row['precision']:.3f} "
            f"| {row['recall']:.3f} | {row['f1']:.3f} | {row['support']:.1f} |"
        )
    lines.extend([
        "",
        "Les résultats détaillés par fenêtre et par catégorie sont dans `reports/tables/`. "
        "Les métriques par fenêtre sont dans `model_comparison_folds.csv`, le détail par catégorie "
        "dans `model_comparison_classes.csv`, les écarts appariés dans `model_comparison_paired.csv` "
        "et le résumé dans `model_comparison.csv`.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
