from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "transaction-dataset.csv"
ERRORS_PATH = ROOT / "reports" / "tables" / "final_test_errors.csv"
AUDIT_TABLE_PATH = ROOT / "reports" / "tables" / "test_error_title_audit.csv"
REPORT_PATH = ROOT / "reports" / "final_test_case_audit.md"


def main():
    raw = pd.read_csv(RAW_PATH)
    errors = pd.read_csv(ERRORS_PATH)
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True)
    errors["created_at"] = pd.to_datetime(errors["created_at"], utc=True)
    errors = errors.assign(error_case_id=range(len(errors)))
    match_columns = ["created_at", "title", "amount_value", "fees"]
    source_rows = errors[
        ["error_case_id", *match_columns, "major_category", "minor_category"]
    ].merge(
        raw[[*match_columns, "major_category", "minor_category"]],
        on=match_columns,
        how="left",
        suffixes=("_error", "_source"),
        indicator=True,
    )
    if source_rows["_merge"].ne("both").any():
        raise ValueError("Certaines erreurs exportées ne sont pas retrouvées dans le CSV brut.")
    source_label_counts = source_rows.groupby("error_case_id").agg(
        minor_labels=("minor_category_source", "nunique"),
        major_labels=("major_category_source", "nunique"),
    )
    if source_label_counts.gt(1).any().any():
        raise ValueError("Les attributs exportés correspondent à plusieurs annotations source.")
    labels_match = pd.Series(True, index=source_rows.index)
    for column in ["major_category", "minor_category"]:
        labels_match &= source_rows[f"{column}_error"].fillna("__null__").eq(
            source_rows[f"{column}_source"].fillna("__null__")
        )
    matched_cases = labels_match.groupby(source_rows["error_case_id"]).all()
    if not matched_cases.all():
        raise ValueError("Certaines catégories exportées ne correspondent pas à la source.")

    error_titles = set(errors["title"].dropna().str.strip().str.upper())
    labeled = raw.loc[raw["minor_category"].notna()].copy()
    labeled["normalized_title"] = labeled["title"].fillna("").str.strip().str.upper()
    title_summary = (
        labeled.loc[labeled["normalized_title"].isin(error_titles)]
        .groupby("normalized_title")
        .agg(
            source_transactions=("transaction_id", "size"),
            distinct_minor_categories=("minor_category", "nunique"),
            category_counts=("minor_category", lambda values: "; ".join(
                f"{label}: {count}" for label, count in values.value_counts().items()
            )),
        )
        .reset_index()
    )
    title_summary = title_summary.loc[title_summary["distinct_minor_categories"].gt(1)]
    title_summary = title_summary.sort_values(
        ["source_transactions", "distinct_minor_categories"], ascending=[False, False]
    )
    title_summary.to_csv(AUDIT_TABLE_PATH, index=False, encoding="utf-8-sig")

    extreme_errors = errors.loc[errors["amount_value"].abs().gt(1_000)].sort_values(
        "amount_value", key=lambda values: values.abs(), ascending=False
    )
    lines = [
        "# Vérification des erreurs dans le CSV source",
        "",
        f"Les {len(errors)} erreurs exportées ont été retrouvées dans le CSV brut à partir de leur date, titre, montant et frais ; les deux catégories annotées ont aussi été vérifiées. Aucun identifiant de transaction n’est exporté.",
        "",
        "## Titres associés à plusieurs catégories fines",
        "",
        "| Titre normalisé | Transactions source | Catégories fines distinctes | Répartition des annotations |",
        "|---|---:|---:|---|",
    ]
    for row in title_summary.head(10).itertuples(index=False):
        lines.append(
            f"| `{row.normalized_title}` | {row.source_transactions} | "
            f"{row.distinct_minor_categories} | {row.category_counts} |"
        )

    lines.extend([
        "",
        "Ces comptages montrent que le titre seul ne détermine pas toujours la catégorie annotée. Ils ne prouvent pas que les annotations sont fausses : le même libellé peut couvrir des achats différents, et le fichier ne fournit pas de justification métier pour chaque annotation.",
        "",
        "## Erreurs avec des montants élevés",
        "",
        "| Titre | Montant | Catégorie annotée | Catégorie prédite | Présence dans le brut |",
        "|---|---:|---|---|---|",
    ])
    for row in extreme_errors.itertuples(index=False):
        lines.append(
            f"| `{row.title}` | {row.amount_value:.2f} | `{row.minor_category}` | "
            f"`{row.predicted_minor_category}` | oui |"
        )

    lines.extend([
        "",
        "Les montants extrêmes sont bien présents dans la source ; il ne s’agit pas d’une altération introduite par la préparation des données. Le fichier ne permet toutefois pas de dire s’ils sont des transactions valides ou des valeurs synthétiques aberrantes. La ligne `Burger King` à -330 978,29 € est annotée `supermarkets`, alors que ce titre est très majoritairement associé à `restaurants` dans le CSV. C’est un cas à signaler comme suspect, sans le supprimer ni réécrire son label sans source métier.",
        "",
        "`APPLE.COM/BILL` est également annoté dans plusieurs catégories fines, principalement `multimedia`, mais aussi `betting` et `others_shopping`. L’erreur sur ce titre peut donc refléter une ambiguïté du signal disponible et/ou une annotation incohérente ; le CSV seul ne permet pas de trancher.",
        "",
        "Aucune ligne ni aucun label n’a été modifié. Les fréquences par titre sont dans `reports/tables/test_error_title_audit.csv`.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Erreurs vérifiées dans la source : {len(errors)}.")
    print(f"Titres conflictuels présents parmi les erreurs : {len(title_summary)}.")
    print(f"Rapport : {REPORT_PATH}")


if __name__ == "__main__":
    main()
