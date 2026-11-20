import hashlib
import json
from pathlib import Path

import pandas as pd
from src.transaction_labeling.data import remove_exact_ref_duplicates


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "transaction-dataset.csv"
OUTPUT_PATH = ROOT / "data" / "processed"
SUMMARY_PATH = OUTPUT_PATH / "split_summary.json"

EXPECTED_COLUMNS = {
    "transaction_id",
    "ref_id",
    "created_at",
    "member_id",
    "member_name",
    "merchant_id",
    "merchant_name",
    "title",
    "amount_value",
    "amount_currency",
    "fees",
    "card_type",
    "card_name",
    "digitalization_type",
    "merchant_pan_entry_method",
    "pfm_recurring_frequency",
    "status",
    "major_category",
    "minor_category",
}
TRAIN_END = pd.Timestamp("2026-05-28", tz="UTC")
EXPECTED_START_DAY = pd.Timestamp("2026-03-13", tz="UTC")
EXPECTED_END_DAY = pd.Timestamp("2026-06-10", tz="UTC")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    df = pd.read_csv(RAW_PATH, dtype={"transaction_id": "string", "ref_id": "string"})
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    extra_columns = set(df.columns) - EXPECTED_COLUMNS
    if missing_columns or extra_columns:
        raise ValueError(f"Schéma CSV inattendu. Manquantes : {sorted(missing_columns)} ; en trop : {sorted(extra_columns)}")

    required_values = ["transaction_id", "ref_id", "created_at", "minor_category"]
    missing_values = df[required_values].isna().sum()
    if missing_values.any():
        raise ValueError(f"Valeurs manquantes dans les champs requis : {missing_values[missing_values > 0].to_dict()}")
    if df["transaction_id"].duplicated().any():
        raise ValueError("transaction_id doit être unique.")

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    if df["created_at"].isna().any():
        raise ValueError("created_at contient des dates absentes ou non analysables.")

    missing_major = df["major_category"].isna()
    is_refund = df["minor_category"].eq("refund")
    if not missing_major.equals(is_refund):
        raise ValueError("Dans le CSV attendu, major_category est absente exactement pour les lignes refund.")

    return df


def assign_splits(df):
    transaction_day = df["created_at"].dt.floor("D")
    if transaction_day.min() != EXPECTED_START_DAY or transaction_day.max() != EXPECTED_END_DAY:
        raise ValueError("La période du CSV a changé ; vérifier le cadrage avant de modifier les bornes.")

    df = df.copy()
    df["_split"] = "test"
    df.loc[transaction_day <= TRAIN_END, "_split"] = "train"

    split_counts_per_group = df.groupby("ref_id", sort=False)["_split"].nunique()
    if split_counts_per_group.gt(1).any():
        raise ValueError("Un ref_id apparaît dans plusieurs splits.")

    splits = {
        name: df.loc[df["_split"].eq(name)].drop(columns="_split").copy()
        for name in ["train", "test"]
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("Au moins un split est vide ; vérifier les bornes temporelles.")
    return splits


def describe_split(df, total_rows, total_groups, output_file):
    dates = df["created_at"]
    return {
        "file": output_file.name,
        "rows": int(len(df)),
        "row_share": round(len(df) / total_rows, 6),
        "ref_id_groups": int(df["ref_id"].nunique()),
        "group_share": round(df["ref_id"].nunique() / total_groups, 6),
        "date_min_utc": dates.min().isoformat(),
        "date_max_utc": dates.max().isoformat(),
        "minor_category_counts": {
            str(label): int(count)
            for label, count in df["minor_category"].value_counts().sort_index().items()
        },
        "major_category_counts": {
            str(label): int(count)
            for label, count in df["major_category"].value_counts().sort_index().items()
        },
        "major_category_missing": int(df["major_category"].isna().sum()),
    }


def main():
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"CSV source introuvable : {RAW_PATH}")

    raw_df = load_data()
    df, duplicate_audit = remove_exact_ref_duplicates(raw_df)
    if len(duplicate_audit) != 120 or len(df) != 8_001:
        raise ValueError(
            f"Le nettoyage attendu doit retirer 120 paires exactes : "
            f"{len(duplicate_audit)} paires et {len(df)} lignes restantes. Revoir le cadrage."
        )
    splits = assign_splits(df)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    old_validation_file = OUTPUT_PATH / "validation.csv"
    if old_validation_file.is_file():
        old_validation_file.unlink()

    total_rows = len(df)
    total_groups = df["ref_id"].nunique()
    split_summary = {}
    for name, part in splits.items():
        output_file = OUTPUT_PATH / f"{name}.csv"
        part.to_csv(output_file, index=False, encoding="utf-8")
        split_summary[name] = describe_split(part, total_rows, total_groups, output_file)
        split_summary[name]["sha256"] = file_sha256(output_file)

    summary = {
        "source_file": str(RAW_PATH.relative_to(ROOT)),
        "source_sha256": file_sha256(RAW_PATH),
        "split_method": "chronological_by_created_at_utc_day",
        "group_column": "ref_id",
        "train_end_inclusive_utc": TRAIN_END.date().isoformat(),
        "targets": ["major_category", "minor_category"],
        "deduplication": {
            "method": "drop repeated ref_id rows only when every column except transaction_id is identical",
            "rows_before_cleaning": len(raw_df),
            "duplicate_pairs_removed": len(duplicate_audit),
            "rows_after_cleaning": len(df),
            "label_conflicts": 0,
            "applied_before_split": True,
        },
        "major_category_missing_policy": "preserve missing labels; they occur exactly for minor_category=refund in this CSV",
        "rows": total_rows,
        "ref_id_groups": total_groups,
        "splits": split_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"Nettoyage : {len(raw_df)} lignes brutes, {len(duplicate_audit)} doublons retirés, "
        f"{total_rows} lignes restantes avant split."
    )
    print(f"Préparation terminée : {total_rows} lignes, {total_groups} groupes ref_id.")
    for name, details in split_summary.items():
        print(
            f"{name}: {details['rows']} lignes, {details['ref_id_groups']} groupes, "
            f"{details['major_category_missing']} major_category manquantes."
        )
    print(f"Résumé : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
