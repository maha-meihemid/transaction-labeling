import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.transaction_labeling.baselines import normalize_title
from src.transaction_labeling.data import remove_exact_ref_duplicates


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "transaction-dataset.csv"
FIGURES_PATH = ROOT / "reports" / "figures"
TABLES_PATH = ROOT / "reports" / "tables"
TARGET = "minor_category"


def save_table(table, name):
    table.to_csv(TABLES_PATH / name, index=True, encoding="utf-8-sig")


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / name, dpi=160, bbox_inches="tight")
    plt.close()


def build_minor_major_audit(df):
    rows = []
    for minor, group in df.groupby(TARGET, dropna=False):
        major_values = sorted(group["major_category"].dropna().unique().tolist())
        rows.append({
            "minor_category": minor,
            "major_category": major_values[0] if len(major_values) == 1 else pd.NA,
            "major_category_count": len(major_values),
            "rows": len(group),
            "mapping_status": "missing_by_design" if minor == "refund" and not major_values
            else "one_to_one" if len(major_values) == 1
            else "inconsistent",
        })

    audit = pd.DataFrame(rows).set_index("minor_category")
    inconsistent = audit["mapping_status"].eq("inconsistent")
    if inconsistent.any():
        raise ValueError(
            "La relation minor_category -> major_category n’est pas univoque : "
            f"{audit.index[inconsistent].tolist()}"
        )
    if "refund" not in audit.index or audit.loc["refund", "mapping_status"] != "missing_by_design":
        raise ValueError("refund doit rester sans major_category dans le jeu fourni.")
    return audit


def build_title_label_consistency(df, minimum_title_rows=5):
    labeled = df.loc[
        df[TARGET].notna() & df[TARGET].ne("refund"), ["title", TARGET]
    ].copy()
    labeled["normalized_title"] = labeled["title"].map(normalize_title)
    counts = pd.crosstab(labeled["normalized_title"], labeled[TARGET])
    total_rows = counts.sum(axis=1)
    dominant_count = counts.max(axis=1)
    consistency = pd.DataFrame({
        "transactions": total_rows,
        "dominant_category": counts.idxmax(axis=1),
        "discordant_rows": total_rows - dominant_count,
        "dominant_class_share": dominant_count.div(total_rows),
    })
    consistency = consistency.loc[consistency["transactions"].ge(minimum_title_rows)]
    consistency.index.name = "normalized_title"
    consistency = consistency.sort_values(
        ["dominant_class_share", "transactions"], ascending=[True, False]
    )
    summary = pd.DataFrame([{
        "minimum_title_rows": minimum_title_rows,
        "eligible_titles": len(consistency),
        "eligible_transactions": int(consistency["transactions"].sum()),
        "discordant_rows": int(consistency["discordant_rows"].sum()),
        "discordant_row_share": float(
            consistency["discordant_rows"].sum() / consistency["transactions"].sum()
        ),
        "minimum_title_purity": float(consistency["dominant_class_share"].min()),
        "title_at_minimum_purity": consistency.index[0],
    }])
    return consistency, summary


def main():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(DATA_PATH, dtype={"transaction_id": "string", "ref_id": "string"})
    raw_df["created_at"] = pd.to_datetime(raw_df["created_at"], utc=True, errors="coerce")
    df, duplicate_audit = remove_exact_ref_duplicates(raw_df)
    duplicate_audit.to_csv(TABLES_PATH / "repeated_ref_id_audit.csv", index=False, encoding="utf-8-sig")

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean().mul(100),
    }).sort_values("missing_count", ascending=False)
    save_table(missing, "missing_values.csv")

    numeric = df.select_dtypes(include="number").describe().T
    save_table(numeric, "numeric_summary.csv")

    amount_summary = df.groupby(TARGET)["amount_value"].agg(["count", "min", "median", "mean", "max"])
    amount_summary["positive_count"] = df.groupby(TARGET)["amount_value"].apply(lambda values: values.gt(0).sum())
    amount_summary["zero_count"] = df.groupby(TARGET)["amount_value"].apply(lambda values: values.eq(0).sum())
    amount_summary["negative_count"] = df.groupby(TARGET)["amount_value"].apply(lambda values: values.lt(0).sum())
    save_table(amount_summary, "amount_by_minor_category.csv")
    save_table(
        df["amount_value"].quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).to_frame("amount_value"),
        "amount_quantiles.csv",
    )
    save_table(
        df["fees"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_frame("fees"),
        "fees_summary.csv",
    )
    save_table(pd.crosstab(df["status"], df[TARGET]), "status_by_minor_category.csv")
    amount_sign = pd.crosstab(df[TARGET], np.sign(df["amount_value"]))
    amount_sign.columns = ["negative", "zero", "positive"]
    save_table(amount_sign, "amount_sign_by_minor_category.csv")

    categories = df[TARGET].value_counts().rename_axis(TARGET).to_frame("count")
    categories["percent"] = categories["count"].div(len(df)).mul(100)
    save_table(categories, "minor_category_distribution.csv")

    major_minor = pd.crosstab(df["major_category"], df[TARGET], dropna=False)
    save_table(major_minor, "major_minor_crosswalk.csv")
    mapping_audit = build_minor_major_audit(df)
    save_table(mapping_audit, "minor_major_mapping_audit.csv")

    title_consistency, title_consistency_summary = build_title_label_consistency(df)
    title_consistency.to_csv(
        TABLES_PATH / "title_label_consistency.csv", index=True, encoding="utf-8-sig"
    )
    title_consistency_summary.to_csv(
        TABLES_PATH / "title_label_consistency_summary.csv", index=False, encoding="utf-8-sig"
    )

    quality = pd.Series({
        "raw_rows_before_deduplication": len(raw_df),
        "removed_exact_duplicate_rows": len(raw_df) - len(df),
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_transaction_id_rows": int(df["transaction_id"].duplicated().sum()),
        "duplicate_ref_id_rows": int(df["ref_id"].duplicated().sum()),
        "repeated_ref_id_groups": int(df.loc[df["ref_id"].duplicated(False), "ref_id"].nunique()),
        "rows_in_repeated_ref_id_groups": int(df["ref_id"].duplicated(False).sum()),
        "duplicate_feature_target_rows_excluding_ids": int(
            df.drop(columns=["transaction_id", "ref_id"]).duplicated().sum()
        ),
        "rows_in_duplicate_feature_target_groups": int(
            df.drop(columns=["transaction_id", "ref_id"]).duplicated(keep=False).sum()
        ),
        "unique_transaction_id": int(df["transaction_id"].nunique()),
        "unique_ref_id": int(df["ref_id"].nunique()),
        "created_at_parse_failures": int(df["created_at"].isna().sum()),
        "created_at_min_utc": df["created_at"].min(),
        "created_at_max_utc": df["created_at"].max(),
        "minor_category_missing": int(df[TARGET].isna().sum()),
        "major_category_missing": int(df["major_category"].isna().sum()),
        "minor_category_classes": int(df[TARGET].nunique()),
        "major_category_classes": int(df["major_category"].nunique()),
    }, name="value")
    quality.to_csv(TABLES_PATH / "data_quality_summary.csv", header=True, encoding="utf-8-sig")

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    cardinality = pd.DataFrame({
        "unique_values": df[categorical_cols].nunique(dropna=True),
        "unique_ratio": df[categorical_cols].nunique(dropna=True).div(len(df)),
        "missing_percent": df[categorical_cols].isna().mean().mul(100),
    }).sort_values("unique_values", ascending=False)
    save_table(cardinality, "categorical_cardinality.csv")

    label_counts = df[TARGET].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    label_counts.plot.barh(color="#3973ac")
    plt.title("Transactions by fine-grained category")
    plt.xlabel("Number of transactions")
    plt.ylabel("minor_category")
    save_figure("01_minor_category_counts.png")

    missing_plot = missing.loc[missing["missing_count"] > 0].sort_values("missing_percent")
    plt.figure(figsize=(9, 4))
    plt.barh(missing_plot.index, missing_plot["missing_percent"], color="#d98c3f")
    plt.title("Missing values by column")
    plt.xlabel("Missing share (%)")
    plt.ylabel("Column")
    save_figure("02_missing_values.png")

    crosswalk = pd.crosstab(df["major_category"], df[TARGET])
    plt.figure(figsize=(12, 6))
    plt.imshow(crosswalk, aspect="auto", cmap="Blues")
    plt.colorbar(label="Number of transactions")
    plt.xticks(range(len(crosswalk.columns)), crosswalk.columns, rotation=45, ha="right")
    plt.yticks(range(len(crosswalk.index)), crosswalk.index)
    plt.title("Relationship between broad and fine-grained categories")
    plt.xlabel("minor_category")
    plt.ylabel("major_category")
    save_figure("03_major_minor_crosswalk.png")

    amount = df["amount_value"].dropna()
    lower, upper = amount.quantile([0.01, 0.99])
    plt.figure(figsize=(9, 5))
    plt.hist(amount.clip(lower, upper), bins=50, color="#3973ac", alpha=0.85)
    plt.title("Signed amounts (central view, visual limits at P1–P99)")
    plt.xlabel("Amount, clipped for display only")
    plt.ylabel("Number of transactions")
    save_figure("04_amount_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.hist(np.log1p(amount.abs()), bins=50, color="#d98c3f", alpha=0.85)
    plt.title("Absolute amount, log1p scale (all observations)")
    plt.xlabel("log(1 + |amount|)")
    plt.ylabel("Number of transactions")
    save_figure("04b_absolute_amount_log_distribution.png")

    amount_by_label = [df.loc[df[TARGET].eq(label), "amount_value"].dropna() for label in label_counts.index]
    plt.figure(figsize=(12, 6))
    plt.boxplot(amount_by_label, tick_labels=label_counts.index, showfliers=False, vert=False)
    plt.xscale("symlog", linthresh=1)
    plt.title("Amounts by category (extreme values hidden for readability)")
    plt.xlabel("Amount")
    plt.ylabel("minor_category")
    save_figure("05_amount_by_category.png")

    monthly = df.set_index("created_at").resample("W")[TARGET].count()
    plt.figure(figsize=(11, 4))
    plt.plot(monthly.index, monthly.values, marker="o", color="#3973ac")
    plt.title("Transaction volume by week")
    plt.xlabel("Week (UTC)")
    plt.ylabel("Number of transactions")
    plt.grid(alpha=0.25)
    save_figure("06_transactions_over_time.png")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = df["created_at"].dt.day_name().value_counts().reindex(weekday_order, fill_value=0)
    plt.figure(figsize=(9, 4))
    weekday_counts.plot.bar(color="#3973ac")
    plt.title("Transaction volume by weekday")
    plt.xlabel("Day (UTC)")
    plt.ylabel("Number of transactions")
    save_figure("07_transactions_by_weekday.png")

    status_target = pd.crosstab(df["status"], df[TARGET], normalize="index")
    plt.figure(figsize=(11, 4))
    plt.imshow(status_target, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    plt.colorbar(label="Share within status")
    plt.xticks(range(len(status_target.columns)), status_target.columns, rotation=45, ha="right")
    plt.yticks(range(len(status_target.index)), status_target.index)
    plt.title("Share of fine-grained categories by status")
    plt.xlabel("minor_category")
    plt.ylabel("status")
    save_figure("08_status_target_relationship.png")

    merchant_rates = pd.crosstab(df["merchant_id"], df[TARGET], normalize="index")
    merchant_purity = merchant_rates.max(axis=1).rename("dominant_class_share")
    merchant_summary = merchant_purity.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_frame()
    merchant_summary.loc["single_class_merchants", "dominant_class_share"] = merchant_purity.eq(1).sum()
    merchant_summary.loc["multi_class_merchants", "dominant_class_share"] = merchant_purity.lt(1).sum()
    save_table(merchant_summary, "merchant_label_purity_summary.csv")

    entity_rows = []
    for column in ["merchant_id", "merchant_name", "title", "member_id"]:
        grouped = df.groupby(column, dropna=False)[TARGET].agg(["size", "nunique"])
        entity_rows.append({
            "column": column,
            "unique_entities": int(df[column].nunique(dropna=False)),
            "transactions_in_multi_label_entities": int(grouped.loc[grouped["nunique"] > 1, "size"].sum()),
            "multi_label_entity_count": int(grouped["nunique"].gt(1).sum()),
        })
    save_table(pd.DataFrame(entity_rows).set_index("column"), "entity_target_multiplicity.csv")

    print("EDA terminée")
    print(f"Lignes brutes : {len(raw_df)} ; lignes après déduplication : {df.shape[0]}")
    print(f"Paires retirées : {len(duplicate_audit)}")
    print(f"Lignes / colonnes analysées : {df.shape[0]} / {df.shape[1]}")
    print(f"Dates UTC : {df['created_at'].min()} - {df['created_at'].max()}")
    print(f"Figures : {FIGURES_PATH}")
    print(f"Tables : {TABLES_PATH}")
    print("\nQualité :")
    print(quality.to_string())
    print("\nCible fine :")
    print(categories.to_string())
    print("\nValeurs manquantes :")
    print(missing.loc[missing["missing_count"] > 0].to_string())
    print("\nCardinalité catégorielle :")
    print(cardinality.to_string())
    print("\nRelation major/minor :")
    print(crosswalk.to_string())
    print("\nCorrespondance catégorie fine -> principale :")
    print(mapping_audit.to_string())
    print("\nCohérence des catégories par titre (hors refunds, titres vus au moins 5 fois) :")
    print(title_consistency_summary.to_string(index=False))
    print("\nMultiplicité des entités :")
    print(pd.DataFrame(entity_rows).to_string(index=False))


if __name__ == "__main__":
    main()
