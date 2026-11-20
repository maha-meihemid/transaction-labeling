import pandas as pd


IDENTIFIER_COLUMN = "transaction_id"
GROUP_COLUMN = "ref_id"


def remove_exact_ref_duplicates(df):
    """Keep one row per ref_id when all non-transaction_id values match."""
    duplicate_mask = df[GROUP_COLUMN].duplicated(keep=False)
    duplicate_rows = df.loc[duplicate_mask]
    audit_rows = []
    compared_columns = [column for column in df.columns if column != IDENTIFIER_COLUMN]

    for ref_id, group in duplicate_rows.groupby(GROUP_COLUMN, sort=False):
        if len(group) != 2:
            raise ValueError(f"Le ref_id {ref_id} apparaît {len(group)} fois ; vérifier avant déduplication.")
        if group[compared_columns].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Le ref_id {ref_id} contient des lignes différentes ; aucune ligne supprimée.")

        audit_rows.append({
            "ref_id": ref_id,
            "transaction_ids": ",".join(group[IDENTIFIER_COLUMN].astype(str)),
            "rows_before_cleaning": len(group),
            "minor_category": group["minor_category"].iloc[0],
            "major_category": group["major_category"].iloc[0],
            "created_at": group["created_at"].iloc[0],
        })

    audit = pd.DataFrame(audit_rows, columns=[
        "ref_id",
        "transaction_ids",
        "rows_before_cleaning",
        "minor_category",
        "major_category",
        "created_at",
    ])
    unique_columns = [column for column in df.columns if column != IDENTIFIER_COLUMN]
    cleaned = df.drop_duplicates(subset=unique_columns, keep="first").copy()

    if len(df) - len(cleaned) != len(audit):
        raise ValueError("Le nombre de lignes retirées ne correspond pas aux groupes audités.")

    return cleaned, audit
