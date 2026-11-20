import pandas as pd

from src.transaction_labeling.targets import MINOR_TARGET, REFUND_LABEL


def normalize_title(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def most_frequent_label(values):
    counts = values.value_counts()
    most_frequent = counts[counts.eq(counts.max())].index
    return sorted(most_frequent)[0]


def fit_title_history_rule(train):
    labeled = train.loc[train[MINOR_TARGET].notna()].copy()
    non_refunds = labeled.loc[labeled[MINOR_TARGET].ne(REFUND_LABEL)].copy()
    if non_refunds.empty:
        raise ValueError("La baseline exige au moins une transaction hors refund dans le train.")

    non_refunds["_normalized_title"] = non_refunds["title"].map(normalize_title)
    title_modes = non_refunds.groupby("_normalized_title")[MINOR_TARGET].agg(
        most_frequent_label
    ).to_dict()
    fallback = most_frequent_label(non_refunds[MINOR_TARGET])

    positive_amounts = labeled.loc[labeled["amount_value"].gt(0)]
    refunds = labeled.loc[labeled[MINOR_TARGET].eq(REFUND_LABEL)]
    refund_sign_rule = (
        not positive_amounts.empty
        and positive_amounts[MINOR_TARGET].eq(REFUND_LABEL).all()
        and refunds["amount_value"].gt(0).all()
    )

    return {
        "title_modes": title_modes,
        "fallback": fallback,
        "refund_sign_rule": bool(refund_sign_rule),
    }


def predict_title_history_rule(rule, rows):
    predictions = []
    for row in rows[["title", "amount_value"]].itertuples(index=False):
        if rule["refund_sign_rule"] and row.amount_value > 0:
            predictions.append(REFUND_LABEL)
        else:
            title = normalize_title(row.title)
            predictions.append(rule["title_modes"].get(title, rule["fallback"]))
    return pd.Series(predictions, index=rows.index, name=MINOR_TARGET)
