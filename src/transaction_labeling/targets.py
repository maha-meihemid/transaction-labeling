import pandas as pd


MINOR_TARGET = "minor_category"
MAJOR_TARGET = "major_category"
REFUND_LABEL = "refund"
NULL_MAJOR_PREDICTION = "__major_null_prediction__"


def build_major_category_mapping(df):
    labeled = df.loc[df[MINOR_TARGET].notna(), [MINOR_TARGET, MAJOR_TARGET]]
    is_refund = labeled[MINOR_TARGET].eq(REFUND_LABEL)

    if labeled.loc[is_refund, MAJOR_TARGET].notna().any():
        raise ValueError("Les lignes refund doivent garder major_category vide.")
    if labeled.loc[~is_refund, MAJOR_TARGET].isna().any():
        raise ValueError("Une catégorie fine autre que refund n’a pas de catégorie principale.")

    mapping_frame = labeled.loc[~is_refund].groupby(MINOR_TARGET)[MAJOR_TARGET].nunique()
    if mapping_frame.ne(1).any():
        inconsistent = mapping_frame[mapping_frame.ne(1)].index.tolist()
        raise ValueError(f"Une catégorie fine pointe vers plusieurs catégories principales : {inconsistent}.")

    mapping = (
        labeled.loc[~is_refund]
        .drop_duplicates(MINOR_TARGET)
        .set_index(MINOR_TARGET)[MAJOR_TARGET]
        .to_dict()
    )
    if is_refund.any():
        mapping[REFUND_LABEL] = None
    return mapping


def map_minor_to_major(minor_predictions, mapping):
    predictions = pd.Series(minor_predictions)
    unknown = set(predictions.dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Catégories fines sans correspondance principale : {sorted(unknown)}.")
    return predictions.map(mapping)
