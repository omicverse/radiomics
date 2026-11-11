"""Shared contracts: explicit IDs, binary targets, and aligned probabilities."""
import numpy as np
import pandas as pd


def binary_target(y, positive_label=1):
    values = np.asarray(y).reshape(-1)
    if pd.isna(values).any() or len(pd.unique(values)) != 2:
        raise ValueError("Target must contain exactly two non-missing classes.")
    if positive_label not in values:
        raise ValueError("positive_label is not present in the target.")
    return (values == positive_label).astype(int)


def probabilities(y, scores, positive_label=1):
    if isinstance(y, pd.Series) and isinstance(scores, (pd.Series, pd.DataFrame)):
        if not y.index.equals(scores.index):
            raise ValueError("Target and score indices must match in the same order.")
    target = binary_target(y, positive_label)
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != len(target):
        raise ValueError("Scores must be one positive-class probability per target.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return target, values


def unique_ids(data, id_col="ID"):
    if id_col not in data or data[id_col].isna().any() or data[id_col].duplicated().any():
        raise ValueError(f"{id_col} must exist and contain unique non-missing IDs.")


def numeric_frame(X):
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if frame.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")
    return frame.astype(float).replace([np.inf, -np.inf], np.nan)


def join_data(left, right, on="ID"):
    """Strict one-to-one join; never silently discard subjects or duplicate columns."""
    unique_ids(left, on)
    unique_ids(right, on)
    if set(left[on]) != set(right[on]):
        raise ValueError("ID sets differ; explicitly resolve missing subjects before joining.")
    overlap = (set(left) & set(right)) - {on}
    if overlap:
        raise ValueError(f"Overlapping non-ID columns: {sorted(overlap)}")
    return left.merge(right, on=on, how="left", validate="one_to_one", sort=False)
