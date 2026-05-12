"""
Build a numeric feature matrix from raw Dragon-style descriptors for ML models.

Typical usage::

    from ml_feature_matrix import build_ml_feature_matrix, raw_features_slice

    features_df = raw_features_slice(data_df, start_col="MW")
    X_ml = build_ml_feature_matrix(features_df, feat_dict)
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Same schema as features_set_1.ipynb — single source for group definitions.
FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "f1_mass": {"exact": ["MW"], "prefix": []},
    "f2_lipophilicity": {"exact": ["MLOGP"], "prefix": []},
    "f3_lipophilicity_extended": {
        "exact": ["ALOGP", "ALOGP2", "MLOGP", "MLOGP2"],
        "prefix": [],
    },
    "f4_volume": {"exact": ["VvdwMG"], "prefix": []},
    "f5_volume": {"exact": ["VvdwZAZ"], "prefix": []},
    "f6_volume": {"exact": ["Sv"], "prefix": []},
    "f7_surface_shape_Mor": {
        "exact": ["Morxxu", "Morxxm", "Morxxv"],
        "prefix": ["Mor"],
    },
    "f8_surface_shape_RDF": {"exact": ["RDFxxxx"], "prefix": ["RDF"]},
    "f9_whim": {"exact": ["L1", "L2u", "L3u", "Tu", "Au"], "prefix": []},
    "f10_geometry_topology": {"exact": [], "prefix": ["HATS", "R"]},
    "f11_spatial_autocorrelation": {"exact": [], "prefix": ["GATS"]},
    "f12_polarity": {"exact": ["PDI"], "prefix": []},
    "f13_p_vsa_logp": {
        "exact": [
            "P_VSA_LogP_2",
            "P_VSA_LogP_3",
            "P_VSA_LogP_4",
            "P_VSA_LogP_5",
            "P_VSA_LogP_7",
        ],
        "prefix": [],
    },
    "f14_spdiam": {
        "exact": ["SpDiam_A", "SpDiam_D", "SpDiam_L", "SpDiam_X"],
        "prefix": [],
    },
}


def raw_features_slice(data_df: pd.DataFrame, start_col: str = "MW") -> pd.DataFrame:
    """Columns from ``start_col`` through the end (raw descriptor block)."""
    if start_col not in data_df.columns:
        raise KeyError(f"Column {start_col!r} not in data_df")
    return data_df.loc[:, start_col:].copy()


def group_columns_for_features(
    feature_columns: Iterable[str],
    feature_groups: Mapping[str, dict[str, list[str]]] | None = None,
) -> dict[str, list[str]]:
    """
    Map each group name to descriptor columns present in ``feature_columns``.

    Matching uses the same exact + prefix rules as the exploratory notebook.
    """
    cols = list(feature_columns)
    column_lookup = {c.lower(): c for c in cols}
    groups = feature_groups if feature_groups is not None else FEATURE_GROUPS

    def match_exact(names: list[str]) -> list[str]:
        return [column_lookup[n.lower()] for n in names if n.lower() in column_lookup]

    def match_prefix(prefixes: list[str]) -> list[str]:
        if not prefixes:
            return []
        out: list[str] = []
        for c in cols:
            cl = c.lower()
            if any(cl.startswith(p.lower()) for p in prefixes):
                out.append(c)
        return out

    result: dict[str, list[str]] = {}
    for group_name, cfg in groups.items():
        exact_found = match_exact(cfg["exact"])
        prefix_found = match_prefix(cfg["prefix"])
        result[group_name] = sorted(set(exact_found + prefix_found))
    return result


def build_ml_feature_matrix(
    features_df: pd.DataFrame,
    feat_dict: Mapping[str, int],
    *,
    feature_groups: Mapping[str, dict[str, list[str]]] | None = None,
    scale: bool = True,
    random_state: int | None = 0,
    fillna: str = "median",
) -> pd.DataFrame:
    """
    Reduce each descriptor group and concatenate into one ML-ready matrix.

    Parameters
    ----------
    features_df
        Raw block of descriptors, e.g. ``raw_features_slice(data_df)`` —
        ``data_df.loc[:, \"MW\":]``.
    feat_dict
        Group name -> target number of dimensions (PCA components) after scaling.
    feature_groups
        Optional override of :data:`FEATURE_GROUPS`.
    scale
        If True, apply :class:`~sklearn.preprocessing.StandardScaler` per group
        before PCA.
    random_state
        Passed to PCA where relevant.
    fillna
        ``\"median\"`` fills NaNs per column within each block using train-column
        medians; ``\"none\"`` raises if NaNs are present after coercion to float.

    Returns
    -------
    pandas.DataFrame
        One row per molecule, columns ``{group}_PC1``, … for each group listed
        in ``feat_dict`` (in definition order of ``FEATURE_GROUPS``).
    """
    groups_def = feature_groups if feature_groups is not None else FEATURE_GROUPS
    gc = group_columns_for_features(features_df.columns, groups_def)

    missing_keys = set(feat_dict.keys()) - set(groups_def.keys())
    if missing_keys:
        raise ValueError(f"feat_dict has unknown groups: {sorted(missing_keys)}")

    parts: list[pd.DataFrame] = []

    for group_name in groups_def:
        if group_name not in feat_dict:
            continue

        cols = [c for c in gc.get(group_name, []) if c in features_df.columns]
        if not cols:
            warnings.warn(f"No columns matched for group {group_name!r}; skipping.")
            continue

        k = feat_dict[group_name]
        X = features_df[cols].apply(pd.to_numeric, errors="coerce")

        if fillna == "median":
            X = X.fillna(X.median(numeric_only=True))
        elif fillna == "none":
            if X.isna().any().any():
                raise ValueError(f"NaNs in feature block for {group_name}")
        else:
            raise ValueError("fillna must be 'median' or 'none'")

        arr = X.to_numpy(dtype=float)
        n_samples, n_features = arr.shape
        n_keep = min(int(k), n_features, n_samples)

        if n_keep < 1:
            warnings.warn(
                f"{group_name}: no components (n_samples={n_samples}, "
                f"n_features={n_features}); skipping."
            )
            continue

        if scale:
            arr = StandardScaler().fit_transform(arr)

        pca = PCA(n_components=n_keep, random_state=random_state)
        Z = pca.fit_transform(arr)

        # Guard against tiny numeric noise turning into NaN in edge cases
        if not np.all(np.isfinite(Z)):
            raise ValueError(f"Non-finite values after PCA for group {group_name}")

        col_names = [f"{group_name}_PC{i + 1}" for i in range(Z.shape[1])]
        parts.append(pd.DataFrame(Z, index=features_df.index, columns=col_names))

    if not parts:
        raise ValueError("No feature groups produced columns; check feat_dict and data.")

    out = pd.concat(parts, axis=1)
    return out
