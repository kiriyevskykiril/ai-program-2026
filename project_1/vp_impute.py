import numpy as np
import pandas as pd

from xgboost import XGBRegressor


def get_best_log_vp_model():
    """
    Best model for prediction of log10(VP).

    """

    model = XGBRegressor(
        objective="reg:absoluteerror",
        colsample_bytree=0.7,
        learning_rate=0.03,
        max_depth=2,
        n_estimators=800,
        subsample=0.7,
        random_state=42
    )

    return model


def train_log_vp_model(X_train, y_train):
    """
    Train the best log10(VP) prediction model.
    """

    model = get_best_log_vp_model()
    model.fit(X_train, y_train)

    return model


def impute_log_vp(
    target_df,
    vp_exp_df,
    dragon_df,
    feature_names,
    X_train,
    y_train,
    cid_col="CID",
    vp_exp_col="vp_exp"
):
    """
    Impute log10(VP) for every molecule.

    Priority:
    1. Use experimental VP if available.
    2. Otherwise predict log10(VP) using Dragon descriptors.

    Parameters
    ----------
    target_df : pd.DataFrame
        Dataset to impute, for example Klio dataset.

    vp_exp_df : pd.DataFrame
        Experimental VP table. Must contain CID and experimental VP column.

    dragon_df : pd.DataFrame
        Dragon descriptors table. Must contain CID and descriptor columns.

    feature_names : list
        Names of Dragon descriptors used by VP model.

    X_train : pd.DataFrame
        Training descriptor matrix for VP prediction.

    y_train : pd.Series or np.array
        Target values: experimental log10(VP).

    Returns
    -------
    pd.DataFrame
        target_df with added columns:
        vp_exp, log_vp_exp, log_vp_predicted, log_vp, vp_mmhg, vp_source
    """

    df = target_df.copy()

    # 1. Train VP prediction model inside this function
    model = train_log_vp_model(X_train, y_train)

    # 2. Add experimental VP by CID
    df = df.merge(
        vp_exp_df[[cid_col, vp_exp_col]],
        on=cid_col,
        how="left"
    )

    # 3. Convert experimental VP to log10(VP)
    df["log_vp_exp"] = np.where(
        df[vp_exp_col].notna() & (df[vp_exp_col] > 0),
        np.log10(df[vp_exp_col]),
        np.nan
    )

    # 4. Add Dragon descriptors
    df = df.merge(
        dragon_df[[cid_col] + list(feature_names)],
        on=cid_col,
        how="left"
    )

    # 5. Prepare rows where experimental VP is missing
    missing_exp_mask = df["log_vp_exp"].isna()

    df["log_vp_predicted"] = np.nan

    X_missing = df.loc[missing_exp_mask, feature_names]

    # 6. Predict only rows where all descriptors exist
    valid_prediction_mask = X_missing.notna().all(axis=1)

    valid_index = X_missing.index[valid_prediction_mask]

    if len(valid_index) > 0:
        df.loc[valid_index, "log_vp_predicted"] = model.predict(
            X_missing.loc[valid_prediction_mask]
        )

    # 7. Final log_vp: experimental first, predicted second
    df["log_vp"] = df["log_vp_exp"].fillna(df["log_vp_predicted"])

    # 8. Source column
    df["vp_source"] = np.where(
        df["log_vp_exp"].notna(),
        "experimental",
        np.where(df["log_vp_predicted"].notna(), "predicted", "missing")
    )

    # 9. Convert log_vp back to VP
    df["vp_mmhg"] = 10 ** df["log_vp"]

    return df