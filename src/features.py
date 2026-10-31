"""
Feature Engineering and Preprocessing Pipeline
=============================================
Project: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

This module transforms the harmonized monthly district panel into feature matrices
with temporal lags, rolling antecedent precipitation, cyclical seasonality encodings,
and anomaly target definitions.

Core Preprocessing Rules (Paper Section V-B & Evaluation Protocol):
1. Strict Chronological Split:
   - Train: Earliest records (2018-07 to 2022-12)
   - Validation: 2023-01 to 2023-12 (Tuning & checkpointing)
   - Test: 2024-01 to 2025-12 (Unseen evaluation)
2. Zero Information Leakage:
   - Climatological normals and Z-score standardization statistics (mu, sigma)
     are computed STRICTLY on the training split.
   - Validation and test sets are scaled using training statistics only.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_features(input_path: str = "data/processed/tn_monthly_features.csv") -> pd.DataFrame:
    """
    Generate lagged, rolling, and cyclical features for each district time series.
    """
    logging.info(f"Reading harmonized panel from {input_path}")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    engineered_dfs = []

    for district, group in df.groupby("district"):
        grp = group.copy().sort_values("date").reset_index(drop=True)

        # 1. ENSO & IOD Lags (t-1 to t-6)
        for lag in range(1, 7):
            grp[f"enso_oni_lag{lag}"] = grp["enso_oni"].shift(lag)
            grp[f"iod_dmi_lag{lag}"] = grp["iod_dmi"].shift(lag)

        # 2. Combined ENSO-IOD interaction state
        grp["enso_iod_lag1"] = grp["enso_oni_lag1"] * grp["iod_dmi_lag1"]

        # 3. Antecedent Precipitation Indicators (30d, 60d, 90d cumulative rainfall)
        grp["rain_lag1"] = grp["rainfall_mm"].shift(1)  # Prior 30-day proxy
        grp["rain_cum_60d"] = grp["rainfall_mm"].shift(1) + grp["rainfall_mm"].shift(2)
        grp["rain_cum_90d"] = (
            grp["rainfall_mm"].shift(1) + grp["rainfall_mm"].shift(2) + grp["rainfall_mm"].shift(3)
        )

        # 4. Antecedent Temperature & Soil Moisture
        grp["temp_lag1"] = grp["temperature_c"].shift(1)
        grp["soil_moisture_lag1"] = grp["soil_moisture"].shift(1)
        grp["ndvi_lag1"] = grp["ndvi"].shift(1)
        grp["humidity_lag1"] = grp["humidity_percent"].shift(1)

        engineered_dfs.append(grp)

    df_feat = pd.concat(engineered_dfs, ignore_index=True)

    # 5. Cyclical Seasonality Encoding
    df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    # 6. Monsoon Regime Flags
    # Tamil Nadu receives major rainfall during Northeast Monsoon (OND)
    df_feat["is_nem"] = df_feat["month"].isin([10, 11, 12]).astype(int)
    # Southwest Monsoon (JJAS)
    df_feat["is_swm"] = df_feat["month"].isin([6, 7, 8, 9]).astype(int)

    # Drop rows with NaN from the 6-month lag warmup period
    initial_len = len(df_feat)
    lag_cols = [f"enso_oni_lag{l}" for l in range(1, 7)] + ["rain_cum_90d"]
    df_feat = df_feat.dropna(subset=lag_cols).reset_index(drop=True)
    logging.info(f"Dropped {initial_len - len(df_feat)} warmup rows; remaining {len(df_feat)} samples.")

    return df_feat


def compute_targets_and_split(df_feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute targets and perform strict chronological split (Train / Val / Test).
    Computes climatology means strictly on the train partition.
    """
    # Define chronological split boundary dates
    train_end = pd.to_datetime("2022-12-31")
    val_end = pd.to_datetime("2023-12-31")

    train_mask = df_feat["date"] <= train_end
    val_mask = (df_feat["date"] > train_end) & (df_feat["date"] <= val_end)
    test_mask = df_feat["date"] > val_end

    df_train = df_feat[train_mask].copy()
    df_val = df_feat[val_mask].copy()
    df_test = df_feat[test_mask].copy()

    logging.info(f"Chronological Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    logging.info(f"Train date range: {df_train['date'].min().date()} to {df_train['date'].max().date()}")
    logging.info(f"Val date range:   {df_val['date'].min().date()} to {df_val['date'].max().date()}")
    logging.info(f"Test date range:  {df_test['date'].min().date()} to {df_test['date'].max().date()}")

    # Compute training-only climatological monthly means per district (Zero leakage)
    clim_rain = df_train.groupby(["district", "month"])["rainfall_mm"].mean().rename("train_clim_rain")
    clim_temp = df_train.groupby(["district", "month"])["temperature_c"].mean().rename("train_clim_temp")

    # Map training climatology to all splits
    for dset in [df_train, df_val, df_test]:
        dset["train_clim_rain"] = dset.set_index(["district", "month"]).index.map(clim_rain)
        dset["train_clim_temp"] = dset.set_index(["district", "month"]).index.map(clim_temp)

        # Fallback to state-level monthly training mean if any district-month pair is missing
        state_rain = df_train.groupby("month")["rainfall_mm"].mean()
        state_temp = df_train.groupby("month")["temperature_c"].mean()
        dset["train_clim_rain"] = dset["train_clim_rain"].fillna(dset["month"].map(state_rain))
        dset["train_clim_temp"] = dset["train_clim_temp"].fillna(dset["month"].map(state_temp))

        # Target 1: Monthly rainfall anomaly (mm)
        dset["target_rain_anomaly"] = dset["rainfall_mm"] - dset["train_clim_rain"]

        # Target 2: Monthly temperature anomaly (C)
        dset["target_temp_anomaly"] = dset["temperature_c"] - dset["train_clim_temp"]

    # Target 3: Risk Level Categorical Encoding
    risk_mapping = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
    for dset in [df_train, df_val, df_test]:
        dset["target_risk_class"] = dset["risk_level"].map(risk_mapping).fillna(0).astype(int)

    metadata = {
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "train_date_range": [str(df_train["date"].min().date()), str(df_train["date"].max().date())],
        "val_date_range": [str(df_val["date"].min().date()), str(df_val["date"].max().date())],
        "test_date_range": [str(df_test["date"].min().date()), str(df_test["date"].max().date())],
        "risk_mapping": risk_mapping
    }

    return df_train, df_val, df_test, metadata


def scale_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardize continuous feature vectors with Z-score normalization.
    Strictly fits on training set only.
    """
    logging.info(f"Fitting StandardScaler on {len(feature_cols)} features using training split only.")
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    train_scaled = df_train.copy()
    val_scaled = df_val.copy()
    test_scaled = df_test.copy()

    scaled_col_names = [f"{col}_scaled" for col in feature_cols]

    train_scaled[scaled_col_names] = scaler.transform(df_train[feature_cols])
    val_scaled[scaled_col_names] = scaler.transform(df_val[feature_cols])
    test_scaled[scaled_col_names] = scaler.transform(df_test[feature_cols])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    logging.info("Saved fitted scaler to models/scaler.joblib")

    return train_scaled, val_scaled, test_scaled, scaler


def run_feature_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end execution of Phase 2 pipeline.
    """
    df_feat = create_features()
    df_train, df_val, df_test, meta = compute_targets_and_split(df_feat)

    # Define predictor feature columns
    feature_cols = [
        # Lagged ENSO & IOD
        "enso_oni_lag1", "enso_oni_lag2", "enso_oni_lag3", "enso_oni_lag4", "enso_oni_lag5", "enso_oni_lag6",
        "iod_dmi_lag1", "iod_dmi_lag2", "iod_dmi_lag3", "iod_dmi_lag4", "iod_dmi_lag5", "iod_dmi_lag6",
        "enso_iod_lag1",
        # Antecedent weather & soil
        "rain_lag1", "rain_cum_60d", "rain_cum_90d",
        "temp_lag1", "soil_moisture_lag1", "ndvi_lag1", "humidity_lag1",
        # Seasonality
        "sin_month", "cos_month", "is_nem", "is_swm",
        # Static district attributes
        "elevation_m", "slope_degree", "drainage_distance_km",
        "soil_ph", "soil_clay_percent", "soil_sand_percent", "soil_nitrogen_mgkg"
    ]

    df_train_scaled, df_val_scaled, df_test_scaled, scaler = scale_features(
        df_train, df_val, df_test, feature_cols
    )

    # Save processed split files
    df_train_scaled.to_csv("data/processed/train_features.csv", index=False)
    df_val_scaled.to_csv("data/processed/val_features.csv", index=False)
    df_test_scaled.to_csv("data/processed/test_features.csv", index=False)

    meta["feature_cols"] = feature_cols
    with open("data/processed/split_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info("Phase 2 Complete. Feature splits saved to data/processed/")
    return df_train_scaled, df_val_scaled, df_test_scaled


if __name__ == "__main__":
    train, val, test = run_feature_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 2 FEATURE ENGINEERING SUMMARY REPORT")
    print("=" * 60)
    print(f"Train samples: {len(train)} (Shape: {train.shape})")
    print(f"Val samples:   {len(val)} (Shape: {val.shape})")
    print(f"Test samples:  {len(test)} (Shape: {test.shape})")
    print("\nTarget Anomaly Statistics (Training Set):")
    print("  Rain Anomaly Mean: {:.2f} mm, Std: {:.2f} mm".format(
        train["target_rain_anomaly"].mean(), train["target_rain_anomaly"].std()
    ))
    print("  Temp Anomaly Mean: {:.2f} C,  Std: {:.2f} C".format(
        train["target_temp_anomaly"].mean(), train["target_temp_anomaly"].std()
    ))
    print("\nTarget Anomaly Statistics (Test Set):")
    print("  Rain Anomaly Mean: {:.2f} mm, Std: {:.2f} mm".format(
        test["target_rain_anomaly"].mean(), test["target_rain_anomaly"].std()
    ))
    print("  Temp Anomaly Mean: {:.2f} C,  Std: {:.2f} C".format(
        test["target_temp_anomaly"].mean(), test["target_temp_anomaly"].std()
    ))
    print("=" * 60)
