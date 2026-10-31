"""
Explainability & Model Interpretability Module (SHAP)
=====================================================
Project: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

Implements TreeSHAP (Lundberg et al., 2017) to provide both:
1. Global Interpretability: Which ocean-atmosphere teleconnections and local variables
   dominate monthly climate forecasts across Tamil Nadu.
2. Local District Attribution: Why the model predicted a specific rainfall deficit/surplus
   for a given district at lead time h.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_shap_attributions(
    model_path: str = "models/xgb_rain_model.joblib",
    test_data_path: str = "data/processed/test_features.csv",
    metadata_path: str = "data/processed/split_metadata.json"
) -> tuple[pd.DataFrame, dict]:
    """
    Compute TreeSHAP attributions across test set samples.
    """
    logging.info(f"Loading trained model from {model_path} for SHAP analysis...")
    model = joblib.load(model_path)

    df_test = pd.read_csv(test_data_path)
    with open(metadata_path) as f:
        meta = json.load(f)

    feature_cols = [f"{c}_scaled" for c in meta["feature_cols"]]
    clean_feature_names = meta["feature_cols"]

    X_test = df_test[feature_cols].values

    logging.info("Computing TreeSHAP values on test set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

    # Compute Global Mean Absolute SHAP importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df_global_importance = pd.DataFrame({
        "feature": clean_feature_names,
        "mean_abs_shap_importance": mean_abs_shap
    }).sort_values("mean_abs_shap_importance", ascending=False).reset_index(drop=True)

    os.makedirs("reports", exist_ok=True)
    df_global_importance.to_csv("reports/shap_feature_importance.csv", index=False)
    logging.info("Saved global SHAP feature importance to reports/shap_feature_importance.csv")

    # Store attributions by district-date for instant lookup in the Streamlit app
    explanations_cache = {}
    for idx, row in df_test.iterrows():
        key = f"{row['district']}_{row['date'][:10]}"
        sample_shap = shap_values[idx]

        # Top 5 positive drivers (pushing anomaly upwards)
        pos_indices = np.argsort(sample_shap)[::-1][:5]
        top_positive = [
            {"feature": clean_feature_names[i], "attribution_mm": round(float(sample_shap[i]), 2)}
            for i in pos_indices if sample_shap[i] > 0
        ]

        # Top 5 negative drivers (pushing anomaly downwards / dry signal)
        neg_indices = np.argsort(sample_shap)[:5]
        top_negative = [
            {"feature": clean_feature_names[i], "attribution_mm": round(float(sample_shap[i]), 2)}
            for i in neg_indices if sample_shap[i] < 0
        ]

        explanations_cache[key] = {
            "district": row["district"],
            "date": row["date"][:10],
            "base_value_mm": round(base_value, 2),
            "predicted_anomaly_mm": round(float(np.sum(sample_shap) + base_value), 2),
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative
        }

    with open("reports/district_shap_cache.json", "w") as f:
        json.dump(explanations_cache, f, indent=2)
    logging.info(f"Cached SHAP explanations for {len(explanations_cache)} district-months in reports/district_shap_cache.json")

    return df_global_importance, explanations_cache


if __name__ == "__main__":
    if os.path.exists("models/xgb_rain_model.joblib"):
        df_imp, cache = generate_shap_attributions()
        print("\n" + "=" * 60)
        print("TOP 10 GLOBAL SHAP FEATURES (Rainfall Anomaly Model)")
        print("=" * 60)
        print(df_imp.head(10).to_string(index=False))

        sample_key = list(cache.keys())[0]
        print(f"\nSample Local SHAP Explanation for {sample_key}:")
        print(json.dumps(cache[sample_key], indent=2))
        print("=" * 60)
    else:
        print("Model file models/xgb_rain_model.joblib not found. Run src/train.py first.")
