"""
Multi-Hazard Risk Assessment Module
====================================
Project: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

Implements Section VII (Eq. 8) of the project paper:
    RiskScore_k = sum_j (w_{k,j} * p_{k,j}), where sum_j (w_{k,j}) = 1.0

Maps continuous forecast values and static district vulnerability parameters
into auditable, decoupled risk scores and discrete risk categories:
    [0.00, 0.25) -> Low
    [0.25, 0.50) -> Moderate
    [0.50, 0.75) -> High
    [0.75, 1.00] -> Severe
"""

import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def score_to_category(score: float) -> str:
    """
    Standard 4-level hazard classification threshold.
    """
    score = np.clip(score, 0.0, 1.0)
    if score < 0.25:
        return "Low"
    elif score < 0.50:
        return "Moderate"
    elif score < 0.75:
        return "High"
    else:
        return "Severe"


def compute_drought_risk(
    predicted_rain_anomaly_mm: float,
    antecedent_90d_rain_mm: float,
    predicted_temp_anomaly_c: float,
    district_metadata: dict,
    normal_90d_rain_mm: float = 300.0
) -> tuple[float, str, dict]:
    """
    Calculate Drought Risk Score for a district based on forecasted deficit,
    antecedent moisture stress, temperature anomaly, and soil vulnerability.
    """
    # 1. Forecasted rainfall deficit: negative anomaly -> high risk
    # If anomaly is -150 mm or worse, p1 -> 1.0; if >= 0 mm, p1 -> 0.0
    p1 = np.clip(-predicted_rain_anomaly_mm / 150.0, 0.0, 1.0)

    # 2. Antecedent 90-day rainfall deficit: below normal -> high risk
    deficit_90d = max(0.0, normal_90d_rain_mm - antecedent_90d_rain_mm)
    p2 = np.clip(deficit_90d / normal_90d_rain_mm, 0.0, 1.0)

    # 3. Temperature anomaly: positive anomaly increases evapotranspiration
    # +2.0C anomaly -> 1.0
    p3 = np.clip(predicted_temp_anomaly_c / 2.0, 0.0, 1.0)

    # 4. District soil vulnerability: high sand, low clay reduces moisture retention
    sand_pct = district_metadata.get("soil_sand_percent", 40.0)
    clay_pct = district_metadata.get("soil_clay_percent", 35.0)
    # Soil vulnerability index: higher sand -> faster moisture depletion
    p4 = np.clip((sand_pct - 25.0) / 35.0, 0.1, 0.9)

    # Weights summing to 1.0 (Auditable domain weights)
    weights = [0.45, 0.25, 0.15, 0.15]
    score = float(weights[0] * p1 + weights[1] * p2 + weights[2] * p3 + weights[3] * p4)
    score = round(float(np.clip(score, 0.0, 1.0)), 4)
    category = score_to_category(score)

    breakdown = {
        "p_rain_deficit": round(float(p1), 3),
        "p_antecedent_90d_deficit": round(float(p2), 3),
        "p_temperature_stress": round(float(p3), 3),
        "p_soil_vulnerability": round(float(p4), 3),
        "weights": weights
    }
    return score, category, breakdown


def compute_flood_risk(
    predicted_rain_anomaly_mm: float,
    antecedent_30d_rain_mm: float,
    district_metadata: dict,
    normal_month_rain_mm: float = 150.0
) -> tuple[float, str, dict]:
    """
    Calculate Flood Risk Score for a district based on forecasted rainfall surplus,
    antecedent 30-day soil saturation proxy, and low elevation / flat slope terrain.
    """
    # 1. Forecasted rainfall surplus: positive anomaly -> high risk
    # +200 mm surplus or more -> 1.0
    p1 = np.clip(predicted_rain_anomaly_mm / 200.0, 0.0, 1.0)

    # 2. Antecedent 30-day saturation proxy: high prior rain -> soil saturated
    p2 = np.clip(antecedent_30d_rain_mm / 300.0, 0.0, 1.0)

    # 3. Terrain & Drainage vulnerability: low elevation and flat slope increase flood ponding
    elev_m = district_metadata.get("elevation_m", 125.0)
    slope_deg = district_metadata.get("slope_degree", 3.5)
    # Low-lying coastal or delta tracts (e.g. elevation < 50m, slope < 2.5 deg)
    p_elev = np.clip((150.0 - elev_m) / 140.0, 0.0, 1.0)
    p_slope = np.clip((5.0 - slope_deg) / 5.0, 0.0, 1.0)
    p3 = 0.6 * p_elev + 0.4 * p_slope

    # Weights summing to 1.0
    weights = [0.50, 0.30, 0.20]
    score = float(weights[0] * p1 + weights[1] * p2 + weights[2] * p3)
    score = round(float(np.clip(score, 0.0, 1.0)), 4)
    category = score_to_category(score)

    breakdown = {
        "p_rain_surplus": round(float(p1), 3),
        "p_soil_saturation": round(float(p2), 3),
        "p_drainage_vulnerability": round(float(p3), 3),
        "weights": weights
    }
    return score, category, breakdown


def assess_district_hazards(
    district_name: str,
    predicted_rain_anomaly: float,
    predicted_temp_anomaly: float,
    antecedent_30d_rain: float,
    antecedent_90d_rain: float,
    metadata_path: str = "data/processed/tn_districts_metadata.json"
) -> dict:
    """
    End-to-end hazard risk assessment for a given district forecast.
    """
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
        d_meta = meta.get(district_name, {})
    except Exception:
        d_meta = {}

    drought_score, drought_cat, drought_bd = compute_drought_risk(
        predicted_rain_anomaly, antecedent_90d_rain, predicted_temp_anomaly, d_meta
    )

    flood_score, flood_cat, flood_bd = compute_flood_risk(
        predicted_rain_anomaly, antecedent_30d_rain, d_meta
    )

    overall_score = max(drought_score, flood_score)
    overall_cat = score_to_category(overall_score)

    return {
        "district": district_name,
        "drought": {
            "score": drought_score,
            "category": drought_cat,
            "breakdown": drought_bd
        },
        "flood": {
            "score": flood_score,
            "category": flood_cat,
            "breakdown": flood_bd
        },
        "overall": {
            "score": overall_score,
            "category": overall_cat,
            "dominant_hazard": "Drought" if drought_score >= flood_score else "Flood"
        }
    }


if __name__ == "__main__":
    # Test case: Severe deficit in Cauvery Delta (Thanjavur)
    res_dry = assess_district_hazards(
        district_name="Thanjavur",
        predicted_rain_anomaly=-95.0,
        predicted_temp_anomaly=1.4,
        antecedent_30d_rain=40.0,
        antecedent_90d_rain=110.0
    )
    print("=== Test Case 1: Drought Scenario (Thanjavur) ===")
    print(json.dumps(res_dry, indent=2))

    # Test case: Severe surplus in Coastal Chennai
    res_wet = assess_district_hazards(
        district_name="Chennai",
        predicted_rain_anomaly=180.0,
        predicted_temp_anomaly=-0.5,
        antecedent_30d_rain=250.0,
        antecedent_90d_rain=480.0
    )
    print("\n=== Test Case 2: Flood Scenario (Chennai) ===")
    print(json.dumps(res_wet, indent=2))
