"""
Adaptive Agricultural Decision Support Module
=============================================
Project: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

Implements Section VIII (Eq. 3) of the project paper:
    A*_{d,t+h} = argmax_A u(A, y_hat_{d,t+h}, s_d) subject to A in C_d

Provides an auditable, domain-rule-based agricultural advisory layer that decouples
crop, sowing, and irrigation recommendations from the statistical forecasting models.
"""

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# District Agro-Climatic Zone Mapping for Tamil Nadu
DISTRICT_ZONES = {
    "Thanjavur": "Cauvery Delta Zone (Rice bowl, heavy clay/alluvial)",
    "Cuddalore": "Cauvery Delta & Coastal Lowland",
    "Tiruchirappalli": "Cauvery Delta Fringe / Central Plain",
    "Coimbatore": "Western Agro-Climatic Zone (Semi-arid, red/black soil)",
    "Erode": "Western Agro-Climatic Zone (Bhavani canal command)",
    "Salem": "North-Western Agro-Climatic Zone (Rainfed/tanks)",
    "Madurai": "Southern Agro-Climatic Zone (Vaigai command/semi-arid)",
    "Tirunelveli": "Southern Agro-Climatic Zone (Tamirabarani river basin)",
    "Chennai": "North-Eastern Coastal Zone (Urban/coastal)",
    "Vellore": "Northern Agro-Climatic Zone (Palar river basin)"
}


def get_adaptive_crop_advisory(
    district_name: str,
    drought_category: str,
    flood_category: str,
    predicted_rain_anomaly_mm: float,
    predicted_temp_anomaly_c: float,
    month: int
) -> dict:
    """
    Generate agronomic guidance based on forecasted climate outlook, hazard severity,
    and regional agro-climatic boundaries.
    """
    zone = DISTRICT_ZONES.get(district_name, "Tamil Nadu Agro-Climatic Zone")

    # Default baseline recommendation
    recommended_crops = []
    crops_to_avoid = []
    sowing_strategy = ""
    irrigation_strategy = ""
    advisory_bullet_points = []

    # Scenario A: High or Severe Drought Risk
    if drought_category in ["High", "Severe"]:
        recommended_crops = [
            {"crop": "Barnyard Millet (Kudiraivali)", "duration": "80-90 days", "water_need": "Low (250-300mm)", "resilience": "Very High"},
            {"crop": "Foxtail Millet (Thinai)", "duration": "75-80 days", "water_need": "Low (200-250mm)", "resilience": "Very High"},
            {"crop": "Black Gram (Vamban-8)", "duration": "65-70 days", "water_need": "Very Low (150-200mm)", "resilience": "High"},
            {"crop": "Sorghum (K12)", "duration": "95-100 days", "water_need": "Moderate-Low", "resilience": "High"}
        ]
        crops_to_avoid = ["Samba Paddy (Long-duration CR 1009)", "Sugarcane", "Banana in rainfed tracts"]
        sowing_strategy = (
            "Delay dry sowing until initial monsoon wetting front reaches 15 cm soil depth. "
            "Adopt community dry nurseries or direct-seeded rice (DSR) with seed priming (1% KCl) "
            "to enhance early seedling osmotic tolerance."
        )
        irrigation_strategy = (
            "Implement strict Alternate Wetting and Drying (AWD) with field water tubes (pani pipe). "
            "Prioritize micro-irrigation / drip fertigation for horticultural tracts. "
            "Schedule deficit irrigation exclusively during critical flowering and grain-filling stages."
        )
        advisory_bullet_points = [
            f"Forecasted rainfall deficit is {predicted_rain_anomaly_mm:.1f} mm below long-term normal.",
            "High evaporative demand expected; apply 5 cm coir pith or paddy straw mulch to conserve root-zone moisture.",
            "Avoid full canal inundation; conserve surface storage in farm ponds for critical supplemental irrigations."
        ]

    # Scenario B: High or Severe Flood Risk
    elif flood_category in ["High", "Severe"]:
        recommended_crops = [
            {"crop": "Submergence-Tolerant Paddy (Swarna-Sub1)", "duration": "140-145 days", "water_need": "High", "resilience": "Survives 14 days full submergence"},
            {"crop": "CR 1009 Sub-1", "duration": "155 days", "water_need": "High", "resilience": "Submergence & lodging resistant"},
            {"crop": "Sesbania (Daincha) Green Manure", "duration": "45 days", "water_need": "Tolerates waterlogging", "resilience": "High"}
        ]
        crops_to_avoid = ["Cotton", "Maize", "Groundnut", "Vegetables in low-lying bunds"]
        sowing_strategy = (
            "Advance sowing by 10-14 days so crops establish strong stem elongation prior to peak flood window, "
            "or postpone nursery transplanting until extreme runoff subsides."
        )
        irrigation_strategy = (
            "Immediately cease all artificial irrigation. "
            "Clear field drainage conduits, open surplus weirs in paddy parcels, and desilt collector drains "
            "to prevent prolonged root asphyxiation."
        )
        advisory_bullet_points = [
            f"Forecasted surplus precipitation is +{predicted_rain_anomaly_mm:.1f} mm above normal.",
            "High risk of waterlogging in delta and coastal depressions; inspect bund integrity.",
            "Apply foliar spray of 1% urea + 1% zinc sulphate once floodwaters recede to revitalize root respiration."
        ]

    # Scenario C: Moderate Risk
    elif drought_category == "Moderate":
        recommended_crops = [
            {"crop": "Medium-Duration Paddy (ADT 45 / TKM 13)", "duration": "115-120 days", "water_need": "Moderate", "resilience": "Moderate-High"},
            {"crop": "Maize (CoH 6)", "duration": "100-105 days", "water_need": "Moderate", "resilience": "Moderate"},
            {"crop": "Groundnut (TMV 14)", "duration": "105 days", "water_need": "Moderate", "resilience": "Moderate"}
        ]
        crops_to_avoid = ["Late-planted long duration varieties"]
        sowing_strategy = "Stagger nursery sowings in two tranches separated by 10 days to buffer onset volatility."
        irrigation_strategy = "Adopt cyclic furrow irrigation; maintain 2 cm standing water only during panicle initiation."
        advisory_bullet_points = [
            f"Expected rainfall anomaly is {predicted_rain_anomaly_mm:.1f} mm; monitor weekly advisory bulletins.",
            "Maintain soil organic mulch across vegetable and pulse rows."
        ]

    # Scenario D: Low Risk / Favorable Climate
    else:
        recommended_crops = [
            {"crop": "Standard Paddy (Kuruvai / Samba ADT 37, CR 1009)", "duration": "125-140 days", "water_need": "Standard", "resilience": "Normal"},
            {"crop": "Sugarcane (Co 86032)", "duration": "300-360 days", "water_need": "High", "resilience": "Normal"},
            {"crop": "Cotton / Pulses rotation", "duration": "130-150 days", "water_need": "Standard", "resilience": "Normal"}
        ]
        crops_to_avoid = ["None; conditions favorable for customary regional cropping patterns"]
        sowing_strategy = "Proceed with recommended seasonal calendar according to the Tamil Nadu Agricultural University (TNAU) guide."
        irrigation_strategy = "Standard scheduled canal / borewell irrigation according to crop water requirement."
        advisory_bullet_points = [
            "Near-normal hydro-meteorological outlook across the district.",
            "Favorable conditions for optimal crop yield targets."
        ]

    return {
        "district": district_name,
        "agro_climatic_zone": zone,
        "forecast_outlook": {
            "predicted_rain_anomaly_mm": round(float(predicted_rain_anomaly_mm), 1),
            "predicted_temp_anomaly_c": round(float(predicted_temp_anomaly_c), 2),
            "drought_hazard_level": drought_category,
            "flood_hazard_level": flood_category
        },
        "recommended_crops": recommended_crops,
        "crops_to_avoid": crops_to_avoid,
        "sowing_strategy": sowing_strategy,
        "irrigation_strategy": irrigation_strategy,
        "actionable_guidelines": advisory_bullet_points
    }


if __name__ == "__main__":
    # Test agricultural advisory generation
    test_adv = get_adaptive_crop_advisory(
        district_name="Thanjavur",
        drought_category="High",
        flood_category="Low",
        predicted_rain_anomaly_mm=-85.0,
        predicted_temp_anomaly_c=1.2,
        month=10
    )
    print("=== Adaptive Agriculture Advisory Test Output ===")
    print(json.dumps(test_adv, indent=2))
