"""
src/early_warning.py
====================
Proactive Early Warning & Precautionary Alert System for Tamil Nadu Climate DSS.

Scans ALL future forecast dates for a district, computes risk for each month,
identifies upcoming High/Severe risk events, and returns time-indexed
precautionary recommendations BEFORE the hazard materialises.

No external API calls — fully local inference using precomputed test features.
"""

from datetime import datetime
from typing import Optional
from src.risk import assess_district_hazards


# ── PRECAUTIONARY ACTION PLANS ───────────────────────────────────────────────

FLOOD_PRECAUTIONS = {
    "60+": {   # 60+ days before
        "phase": "PREPARE",
        "color": "#F59E0B",
        "icon": "🟡",
        "label": "Early Preparation Phase",
        "actions": [
            "Inspect and repair all embankments, bunds, and field drainage channels",
            "Desilt and widen main canal outlets before the monsoon season",
            "Procure flood-resilient paddy seeds: Swarna-Sub1, CR 1009 Sub1, Samba Mahsuri Sub1",
            "Install water-level gauge markers on nearby rivers/channels",
            "Register with district NDRF/SDRF emergency contact list",
            "Review and update crop insurance (PMFBY) — enrol before cutoff date",
            "Build or reinforce farm pond bunds to act as temporary flood buffers",
            "Pre-position sandbags and water pumps at vulnerable field edges",
        ],
    },
    "30-60": {  # 30–60 days before
        "phase": "ALERT",
        "color": "#F97316",
        "icon": "🟠",
        "label": "Flood Alert Phase",
        "actions": [
            "Clear ALL drainage channels, sluices, and field outlets immediately",
            "Harvest any standing mature crops NOW — do not wait for full maturity",
            "Shift portable machinery, seeds, and fertilizers to elevated storage",
            "Avoid new transplanting/sowing — wait until risk window passes",
            "Stock 15-day emergency food, drinking water, and medicine supply",
            "Identify and share evacuation routes with family members",
            "Monitor IMD/TN Weather bulletins daily; set weather alert on phone",
            "Contact Gram Panchayat to confirm community shelter locations",
        ],
    },
    "0-30": {   # Less than 30 days before / imminent
        "phase": "WARNING",
        "color": "#EF4444",
        "icon": "🔴",
        "label": "IMMINENT FLOOD WARNING",
        "actions": [
            "ACTIVATE EMERGENCY PROTOCOL — inform all household members NOW",
            "Move livestock and valuable assets to highest ground available",
            "Switch off electrical mains in flood-prone structures",
            "Keep emergency bag ready: ID docs, medicine, cash, phone charger",
            "Do NOT enter flooded fields — risk of electrocution and drowning",
            "Post-flood: apply 2% urea foliar spray within 48hrs of drainage to recover crops",
            "Report crop damage immediately to Block Agriculture Officer for PMFBY claim",
            "Apply Bordeaux mixture to all wounds/cuts on surviving plants to prevent fungal rot",
        ],
    },
}

DROUGHT_PRECAUTIONS = {
    "60+": {
        "phase": "PREPARE",
        "color": "#F59E0B",
        "icon": "🟡",
        "label": "Early Preparation Phase",
        "actions": [
            "Deep plough fields (30 cm) to maximise rain-water infiltration capacity",
            "Construct contour bunds and farm ponds (MNREGA-eligible — apply now)",
            "Procure drought-tolerant seeds: CO 51, TRY 3 (paddy), Kambu, Horsegram",
            "Install drip/sprinkler irrigation — avail 45–90% subsidy from TN Agri Dept",
            "Desilt existing water tanks and check-dams before dry season begins",
            "Apply 25 kg/ha of Potash (MOP) to improve plant water-use efficiency",
            "Enrol in PMFBY crop insurance before the season cutoff date",
            "Plan crop substitution: shift paddy area to millets or pulses",
        ],
    },
    "30-60": {
        "phase": "ALERT",
        "color": "#F97316",
        "icon": "🟠",
        "label": "Drought Alert Phase",
        "actions": [
            "Activate AWD (Alternate Wetting & Drying) immediately for paddy crops",
            "Switch all irrigation to early morning (4–7 AM) to cut evaporation losses",
            "Apply paddy straw or dry sugarcane trash mulch across all inter-rows",
            "Postpone any new transplanting — wait for the next confirmed rainfall window",
            "Prioritize irrigation for critical growth stages only (flowering, grain fill)",
            "Apply anti-transpirant spray (kaolin 5% or NSKE 5%) to reduce crop water loss",
            "Contact nearest Krishi Vigyan Kendra (KVK) for drought advisory package",
            "Reduce livestock headcount if fodder water availability is critically low",
        ],
    },
    "0-30": {
        "phase": "WARNING",
        "color": "#EF4444",
        "icon": "🔴",
        "label": "SEVERE DROUGHT WARNING",
        "actions": [
            "EMERGENCY: Apply for SDRF drought relief compensation at District Collector office",
            "Harvest all crops that have reached 80%+ maturity — do not delay",
            "Stop irrigation of standing crops beyond physiological maturity",
            "Switch to contingency crops: cowpea, black-eyed pea (50-day cycle)",
            "Share/pool remaining irrigation water with neighbouring farmers",
            "Request KVK for emergency seed kit distribution if replanting needed",
            "Check Mettur/Amaravathi/Vaigai reservoir levels — plan 45-day water budget",
            "File preliminary crop loss report with Block Agriculture Officer immediately",
        ],
    },
}

# ── DISTRICT VULNERABILITY WEIGHTS ──────────────────────────────────────────
# Some districts are historically more vulnerable to specific hazards
DISTRICT_FLOOD_WEIGHT = {
    "Chennai": 1.3,      # Coastal + cyclone exposure
    "Cuddalore": 1.4,    # Cauvery delta, very flood-prone
    "Thanjavur": 1.35,   # Cauvery delta, NEM floods
    "Tiruchirappalli": 1.2,
    "Coimbatore": 0.9,
    "Madurai": 0.95,
    "Tirunelveli": 1.1,
    "Erode": 0.9,
    "Salem": 0.85,
    "Vellore": 0.8,
}

DISTRICT_DROUGHT_WEIGHT = {
    "Chennai": 0.8,
    "Cuddalore": 0.85,
    "Thanjavur": 0.9,
    "Tiruchirappalli": 1.0,
    "Coimbatore": 1.25,   # Rain-shadow, semi-arid
    "Madurai": 1.2,
    "Tirunelveli": 1.15,
    "Erode": 1.2,
    "Salem": 1.2,
    "Vellore": 1.3,        # Most drought-prone district
}


def scan_future_risks(
    df_test,
    district_name: str,
    selected_date: str,
    horizon_months: int = 12
):
    """
    Scan all future test dates (after selected_date) for the given district
    and compute risk scores for each month.

    Returns
    -------
    list of dicts, each containing:
        date, months_ahead, flood_score, drought_score,
        flood_cat, drought_cat, overall_cat, rain_mm, rain_anomaly
    """
    from datetime import datetime, timedelta

    try:
        sel_dt = datetime.strptime(selected_date, "%Y-%m-%d")
    except Exception:
        sel_dt = datetime.now()

    district_df = df_test[df_test["district"] == district_name].copy()
    district_df["date_parsed"] = pd.to_datetime(district_df["date"].str[:10])
    future_df = district_df[district_df["date_parsed"] > sel_dt].sort_values("date_parsed")

    f_weight = DISTRICT_FLOOD_WEIGHT.get(district_name, 1.0)
    d_weight = DISTRICT_DROUGHT_WEIGHT.get(district_name, 1.0)

    results = []
    for _, row in future_df.iterrows():
        r_anom = row.get("target_rain_anomaly", 0.0)
        t_anom = row.get("target_temp_anomaly", 0.0)
        prior30 = row.get("rain_lag1", 100.0)
        prior90 = row.get("rain_cum_90d", 300.0)

        risk = assess_district_hazards(district_name, r_anom, t_anom, prior30, prior90)

        flood_score = min(risk["flood"]["score"] * f_weight, 1.0)
        drought_score = min(risk["drought"]["score"] * d_weight, 1.0)

        # Re-categorise with adjusted scores
        def categorize(score):
            if score >= 0.75: return "Severe"
            if score >= 0.50: return "High"
            if score >= 0.25: return "Moderate"
            return "Low"

        flood_cat = categorize(flood_score)
        drought_cat = categorize(drought_score)
        overall_score = max(flood_score, drought_score)
        overall_cat = categorize(overall_score)
        dominant = "Flood" if flood_score >= drought_score else "Drought"

        date_str = row["date_parsed"].strftime("%Y-%m")
        months_ahead = (
            (row["date_parsed"].year - sel_dt.year) * 12
            + (row["date_parsed"].month - sel_dt.month)
        )

        results.append({
            "date": date_str,
            "date_full": row["date_parsed"],
            "months_ahead": months_ahead,
            "flood_score": round(flood_score, 3),
            "drought_score": round(drought_score, 3),
            "overall_score": round(overall_score, 3),
            "flood_cat": flood_cat,
            "drought_cat": drought_cat,
            "overall_cat": overall_cat,
            "dominant_hazard": dominant,
            "rain_mm": round(row.get("rainfall_mm", 0.0), 1),
            "rain_anomaly": round(r_anom, 1),
            "temp_anomaly": round(t_anom, 2),
        })

        if len(results) >= horizon_months:
            break

    return results


def get_precautionary_plan(hazard_type: str, months_ahead: int, district: str) -> dict:
    """
    Return time-indexed precautionary action plan.

    Parameters
    ----------
    hazard_type  : 'Flood' or 'Drought'
    months_ahead : How many months until the high-risk event
    district     : District name (for context-specific notes)

    Returns
    -------
    dict with phase, color, icon, label, actions, district_note
    """
    days_ahead = months_ahead * 30  # approximate

    plans = FLOOD_PRECAUTIONS if hazard_type == "Flood" else DROUGHT_PRECAUTIONS

    if days_ahead >= 60:
        plan = plans["60+"].copy()
    elif days_ahead >= 30:
        plan = plans["30-60"].copy()
    else:
        plan = plans["0-30"].copy()

    # Add district-specific note
    if hazard_type == "Flood":
        if district in ["Thanjavur", "Cuddalore", "Tiruchirappalli"]:
            note = (f"⚡ **{district} is in the Cauvery Delta** — historically very high flood exposure "
                    f"during NEM (Oct–Dec). Coordinate with Cauvery Water Regulation Committee for "
                    f"advance dam release schedules.")
        elif district == "Chennai":
            note = (f"⚡ **{district} Coastal Risk** — cyclone-driven storm surge amplifies flood risk. "
                    f"Monitor IMD cyclone bulletins and activate Coastal District Emergency Plan.")
        else:
            note = (f"⚡ **{district}** — ensure all storm-water drainage systems are inspected "
                    f"and cleared at least 45 days before NEM onset.")
    else:
        if district in ["Coimbatore", "Erode", "Salem", "Vellore"]:
            note = (f"⚠️ **{district} is in a semi-arid zone** — highest drought vulnerability in TN. "
                    f"Activate Jal Jeevan Mission water supply contingency; check Mettur/Amaravathi "
                    f"reservoir levels immediately.")
        else:
            note = (f"⚠️ **{district}** — coordinate with local PWD and Tank Panchayat for "
                    f"water-sharing agreements with upstream users during deficit period.")

    plan["district_note"] = note
    plan["days_ahead"] = days_ahead
    plan["months_ahead"] = months_ahead
    return plan


def get_global_alert_status(future_risks: list) -> dict:
    """
    Summarise the most critical upcoming alert from a district's future risk scan.
    Used for the global top-of-page banner.
    """
    if not future_risks:
        return {"level": "Normal", "color": "#10B981", "icon": "✅", "message": "No significant hazard detected in the forecast horizon."}

    # Find worst upcoming risk
    severe_events = [r for r in future_risks if r["overall_cat"] in ["Severe", "High"]]

    if not severe_events:
        moderate_events = [r for r in future_risks if r["overall_cat"] == "Moderate"]
        if moderate_events:
            e = moderate_events[0]
            return {
                "level": "Watch",
                "color": "#F59E0B",
                "icon": "👀",
                "message": (f"Moderate {e['dominant_hazard']} risk forecast for "
                            f"{e['date']} ({e['months_ahead']} month(s) ahead). "
                            f"Monitor closely and prepare contingency plans."),
                "event": e,
            }
        return {"level": "Normal", "color": "#10B981", "icon": "✅", "message": "Conditions appear near-normal for the next 12 months."}

    # Sort by months_ahead (earliest first), then by severity
    severe_events.sort(key=lambda x: (x["months_ahead"], -x["overall_score"]))
    worst = severe_events[0]

    if worst["overall_cat"] == "Severe":
        return {
            "level": "SEVERE ALERT",
            "color": "#EF4444",
            "icon": "🚨",
            "message": (f"SEVERE {worst['dominant_hazard'].upper()} RISK detected for "
                        f"{worst['date']} — {worst['months_ahead']} month(s) ahead! "
                        f"Risk score: {worst['overall_score']:.2f}. Immediate precautions required."),
            "event": worst,
        }
    else:
        return {
            "level": "HIGH ALERT",
            "color": "#F97316",
            "icon": "⚠️",
            "message": (f"HIGH {worst['dominant_hazard'].upper()} RISK forecast for "
                        f"{worst['date']} — {worst['months_ahead']} month(s) ahead. "
                        f"Risk score: {worst['overall_score']:.2f}. Begin precautionary actions now."),
            "event": worst,
        }


# Make pandas available inside this module
import pandas as pd
