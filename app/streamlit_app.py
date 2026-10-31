"""
Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)
============================================================================
Interactive Streamlit Dashboard Grounded in Research Report:
"AI-Driven Climate Risk Forecasting, Disaster Risk Assessment, and Adaptive Decision Support System for India"

Features:
1. Multi-Hazard Early Warning (Drought & Flood) for Tamil Nadu Districts
2. ML Forecasts (Random Forest, XGBoost, LSTM) vs Climatology Baselines
3. Ocean-Atmosphere Oscillation Telemetry (ENSO ONI & IOD DMI)
4. Decoupled Adaptive Agriculture Recommendations (Crops, Sowing, Irrigation)
5. Explainable AI (SHAP attributions & drivers)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Ensure project root directory is always on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk import assess_district_hazards
from src.advisory import get_adaptive_crop_advisory
from src.chatbot import get_agriculture_response

# Set Page Config
st.set_page_config(
    page_title="Tamil Nadu Climate Risk & Adaptive Agriculture DSS",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and Base64 encode the user's rain background image
import base64

def get_base64_image(image_path: str) -> str:
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

rain_b64 = get_base64_image("app/rain_bg.png")

# Custom Styling: Live Animated Raining Motion with User Rain Image Background
st.markdown(f"""
<style>
    /* Full Page Rain Image Background with Deep Atmospheric Overlay */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(to bottom, rgba(5, 12, 28, 0.72) 0%, rgba(3, 8, 20, 0.82) 100%),
                    url("data:image/png;base64,{rain_b64}") no-repeat center center fixed !important;
        background-size: cover !important;
        position: relative;
        overflow-x: hidden;
    }}

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {{
        background: rgba(8, 14, 28, 0.88) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.15) !important;
    }}

    /* Animated Falling Rain Layer 1 (Fine Fast Drizzle) */
    .rain-layer-front {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        background-image: 
            repeating-linear-gradient(105deg, transparent 0%, transparent 92%, rgba(224, 242, 254, 0.35) 94%, transparent 96%),
            repeating-linear-gradient(100deg, transparent 0%, transparent 88%, rgba(186, 230, 253, 0.28) 91%, transparent 94%);
        background-size: 240px 360px, 180px 280px;
        animation: rainfallFront 0.65s linear infinite;
        opacity: 0.85;
    }}

    @keyframes rainfallFront {{
        0% {{ background-position: 0px 0px, 0px 0px; }}
        100% {{ background-position: -60px 360px, -45px 280px; }}
    }}

    /* Animated Falling Rain Layer 2 (Heavier Slanted Rain Drops) */
    .rain-layer-back {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        background-image: 
            repeating-linear-gradient(110deg, transparent 0%, transparent 95%, rgba(147, 197, 253, 0.40) 97%, transparent 99%),
            repeating-linear-gradient(103deg, transparent 0%, transparent 90%, rgba(191, 219, 254, 0.22) 93%, transparent 96%);
        background-size: 320px 520px, 140px 220px;
        animation: rainfallBack 0.45s linear infinite;
        opacity: 0.75;
    }}

    @keyframes rainfallBack {{
        0% {{ background-position: 0px 0px, 0px 0px; }}
        100% {{ background-position: -90px 520px, -40px 220px; }}
    }}

    /* Splashing Mist Reflection on Ground */
    .rain-splash-bottom {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw;
        height: 140px;
        pointer-events: none;
        z-index: 0;
        background: radial-gradient(ellipse at bottom, rgba(186, 230, 253, 0.22) 0%, rgba(56, 189, 248, 0.08) 50%, transparent 80%);
        animation: splashGlow 1.8s ease-in-out infinite alternate;
    }}

    @keyframes splashGlow {{
        0% {{ opacity: 0.6; transform: scaleY(0.95); }}
        100% {{ opacity: 1.0; transform: scaleY(1.08); }}
    }}

    /* Main Container & Glassmorphic Content Cards */
    .main .block-container {{
        position: relative;
        z-index: 2;
    }}

    .main-header {{
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #BAE6FD 0%, #38BDF8 40%, #6EE7B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}

    .sub-header {{
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.3rem;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(8, 16, 32, 0.78) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(186, 230, 253, 0.22) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
    }}

    .risk-severe {{
        background: rgba(220, 38, 38, 0.30);
        border: 1px solid #DC2626;
        border-left: 6px solid #DC2626;
        color: #FCA5A5;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }}
    .risk-high {{
        background: rgba(234, 88, 12, 0.30);
        border: 1px solid #EA580C;
        border-left: 6px solid #EA580C;
        color: #FDBA74;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }}
    .risk-moderate {{
        background: rgba(217, 119, 6, 0.30);
        border: 1px solid #D97706;
        border-left: 6px solid #D97706;
        color: #FCD34D;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }}
    .risk-low {{
        background: rgba(22, 163, 74, 0.30);
        border: 1px solid #16A34A;
        border-left: 6px solid #16A34A;
        color: #86EFAC;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }}
</style>

<!-- Live Animated Raining Layers -->
<div class="rain-layer-front"></div>
<div class="rain-layer-back"></div>
<div class="rain-splash-bottom"></div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df_panel = pd.read_csv("data/processed/tn_monthly_features.csv")
    df_test = pd.read_csv("data/processed/test_features.csv")
    df_clim = pd.read_csv("data/processed/tn_imd_climatology.csv")
    
    with open("data/processed/tn_districts_metadata.json") as f:
        districts_meta = json.load(f)
        
    with open("reports/district_shap_cache.json") as f:
        shap_cache = json.load(f)

    df_metrics = pd.read_csv("reports/benchmark_metrics.csv")
    with open("reports/classification_metrics.json") as f:
        clf_metrics = json.load(f)

    return df_panel, df_test, df_clim, districts_meta, shap_cache, df_metrics, clf_metrics


# Load datasets & artifacts
try:
    df_panel, df_test, df_clim, districts_meta, shap_cache, df_metrics, clf_metrics = load_data()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}. Please ensure Phases 1-5 have executed successfully.")
    st.stop()

# Load models
xgb_rain = joblib.load("models/xgb_rain_model.joblib")
rf_rain = joblib.load("models/rf_rain_model.joblib")
xgb_clf = joblib.load("models/xgb_risk_classifier.joblib")

# Sidebar Configuration
st.sidebar.markdown("### 🏛️ TN Climate & Agri DSS")
st.sidebar.title("System Controls")

# Role Selection
user_role = st.sidebar.radio(
    "Select Stakeholder Perspective:",
    ["Disaster Management Official", "Agricultural Extension & Farmer", "Research & Model Diagnostics"]
)

st.sidebar.markdown("---")
districts_list = sorted(list(districts_meta.keys()))
selected_district = st.sidebar.selectbox("Select Target District:", districts_list, index=6)  # Default: Thanjavur

available_dates = sorted(df_test["date"].str[:10].unique().tolist())
selected_date = st.sidebar.select_slider(
    "Select Forecast Horizon / Date:",
    options=available_dates,
    value=available_dates[-6]
)

# Filter test sample for selected district and date
sample_row = df_test[(df_test["district"] == selected_district) & (df_test["date"].str.startswith(selected_date))]
if sample_row.empty:
    sample_row = df_test[df_test["district"] == selected_district].iloc[-1:]

sample_dict = sample_row.iloc[0].to_dict()

# Extract key telemetry
curr_enso = sample_dict.get("enso_oni", 0.0)
curr_iod = sample_dict.get("iod_dmi", 0.0)
obs_rain = sample_dict.get("rainfall_mm", 0.0)
clim_rain = sample_dict.get("train_clim_rain", 100.0)
pred_rain_anomaly = sample_dict.get("target_rain_anomaly", 0.0)
pred_temp_anomaly = sample_dict.get("target_temp_anomaly", 0.0)

risk_assessment = assess_district_hazards(
    district_name=selected_district,
    predicted_rain_anomaly=pred_rain_anomaly,
    predicted_temp_anomaly=pred_temp_anomaly,
    antecedent_30d_rain=sample_dict.get("rain_lag1", 100.0),
    antecedent_90d_rain=sample_dict.get("rain_cum_90d", 300.0)
)

month_num = int(sample_dict.get("month", 10))
advisory = get_adaptive_crop_advisory(
    district_name=selected_district,
    drought_category=risk_assessment["drought"]["category"],
    flood_category=risk_assessment["flood"]["category"],
    predicted_rain_anomaly_mm=pred_rain_anomaly,
    predicted_temp_anomaly_c=pred_temp_anomaly,
    month=month_num
)

# Header
st.markdown("<div class='main-header'>AI-Driven Climate Risk Forecasting & Decision Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Operational Prototype for Tamil Nadu Districts — Predict → Assess → Prepare → Adapt Architecture</div>", unsafe_allow_html=True)

# Telemetry Banner
tcol1, tcol2, tcol3, tcol4 = st.columns(4)
with tcol1:
    st.metric("ENSO Phase (Niño 3.4)", f"{curr_enso:+.2f} °C", "El Niño" if curr_enso >= 0.5 else ("La Niña" if curr_enso <= -0.5 else "Neutral"))
with tcol2:
    st.metric("Indian Ocean Dipole (DMI)", f"{curr_iod:+.3f}", "Positive IOD" if curr_iod >= 0.2 else ("Negative IOD" if curr_iod <= -0.2 else "Neutral"))
with tcol3:
    st.metric("Rainfall Forecast", f"{obs_rain:.1f} mm", f"{pred_rain_anomaly:+.1f} mm vs normal")
with tcol4:
    dominant_cat = risk_assessment["overall"]["category"]
    st.metric("Overall Hazard Status", dominant_cat, risk_assessment["overall"]["dominant_hazard"])

st.markdown("---")

# =========================================================================
# PERSPECTIVE 1: DISASTER MANAGEMENT OFFICIAL
# =========================================================================
if user_role == "Disaster Management Official":
    st.subheader(f"Disaster Preparedness Overview: {selected_district} ({selected_date})")

    col_map, col_hazards = st.columns([3, 2])

    with col_map:
        st.markdown("#### Tamil Nadu District Hazard Overview")
        # Build map dataframe across all districts for the selected date
        map_records = []
        for dname, dinfo in districts_meta.items():
            d_sample = df_test[(df_test["district"] == dname) & (df_test["date"].str.startswith(selected_date))]
            if not d_sample.empty:
                r_anom = d_sample.iloc[0]["target_rain_anomaly"]
                t_anom = d_sample.iloc[0]["target_temp_anomaly"]
                prior30 = d_sample.iloc[0]["rain_lag1"]
                prior90 = d_sample.iloc[0]["rain_cum_90d"]
            else:
                r_anom, t_anom, prior30, prior90 = 0.0, 0.0, 100.0, 300.0

            d_risk = assess_district_hazards(dname, r_anom, t_anom, prior30, prior90)
            map_records.append({
                "district": dname,
                "lat": dinfo["latitude"],
                "lon": dinfo["longitude"],
                "drought_score": d_risk["drought"]["score"],
                "flood_score": d_risk["flood"]["score"],
                "overall_score": d_risk["overall"]["score"],
                "category": d_risk["overall"]["category"],
                "dominant_hazard": d_risk["overall"]["dominant_hazard"]
            })
        df_map = pd.DataFrame(map_records)

        color_map = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#F97316", "Severe": "#EF4444"}
        scatter_fn = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
        map_style_arg = {"map_style": "carto-darkmatter"} if hasattr(px, "scatter_map") else {"mapbox_style": "carto-darkmatter"}

        fig_map = scatter_fn(
            df_map,
            lat="lat",
            lon="lon",
            color="category",
            color_discrete_map=color_map,
            size="overall_score",
            size_max=22,
            hover_name="district",
            hover_data={"overall_score": ":.3f", "dominant_hazard": True, "lat": False, "lon": False},
            zoom=6.3,
            center={"lat": 10.95, "lon": 78.45},
            title=f"Multi-Hazard Risk Map ({selected_date})",
            **map_style_arg
        )
        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            margin={"r":0,"t":40,"l":0,"b":0},
            height=450
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_hazards:
        st.markdown(f"#### Multi-Hazard Risk Scoring for **{selected_district}**")
        d_cat = risk_assessment["drought"]["category"]
        f_cat = risk_assessment["flood"]["category"]

        # Gauge Charts
        fig_gauge = go.Figure()
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_assessment["drought"]["score"],
            domain={'row': 0, 'column': 0},
            title={'text': f"Drought Risk: {d_cat}"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#EF4444" if d_cat in ["High", "Severe"] else "#F59E0B"},
                'steps': [
                    {'range': [0, 0.25], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [0.25, 0.5], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [0.5, 0.75], 'color': "rgba(249, 115, 22, 0.2)"},
                    {'range': [0.75, 1.0], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_assessment["flood"]["score"],
            domain={'row': 0, 'column': 1},
            title={'text': f"Flood Risk: {f_cat}"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#3B82F6" if f_cat in ["High", "Severe"] else "#60A5FA"},
                'steps': [
                    {'range': [0, 0.25], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [0.25, 0.5], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [0.5, 0.75], 'color': "rgba(249, 115, 22, 0.2)"},
                    {'range': [0.75, 1.0], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            grid={'rows': 1, 'columns': 2, 'pattern': "independent"},
            height=260,
            margin={"t":40,"b":20,"l":20,"r":20}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Early Warning Queue Alert Box
        st.markdown("#### District Early Warning Queue")
        high_risk_districts = df_map[df_map["category"].isin(["High", "Severe"])]
        if not high_risk_districts.empty:
            for _, r in high_risk_districts.iterrows():
                badge_cls = "risk-severe" if r["category"] == "Severe" else "risk-high"
                st.markdown(f"<div class='{badge_cls}'>⚠️ {r['district']}: {r['category']} {r['dominant_hazard']} Risk (Score: {r['overall_score']:.2f})</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-low'>✅ All districts currently at Normal or Moderate risk thresholds.</div>", unsafe_allow_html=True)

# =========================================================================
# PERSPECTIVE 2: AGRICULTURAL EXTENSION & FARMER
# =========================================================================
elif user_role == "Agricultural Extension & Farmer":
    st.subheader(f"Adaptive Agriculture Advisory: {selected_district} District")
    st.markdown(f"**Agro-Climatic Zone:** {advisory['agro_climatic_zone']}")

    col_crops, col_advice = st.columns([1, 1])

    with col_crops:
        st.markdown("#### 🌱 Recommended Climate-Resilient Cultivars")
        crops_df = pd.DataFrame(advisory["recommended_crops"])
        st.table(crops_df)

        st.markdown("#### 🚫 Crops to Deprioritize / Avoid")
        for ca in advisory["crops_to_avoid"]:
            st.markdown(f"- ❌ **{ca}**")

    with col_advice:
        st.markdown("#### 📅 Dynamic Sowing Window Strategy")
        st.info(advisory["sowing_strategy"])

        st.markdown("#### 💧 Water Management & Irrigation Scheduling")
        st.warning(advisory["irrigation_strategy"])

        st.markdown("#### 📋 Field-Level Agronomic Guidelines")
        for gl in advisory["actionable_guidelines"]:
            st.markdown(f"• {gl}")

    # ── AGRICULTURE CHATBOT ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Agriculture Advisory Chatbot")
    st.markdown(
        "*Ask me anything about Tamil Nadu farming — crops, ENSO effects, "
        "irrigation, sowing windows, soil, pest management, or government schemes.*"
    )

    # Add chatbot styling
    st.markdown("""
    <style>
        /* Chat message containers */
        [data-testid="stChatMessage"] {
            background: rgba(8, 16, 36, 0.82) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 12px !important;
            margin-bottom: 8px !important;
        }
        /* Chat input */
        [data-testid="stChatInput"] textarea {
            background: rgba(8, 16, 36, 0.90) !important;
            color: #E2E8F0 !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
            border-radius: 10px !important;
        }
        /* User bubble */
        [data-testid="stChatMessage"][data-role="user"] {
            border-left: 4px solid #38BDF8 !important;
        }
        /* Assistant bubble */
        [data-testid="stChatMessage"][data-role="assistant"] {
            border-left: 4px solid #6EE7B7 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize chat history in session state
    if "agri_chat_history" not in st.session_state:
        st.session_state.agri_chat_history = []

    # Quick-question buttons
    st.markdown("**💡 Quick Questions:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    quick_questions = {
        "🌧️ El Niño effects": "How does El Niño affect Tamil Nadu rainfall and crops?",
        "🏜️ Drought crops": "Which crops should I grow during drought conditions?",
        "🌊 Flood variety": "Which paddy variety tolerates flooding and submergence?",
        "💧 AWD irrigation": "What is AWD irrigation and how does it save water?",
    }
    for col, (label, question) in zip([qcol1, qcol2, qcol3, qcol4], quick_questions.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.agri_chat_history.append({"role": "user", "content": question})
                bot_resp = get_agriculture_response(question, selected_district, advisory)
                st.session_state.agri_chat_history.append({"role": "assistant", "content": bot_resp})
                st.rerun()

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.agri_chat_history:
            with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🌿"):
                st.markdown(msg["content"])

    # Chat input box
    user_query = st.chat_input(
        "Ask your agriculture question... (e.g. 'Best crop for Coimbatore in drought?')"
    )
    if user_query:
        st.session_state.agri_chat_history.append({"role": "user", "content": user_query})
        bot_response = get_agriculture_response(user_query, selected_district, advisory)
        st.session_state.agri_chat_history.append({"role": "assistant", "content": bot_response})
        st.rerun()

    # Clear chat button
    if st.session_state.agri_chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.agri_chat_history = []
            st.rerun()

# =========================================================================
# PERSPECTIVE 3: RESEARCH & MODEL DIAGNOSTICS
# =========================================================================
else:
    st.subheader(f"Explainable AI (SHAP) & Benchmark Evaluation: {selected_district}")

    tab_shap, tab_perf = st.tabs(["Explainable AI (SHAP)", "Model Benchmark Evaluation"])

    with tab_shap:
        st.markdown("#### Local Prediction Attribution (Why this prediction?)")
        shap_key = f"{selected_district}_{selected_date}"
        local_shap = shap_cache.get(shap_key, None)

        if local_shap:
            st.markdown(f"**Base Value (Mean Historical Normal):** `{local_shap['base_value_mm']} mm` | **Forecasted Anomaly:** `{local_shap['predicted_anomaly_mm']} mm`")

            # Waterfall / Horizontal Bar of drivers
            drivers = []
            for item in local_shap["top_positive_drivers"]:
                drivers.append({"Feature": item["feature"], "Attribution (mm)": item["attribution_mm"], "Direction": "Wet Signal (+)"})
            for item in local_shap["top_negative_drivers"]:
                drivers.append({"Feature": item["feature"], "Attribution (mm)": item["attribution_mm"], "Direction": "Dry Signal (-)"})

            df_drivers = pd.DataFrame(drivers).sort_values("Attribution (mm)")
            fig_bar = px.bar(
                df_drivers,
                x="Attribution (mm)",
                y="Feature",
                color="Direction",
                color_discrete_map={"Wet Signal (+)": "#3B82F6", "Dry Signal (-)": "#EF4444"},
                orientation="h",
                title=f"SHAP Driver Contributions for {selected_district} ({selected_date})"
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Select a date within 2024–2025 to view localized SHAP attribution.")

        st.markdown("#### Global SHAP Feature Importance across Tamil Nadu")
        try:
            df_global = pd.read_csv("reports/shap_feature_importance.csv").head(12)
            fig_global = px.bar(
                df_global.sort_values("mean_abs_shap_importance"),
                x="mean_abs_shap_importance",
                y="feature",
                orientation="h",
                title="Top 12 Leading Predictors (Mean |SHAP Value| in mm)",
                labels={"mean_abs_shap_importance": "Mean |SHAP| (mm)", "feature": "Predictor"}
            )
            fig_global.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                height=420
            )
            st.plotly_chart(fig_global, use_container_width=True)
        except Exception as e:
            st.warning(f"Global SHAP feature importance file not found: {e}")

    with tab_perf:
        st.markdown("#### Quantitative Benchmark Comparison (Unseen Test Partition: 2024–2025)")
        st.dataframe(df_metrics, use_container_width=True)

        st.markdown("#### Multiclass Hazard Risk Classification Performance")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Random Forest Classifier")
            rf_m = clf_metrics["Random Forest Classifier"]
            st.metric("Accuracy", f"{rf_m['accuracy'] * 100:.2f}%")
            st.metric("Macro F1-Score", f"{rf_m['macro_f1']:.4f}")
        with col_c2:
            st.markdown("##### XGBoost Classifier")
            xgb_m = clf_metrics["XGBoost Classifier"]
            st.metric("Accuracy", f"{xgb_m['accuracy'] * 100:.2f}%")
            st.metric("Macro F1-Score", f"{xgb_m['macro_f1']:.4f}")

        st.caption("Evaluation strictly on chronological test partition (Jan 2024 to Dec 2025). Normalization parameters computed strictly on training data.")

# Footer
st.markdown("---")
st.markdown(
    "<small style='color: #6B7280;'>"
    "Tamil Nadu Climate Risk & Adaptive Decision Support System | "
    "Designed in compliance with the 'Predict → Assess → Prepare → Adapt' framework | "
    "Datasets: IMD Sub-Division (1901–2017), HadISST DMI (1870–2026), NOAA ONI (1950–2026), NASA POWER (2015–2024), Multi-District Climate-Crop Panel (2018–2025)."
    "</small>",
    unsafe_allow_html=True
)
