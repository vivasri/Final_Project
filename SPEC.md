# Project Specification: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

**Grounded Reference:** Based on the research paper *"AI-Driven Climate Risk Forecasting, Disaster Risk Assessment, and Adaptive Decision Support System for India"* (Mohan Kumar M., Santhosh Kumar P., Sushmitha K., Tharani M., Vivasri S. — KGiSL Institute of Technology).

---

## 1. Executive Summary & Objective

The objective of this project is to develop an end-to-end, reproducible, and explainable prototype for monthly climate risk forecasting and adaptive agricultural decision support for districts in **Tamil Nadu, India**. 

The system implements the **"Predict → Assess → Prepare → Adapt"** workflow:
1. **Predict:** Forecast monthly rainfall and temperature anomalies using leading global ocean-atmosphere oscillation indices (ENSO Niño 3.4 and IOD DMI) along with antecedent local meteorological conditions.
2. **Assess:** Compute multi-hazard risk scores for **Drought** and **Flood** using an auditable, multi-criteria risk formulation, categorizing risks into *Low*, *Moderate*, *High*, and *Severe*.
3. **Prepare & Adapt:** Generate actionable, district-specific agronomic advice (crop selection, sowing window shifts, irrigation scheduling) and hazard mitigation alerts through an auditable rule-based decision layer decoupled from the statistical models.
4. **Explain:** Interpret model forecasts and risk drivers using Shapley Additive exPlanations (**SHAP**), providing trust and transparency for decision-makers.
5. **Present:** Surface interactive hazard maps, forecasts, alerts, and explanations in an intuitive **Streamlit** dashboard.

---

## 2. Project Scope & Boundary Conditions (MVP)

* **Target Region:** Tamil Nadu, India (district-level resolution, e.g., Coimbatore, Thanjavur, Chennai, Madurai, Tiruchirappalli, Salem, Tirunelveli, etc.).
* **Temporal Resolution:** Monthly time step; forecast lead times of $h = 1$ to $3$ months ahead.
* **Target Hazards (MVP):** 
  * **Drought:** Driven by negative rainfall anomalies, antecedent precipitation deficit, and soil moisture stress.
  * **Flood:** Driven by extreme surplus precipitation, high antecedent wetness, and regional drainage vulnerability.
  * *(Heatwaves, landslides, and extended water scarcity deferred to post-MVP).*
* **Model Families:**
  * **Baselines (Mandatory):** Climatological Monthly Mean, Persistence ($y_{t+h} = y_t$).
  * **Machine Learning:** Random Forest Regressor/Classifier, Regularized XGBoost.
  * **Deep Learning (Sequential):** Long Short-Term Memory (LSTM) network.
* **Core Language & Frameworks:** Python 3.11+, Pandas, Scikit-learn, XGBoost, PyTorch/TensorFlow, SHAP, Streamlit.

---

## 3. Data Architecture & Ingestion Plan

The system integrates four core datasets provided for the project:

| Dataset Identifier | Format | Source / Coverage | Role in System | Key Variables |
| :--- | :--- | :--- | :--- | :--- |
| **`Sub_Division_IMD_2017.xlsx`** | Excel (.xlsx) | IMD (1901–2017) | Long-term monthly precipitation climatology & historical anomaly baselines for Tamil Nadu | `JAN`, `FEB`, ..., `DEC`, `ANNUAL`, `JF`, `MAM`, `JJAS`, `OND` |
| **`POWER_Point_Daily_...`** | CSV / Text | NASA POWER (2015–2024) | High-resolution daily local meteorology for Tamil Nadu coordinates | `PRECTOTCORR` (rain), `T2M` (temp), `T2M_MAX`, `T2M_MIN`, `RH2M` (humidity), `ALLSKY_SFC_SW_DWN` |
| **`El-nino-dataset`** | CSV / Text | NOAA / Climate Prediction Center | Oceanic Niño Index (ONI) / Niño 3.4 Sea Surface Temperature Anomaly | Monthly Niño 3.4 SST anomaly, ENSO phase indicator |
| **`dmi.had.long.nc`** | NetCDF (.nc) | HadISST / NOAA PSL | Indian Ocean Dipole (IOD) Dipole Mode Index (long-term monthly) | `dmi` (Dipole Mode Index monthly time series) |
| **District Geo & Static Data** | JSON / Shapefile / Tabular | State Agricultural & Disaster Portals | Static district characteristics ($s_d$) | Soil type, terrain/slope, irrigation access, drainage capacity |

### Data Processing Integrity Rules:
1. **Strict No-Fabrication:** Missing or incomplete records must be handled via verified interpolation or explicit masking; no synthetic numbers fabricated.
2. **Harmonization:** Daily records (NASA POWER) will be aggregated to calendar-month summaries (total precipitation, mean/min/max temperature, mean relative humidity).
3. **Temporal Alignment:** All time series will be aligned on a common monthly date index `YYYY-MM-01`.

---

## 4. Feature Engineering & Preprocessing

Following Section V-B of the project report:

### A. Feature Extraction
1. **Teleconnection Lag Features:**
   * ENSO (Niño 3.4 anomaly) at lags $t-1, t-2, t-3, t-4, t-5, t-6$.
   * IOD (DMI) at lags $t-1, t-2, t-3, t-4, t-5, t-6$.
   * Combined ENSO-IOD state interaction ($ENSO_{t-1} \times IOD_{t-1}$).
2. **Antecedent Local Precipitation:**
   * Cumulative 30-day, 60-day, and 90-day rainfall ($R_{\text{cum30}}, R_{\text{cum60}}, R_{\text{cum90}}$).
   * Rolling precipitation deviation from 10-year monthly moving average.
3. **Temperature & Moisture Features:**
   * Monthly mean temperature $T_{\text{mean}}$, maximum temperature $T_{\text{max}}$, and relative humidity $RH$.
   * Vapor Pressure Deficit (VPD) proxy where temperature and humidity coincide.
4. **Seasonality Indicators:**
   * Cyclical month encoding: $\sin(2\pi \cdot m / 12)$ and $\cos(2\pi \cdot m / 12)$.
   * Monsoon season flags: Northeast Monsoon (NEM, Oct–Dec: primary rainfall period for Tamil Nadu) and Southwest Monsoon (SWM, Jun–Sep).

### B. Target Variables
* **Primary Target 1:** Monthly rainfall anomaly $\Delta R_{d, t+h} = R_{d, t+h} - \mu_{m, d}^{\text{train}}$ (mm).
* **Primary Target 2:** Monthly temperature anomaly $\Delta T_{d, t+h} = T_{d, t+h} - \mu_{m, d}^{\text{train}}$ (°C).

### C. Normalization & Leakage Prevention
Continuous predictors are standardized using Z-score normalization:
$$x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
* $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are calculated **strictly on training partitions**.
* No test or validation information is exposed during pre-processing, imputation, or scaling.

---

## 5. Splitting Strategy & Evaluation Protocol

### A. Strict Chronological Split
Randomized K-fold splits are strictly prohibited due to temporal autocorrelation in climate sequences.
* **Training Partition:** Earliest available years up to December 2018 (~75% historical data).
* **Validation Partition:** January 2019 to December 2021 (~15% tuning/hyperparameter selection).
* **Test Partition:** January 2022 to December 2024 (~10% completely unseen recent seasons).

### B. Quantitative Evaluation Metrics
* **Regression (Anomaly Forecasting):**
  * Root Mean Squared Error (RMSE)
  * Mean Absolute Error (MAE)
  * Coefficient of Determination ($R^2$)
  * Correlation Coefficient ($r$) between observed and predicted anomalies
* **Classification (Hazard Risk Categories):**
  * Accuracy, Precision, Recall, Macro-averaged F1-Score
  * Confusion Matrix across Low, Moderate, High, Severe classes.
* **Honest Benchmarking Rule:** All ML/DL models will be benchmarked directly against Climatology and Persistence baselines. Any failure of advanced models to surpass baselines will be reported transparently with diagnostic analysis.

---

## 6. Modeling Pipeline

```
           +-----------------------------------------------+
           | Raw Datasets (IMD, NASA POWER, ENSO, DMI)     |
           +-----------------------------------------------+
                                  |
                                  v
           +-----------------------------------------------+
           | Ingestion, Harmonization & Monthly Alignment  |
           +-----------------------------------------------+
                                  |
                                  v
           +-----------------------------------------------+
           | Feature Engineering (Lags 1-6, Rolling 30/60/90)|
           +-----------------------------------------------+
                                  |
               +------------------+------------------+
               |                  |                  |
               v                  v                  v
        +--------------+   +--------------+   +--------------+
        |  Baselines   |   | Random Forest|   |    LSTM      |
        | Climatology, |   |  & XGBoost   |   |  Recurrent   |
        | Persistence  |   | (Tabular ML) |   | (Sequential) |
        +--------------+   +--------------+   +--------------+
               |                  |                  |
               +------------------+------------------+
                                  |
                                  v
           +-----------------------------------------------+
           | Forecast Output: y_hat (Rain & Temp Anomaly)  |
           +-----------------------------------------------+
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
+-------------------------------+     +-------------------------------+
| Multi-Hazard Risk Module      |     | Adaptive Agriculture Module   |
| (Drought & Flood RiskScore)   |     | (Crop, Sowing, Irrigation)    |
| Low, Moderate, High, Severe   |     | Rule-Based Decoupled Layer    |
+-------------------------------+     +-------------------------------+
               |                                     |
               +------------------+------------------+
                                  |
                                  v
           +-----------------------------------------------+
           | Explainable AI (SHAP attributions & drivers)  |
           +-----------------------------------------------+
                                  |
                                  v
           +-----------------------------------------------+
           | Streamlit Decision Support Dashboard          |
           +-----------------------------------------------+
```

### A. Model Specifications
1. **Baselines:**
   * *Climatology:* $\hat{y}_{t+h} = \text{mean}(y_{\text{train, month}})$. Anomaly prediction is zero.
   * *Persistence:* $\hat{y}_{t+h} = y_t$. Future anomaly equals the current month's observed anomaly.
2. **Random Forest & XGBoost:**
   * Optimized with time-series rolling cross-validation on the training set.
   * Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`.
3. **LSTM Network:**
   * Sequence-to-point architecture ingesting a sliding window of $L = 6$ to $12$ months.
   * Architecture: 1-2 LSTM layers (32-64 hidden units), Dropout (0.2), Dense output layer.

---

## 7. Multi-Hazard Risk Scoring Formulation

Following Section VII (Eq. 8) of the paper, continuous forecasts are transformed into standardized risk scores:

$$\text{RiskScore}_k = \sum_{j} w_{k,j} \cdot p_{k,j}, \quad \text{where } \sum_{j} w_{k,j} = 1$$

* $p_{k,j} \in [0, 1]$: Normalized indicator $j$ for hazard $k$.
* $w_{k,j}$: Weight assigned to indicator $j$.

### Hazard-Specific Indicator Mapping (MVP):

#### 1. Drought Risk ($k = \text{drought}$):
* Predictor $p_1$: Negative rainfall anomaly magnitude (severe deficit $\to 1.0$).
* Predictor $p_2$: Antecedent 90-day rainfall deficit.
* Predictor $p_3$: Positive temperature anomaly (evaporative demand).
* Predictor $p_4$: District groundwater/irrigation vulnerability factor ($s_d$).
* Initial Weights: $w = [0.45, 0.25, 0.15, 0.15]$.

#### 2. Flood Risk ($k = \text{flood}$):
* Predictor $p_1$: Forecasted surplus rainfall anomaly (heavy precipitation $\to 1.0$).
* Predictor $p_2$: Antecedent 30-day cumulative rainfall (soil saturation proxy).
* Predictor $p_3$: District low-elevation / poor-drainage index ($s_d$).
* Initial Weights: $w = [0.50, 0.30, 0.20]$.

### Categorical Risk Mapping:
* $[0.00, 0.25) \to$ **Low**
* $[0.25, 0.50) \to$ **Moderate**
* $[0.50, 0.75) \to$ **High**
* $[0.75, 1.00] \to$ **Severe**

*Note:* Thresholds and weights are fully auditable and decoupled from the ML models, allowing updates based on historical disaster records or stakeholder review.

---

## 8. Adaptive Agriculture Advisory Module

The advisory module solves the rule-based optimization problem:
$$A^*_d = \arg\max_{A} u(A, \hat{y}_{d, t+h}, s_d) \quad \text{subject to } A \in \mathcal{C}_d$$

### Agronomic Decision Matrix for Tamil Nadu Agro-Climatic Zones:

| Climate Outlook / Hazard Level | Recommended Crop Strategy | Sowing Window Strategy | Irrigation Advisory |
| :--- | :--- | :--- | :--- |
| **Severe Drought Risk** | Switch to ultra-short-duration millets (Barnyard millet / Kudiraivali, Foxtail millet / Thinai) or drought-hardy pulses (Black gram / Vamban-8, Horsegram). Avoid Samba paddy. | Delay dry sowing until initial soil moisture recharge; adopt dry nursery technique if paddy is unavoidable. | Alternate Wetting and Drying (AWD), micro-irrigation / drip priority, restrict canal releases. |
| **Moderate Drought Risk** | Shift to medium-duration drought-tolerant rice varieties (CR 1009 Sub 1, TKM 13) or maize / pulses. | Stagger sowing by 10–14 days based on predicted monsoon onset. | Deficit irrigation scheduling; mulch application to reduce evaporative loss. |
| **Normal / Low Risk** | Standard regional cropping pattern (Kuruvai / Thaladi / Samba paddy in Thanjavur/Cauvery delta; cotton/maize in Western zone). | Normal seasonal sowing schedule. | Standard scheduled irrigation based on crop phenological stage. |
| **High / Severe Flood Risk** | Submergence-tolerant paddy varieties (Swarna-Sub1, CR 1009 Sub 1); avoid flood-sensitive cash crops in low-lying tracts. | Advance sowing to achieve stem establishment before peak flood window or postpone nursery transplanting. | Open field drainage channels, clear drainage silt, de-silt farm ponds and bund breaches. |

---

## 9. Explainable AI (SHAP) Strategy

To ensure model trust for both agricultural extension officers and district disaster managers:
* **TreeSHAP** applied to Random Forest and XGBoost forecasting models.
* **Global Explanations:** Feature importance ranking identifying leading predictors across the entire dataset (e.g., verifying whether Niño 3.4 at lag-3 or lag-4 dominates Northeast Monsoon forecast).
* **Local Explanations:** For each district's monthly forecast, compute a waterfall/force plot showing how individual predictors (e.g., $+1.2^\circ\text{C}$ Niño 3.4 anomaly, $-45\text{ mm}$ 90-day antecedent rain) pushed the prediction away from the historical base value.

---

## 10. Streamlit Dashboard Architecture

The dashboard will be developed in `app/streamlit_app.py` with the following views:
1. **Executive Disaster Management View:**
   * Interactive choropleth map of Tamil Nadu districts colored by Drought and Flood risk levels (*Low, Moderate, High, Severe*).
   * Active Warning Queue: Districts currently in *High* or *Severe* status with one-click drilldown.
2. **District Climate Forecast View:**
   * Time series chart of historical rainfall/temperature with future forecasts and uncertainty band.
   * Global Teleconnection telemetry: Live ENSO Niño 3.4 index, IOD DMI status, and historical analogues.
3. **Adaptive Agriculture Decision Portal:**
   * District-specific crop advisories, optimal sowing calendar shifts, and water-conserving irrigation guidelines.
4. **Model Transparency & SHAP Explorer:**
   * SHAP waterfall plots for selected district predictions.
   * Model comparison scorecard (Baselines vs Random Forest vs XGBoost vs LSTM).

---

## 11. Project Directory Structure

```
c:\Users\user\Downloads\DISASTER\
│
├── data/
│   ├── raw/                 # Original, untouched input datasets
│   │   ├── Sub_Division_IMD_2017.xlsx
│   │   ├── POWER_Point_Daily_20150101_20241231_011d02N_076d96E_LST.csv
│   │   ├── El-nino-dataset.csv
│   │   ├── dmi.had.long.nc
│   │   └── tn_districts_metadata.json
│   └── processed/           # Cleaned, aggregated, and aligned datasets
│       ├── tn_monthly_features.csv
│       ├── train_features.csv
│       ├── val_features.csv
│       └── test_features.csv
│
├── src/
│   ├── __init__.py
│   ├── ingest.py            # Dataset loading, raw file parsing, validation
│   ├── features.py          # Lag creation, rolling sums, no-leakage Z-score scaling
│   ├── train.py             # Baseline, RF, XGBoost, and LSTM training & evaluation
│   ├── risk.py              # Multi-hazard risk scoring (Drought, Flood) & thresholding
│   ├── advisory.py          # Decoupled agro-advisory rule engine
│   └── explain.py           # SHAP attribution generation & explanation utilities
│
├── models/                  # Serialized model artifacts (.pkl, .pt / .keras)
├── app/
│   ├── __init__.py
│   └── streamlit_app.py     # Interactive Streamlit dashboard
│
├── notebooks/               # Exploratory notebooks (data exploration, validation)
├── reports/                 # Evaluation metric tables, SHAP figures, error analysis
│
├── SPEC.md                  # System specification & technical architecture (this document)
├── README.md                # Quickstart, execution guide, and phase-by-phase updates
└── requirements.txt         # Pinned Python package dependencies
```

---

## 12. Implementation Phases & Review Gates

| Phase | Title | Key Deliverables | Verification / Review Gate |
| :---: | :--- | :--- | :--- |
| **0** | **Specification & Setup** | `SPEC.md`, `README.md`, `requirements.txt`, directory scaffolding | **Current Phase (Waiting for user approval)** |
| **1** | **Data Ingestion & Harmonization** | `src/ingest.py`: parse IMD, NASA POWER, ENSO, DMI; align monthly to Tamil Nadu; write `data/processed/tn_monthly_features.csv` | Shapes, date ranges, null counts, sample head/tail rows |
| **2** | **Feature Engineering & Leakage-Free Splitting** | `src/features.py`: Lags 1–6, 30/60/90-day rolling sums, chronological train/val/test splits, training-only scaler statistics | Feature count, split dates, verify $\mu_{\text{train}}, \sigma_{\text{train}}$ unexposed to test |
| **3** | **Model Training & Benchmarking** | `src/train.py`: Climatology & Persistence baselines $\to$ Random Forest $\to$ XGBoost $\to$ LSTM | Honest test RMSE, MAE, $R^2$, benchmark comparison table |
| **4** | **Risk Assessment & Adaptive Agriculture** | `src/risk.py` & `src/advisory.py`: Multi-hazard formula (Eq. 8), category calibration, agronomic decision matrix | Sample district risk outputs, classification metrics, advisory audit |
| **5** | **Explainability (SHAP) & Diagnostics** | `src/explain.py`: TreeSHAP summary, district waterfall plots, feature importance exports | SHAP summary plots, top driving features for extreme events |
| **6** | **Interactive Streamlit Dashboard** | `app/streamlit_app.py`: District maps, forecast trends, alerts, crop advice, and SHAP cards | Functional app execution, screenshot/inspection, end-to-end demo |

---
*End of SPEC.md*
