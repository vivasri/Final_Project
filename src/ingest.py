"""
Data Ingestion and Harmonization Module
=======================================
Project: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

This module ingests raw hydro-meteorological, ocean-atmosphere teleconnection,
and district-level risk datasets, aligns them along a monthly temporal reference,
and produces harmonized analysis-ready datasets in data/processed/.

Data Sources Ingested:
1. Sub_Division_IMD_2017.xlsx: IMD monthly sub-divisional rainfall (1901-2017).
2. dmi.had.long.nc: HadISST Indian Ocean Dipole Mode Index (1870-2026).
3. El_nino_dataset.csv: NOAA ONI / Niño 3.4 SST anomaly index (1950-2026).
4. POWER_Point_Daily.csv: NASA POWER daily meteorology for Tamil Nadu (2015-2024).
5. AI_Climate_Disaster_Crop_Dataset_10000.csv: Tamil Nadu 10-district climate,
   crop, and multi-hazard records (2018-2025).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_imd_subdivision(filepath: str = "data/raw/Sub_Division_IMD_2017.xlsx") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingest IMD monthly sub-divisional rainfall dataset and extract Tamil Nadu.
    
    Returns:
        df_monthly: Long-format monthly rainfall time series (1901-2017).
        df_climatology: Long-term monthly mean and standard deviation (117 years).
    """
    logging.info(f"Loading IMD sub-divisional rainfall from {filepath}")
    df_raw = pd.read_excel(filepath)
    
    # Filter for Tamil Nadu subdivision
    sub_col = df_raw.columns[0]
    df_tn = df_raw[df_raw[sub_col].astype(str).str.strip().str.upper() == "TAMIL NADU"].copy()
    
    month_cols = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    month_map = {m: i + 1 for i, m in enumerate(month_cols)}
    
    # Melt monthly columns to tidy long format
    df_monthly = df_tn.melt(
        id_vars=["YEAR"],
        value_vars=month_cols,
        var_name="month_str",
        value_name="rainfall_imd_subdiv_mm"
    )
    df_monthly["month"] = df_monthly["month_str"].map(month_map)
    df_monthly = df_monthly.rename(columns={"YEAR": "year"})
    df_monthly["date"] = pd.to_datetime(
        df_monthly["year"].astype(str) + "-" + df_monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    df_monthly = df_monthly.sort_values("date").reset_index(drop=True)
    df_monthly = df_monthly[["date", "year", "month", "rainfall_imd_subdiv_mm"]]
    
    # Compute 117-year monthly climatology
    df_climatology = df_monthly.groupby("month")["rainfall_imd_subdiv_mm"].agg(
        imd_clim_mean_mm="mean",
        imd_clim_std_mm="std"
    ).reset_index()
    
    logging.info(f"Loaded IMD Tamil Nadu: {len(df_monthly)} monthly records ({df_monthly['year'].min()}-{df_monthly['year'].max()})")
    return df_monthly, df_climatology


def load_dmi_index(filepath: str = "data/raw/dmi.had.long.nc") -> pd.DataFrame:
    """
    Ingest HadISST Indian Ocean Dipole Mode Index (DMI) from NetCDF.
    
    Returns:
        df_dmi: Monthly DMI values with standard date index.
    """
    logging.info(f"Loading HadISST DMI index from {filepath}")
    ds = xr.open_dataset(filepath)
    df = ds.to_dataframe().reset_index()
    df = df.rename(columns={"time": "date", "value": "iod_dmi"})
    
    # Standardize to 1st of each month
    df["date"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-01"))
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df[["date", "year", "month", "iod_dmi"]].sort_values("date").reset_index(drop=True)
    
    logging.info(f"Loaded DMI index: {len(df)} monthly observations ({df['year'].min()}-{df['year'].max()})")
    return df


def load_enso_oni(filepath: str = "data/raw/El_nino_dataset.csv") -> pd.DataFrame:
    """
    Ingest NOAA Oceanic Niño Index (ONI) 3-month running mean SST anomalies.
    The raw file contains overlapping seasons: DJF, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND, NDJ.
    
    Returns:
        df_enso: Monthly ONI anomaly mapped to centered months (1 to 12).
    """
    logging.info(f"Loading ENSO ONI index from {filepath}")
    df_raw = pd.read_excel(filepath)
    
    seasons = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    season_to_month = {s: i + 1 for i, s in enumerate(seasons)}
    
    df_melt = df_raw.melt(
        id_vars=["Year"],
        value_vars=seasons,
        var_name="season",
        value_name="enso_oni"
    )
    df_melt["month"] = df_melt["season"].map(season_to_month)
    df_melt = df_melt.rename(columns={"Year": "year"})
    df_melt["date"] = pd.to_datetime(
        df_melt["year"].astype(str) + "-" + df_melt["month"].astype(str).str.zfill(2) + "-01"
    )
    df_enso = df_melt.sort_values("date").dropna(subset=["enso_oni"]).reset_index(drop=True)
    df_enso = df_enso[["date", "year", "month", "enso_oni"]]
    
    logging.info(f"Loaded ENSO ONI index: {len(df_enso)} records ({df_enso['year'].min()}-{df_enso['year'].max()})")
    return df_enso


def load_nasa_power_monthly(filepath: str = "data/raw/POWER_Point_Daily.csv") -> pd.DataFrame:
    """
    Ingest NASA POWER daily meteorological observations and aggregate to monthly.
    Location: 11.0168°N, 76.9558°E (Tamil Nadu, Coimbatore).
    
    Returns:
        df_power_monthly: Monthly aggregations of temperature, humidity, wind, solar radiation.
    """
    logging.info(f"Loading NASA POWER daily observations from {filepath}")
    df_raw = pd.read_excel(filepath, header=14)
    
    # Filter valid rows and parse Day of Year (DOY) to datetime
    df_raw = df_raw.dropna(subset=["YEAR", "DOY"]).copy()
    df_raw["YEAR"] = df_raw["YEAR"].astype(int)
    df_raw["DOY"] = df_raw["DOY"].astype(int)
    
    df_raw["date"] = pd.to_datetime(
        df_raw["YEAR"].astype(str) + df_raw["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )
    
    # Replace missing value sentinel (-999) with NaN if present
    cols_to_check = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M", "ALLSKY_SFC_SW_DWN"]
    for col in cols_to_check:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].replace(-999, np.nan)
            
    df_raw["year"] = df_raw["date"].dt.year
    df_raw["month"] = df_raw["date"].dt.month
    
    # Aggregate daily observations to calendar-month summaries
    df_monthly = df_raw.groupby(["year", "month"]).agg(
        power_t2m_mean=("T2M", "mean"),
        power_t2m_max=("T2M_MAX", "mean"),
        power_t2m_min=("T2M_MIN", "mean"),
        power_rh2m_mean=("RH2M", "mean"),
        power_ws10m_mean=("WS10M", "mean"),
        power_solar_irradiance=("ALLSKY_SFC_SW_DWN", "mean")
    ).reset_index()
    
    df_monthly["date"] = pd.to_datetime(
        df_monthly["year"].astype(str) + "-" + df_monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    df_monthly = df_monthly[["date", "year", "month", "power_t2m_mean", "power_t2m_max", "power_t2m_min",
                             "power_rh2m_mean", "power_ws10m_mean", "power_solar_irradiance"]]
    
    logging.info(f"Aggregated NASA POWER: {len(df_monthly)} monthly steps ({df_monthly['year'].min()}-{df_monthly['year'].max()})")
    return df_monthly


def load_district_dataset(filepath: str = "data/raw/AI_Climate_Disaster_Crop_Dataset_10000.csv") -> tuple[pd.DataFrame, dict]:
    """
    Ingest Tamil Nadu multi-district climate, crop, and disaster risk dataset (10,000 records).
    Aggregates to a monthly district panel (960 district-months across 10 districts).
    
    Returns:
        df_district_monthly: District-month panel dataframe.
        district_metadata: Dictionary of static geographic and soil characteristics per district.
    """
    logging.info(f"Loading Tamil Nadu district dataset from {filepath}")
    df_raw = pd.read_csv(filepath)
    df_raw["date_dt"] = pd.to_datetime(df_raw["date"], format="%d/%m/%Y")
    df_raw["year"] = df_raw["date_dt"].dt.year
    df_raw["month"] = df_raw["date_dt"].dt.month
    
    districts = sorted(df_raw["district"].unique().tolist())
    logging.info(f"Districts present ({len(districts)}): {districts}")
    
    # Extract static district characteristics
    static_cols = [
        "latitude", "longitude", "elevation_m", "slope_degree",
        "drainage_distance_km", "population_density_per_km2",
        "soil_ph", "soil_nitrogen_mgkg", "soil_clay_percent",
        "soil_sand_percent", "soil_organic_carbon_percent"
    ]
    df_static = df_raw.groupby("district")[static_cols].mean().reset_index()
    
    metadata = {}
    for _, row in df_static.iterrows():
        d_name = row["district"]
        metadata[d_name] = {
            "latitude": round(float(row["latitude"]), 4),
            "longitude": round(float(row["longitude"]), 4),
            "elevation_m": round(float(row["elevation_m"]), 2),
            "slope_degree": round(float(row["slope_degree"]), 2),
            "drainage_distance_km": round(float(row["drainage_distance_km"]), 2),
            "population_density": round(float(row["population_density_per_km2"]), 1),
            "soil_ph": round(float(row["soil_ph"]), 2),
            "soil_clay_percent": round(float(row["soil_clay_percent"]), 1),
            "soil_sand_percent": round(float(row["soil_sand_percent"]), 1),
            "soil_nitrogen_mgkg": round(float(row["soil_nitrogen_mgkg"]), 1),
            "soil_organic_carbon_percent": round(float(row["soil_organic_carbon_percent"]), 2)
        }
        
    # Aggregate observations by (district, year, month)
    # Climate variables: monthly averages / sums
    # Hazard probabilities: average monthly risk exposure
    agg_dict = {
        "rainfall_mm": "sum",             # Monthly cumulative rainfall
        "temperature_c": "mean",          # Monthly mean temperature
        "humidity_percent": "mean",       # Monthly mean humidity
        "wind_speed_kmph": "mean",        # Monthly mean wind speed
        "pressure_hpa": "mean",           # Monthly mean pressure
        "soil_moisture": "mean",          # Monthly mean soil moisture
        "ndvi": "mean",                   # Monthly vegetation index
        "rainfall_deficit_mm": "mean",    # Monthly rainfall deficit
        "drought_probability": "mean",    # Monthly drought exposure
        "flood_probability": "mean",      # Monthly flood exposure
        "heatwave_probability": "mean",   # Monthly heatwave exposure
        "landslide_probability": "mean",  # Monthly landslide exposure
        "water_stress_level": "mean",     # Monthly water stress level
        "overall_risk_score": "mean",     # Multi-hazard score
        "crop_yield_ton_per_ha": "mean",  # Average crop yield
        "crop_suitability_score": "mean"  # Average suitability
    }
    
    df_panel = df_raw.groupby(["district", "year", "month"]).agg(agg_dict).reset_index()
    
    # Map predominant risk level per district-month
    risk_mode = df_raw.groupby(["district", "year", "month"])["risk_level"].agg(
        lambda s: s.mode().iloc[0] if not s.empty else "Low"
    ).reset_index()
    df_panel = df_panel.merge(risk_mode, on=["district", "year", "month"])
    
    # Add standardized date
    df_panel["date"] = pd.to_datetime(
        df_panel["year"].astype(str) + "-" + df_panel["month"].astype(str).str.zfill(2) + "-01"
    )
    
    # Merge static attributes back for complete district feature vector
    df_panel = df_panel.merge(df_static, on="district", how="left")
    
    df_panel = df_panel.sort_values(["district", "date"]).reset_index(drop=True)
    logging.info(f"Constructed district-month panel: {df_panel.shape} ({df_panel['year'].min()}-{df_panel['year'].max()})")
    
    return df_panel, metadata


def harmonize_all_datasets() -> pd.DataFrame:
    """
    Execute full Phase 1 ingestion and harmonization pipeline.
    Combines:
    - Tamil Nadu 10-district panel (2018-2025)
    - Global ENSO ONI index (1950-2026)
    - Global IOD DMI index (1870-2026)
    - IMD Tamil Nadu historical monthly climatology (1901-2017)
    - NASA POWER regional meteorological data (2015-2024)
    
    Saves outputs to:
    - data/processed/tn_monthly_features.csv
    - data/processed/tn_imd_climatology.csv
    - data/processed/tn_districts_metadata.json
    """
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Ingest IMD
    df_imd_monthly, df_imd_clim = load_imd_subdivision()
    df_imd_clim.to_csv("data/processed/tn_imd_climatology.csv", index=False)
    logging.info("Saved IMD climatology to data/processed/tn_imd_climatology.csv")
    
    # 2. Ingest DMI
    df_dmi = load_dmi_index()
    
    # 3. Ingest ENSO
    df_enso = load_enso_oni()
    
    # 4. Ingest NASA POWER
    df_power = load_nasa_power_monthly()
    
    # 5. Ingest District Panel & Metadata
    df_panel, district_meta = load_district_dataset()
    with open("data/processed/tn_districts_metadata.json", "w") as f:
        json.dump(district_meta, f, indent=2)
    logging.info("Saved district metadata to data/processed/tn_districts_metadata.json")
    
    # 6. Merge Teleconnections (ENSO & IOD) with District Panel
    df_merged = df_panel.merge(
        df_enso[["year", "month", "enso_oni"]],
        on=["year", "month"],
        how="left"
    )
    df_merged = df_merged.merge(
        df_dmi[["year", "month", "iod_dmi"]],
        on=["year", "month"],
        how="left"
    )
    
    # 7. Merge IMD Climatology Normals
    df_merged = df_merged.merge(
        df_imd_clim,
        on="month",
        how="left"
    )
    
    # 8. Merge NASA POWER reference data (where available)
    power_cols = ["year", "month", "power_t2m_mean", "power_rh2m_mean", "power_solar_irradiance"]
    df_merged = df_merged.merge(
        df_power[power_cols],
        on=["year", "month"],
        how="left"
    )
    
    # Sort chronologically by district and date
    df_merged = df_merged.sort_values(["district", "date"]).reset_index(drop=True)
    
    # Save final harmonized dataset
    out_csv = "data/processed/tn_monthly_features.csv"
    df_merged.to_csv(out_csv, index=False)
    logging.info(f"Phase 1 Complete. Saved harmonized dataset to {out_csv} (Shape: {df_merged.shape})")
    
    return df_merged


if __name__ == "__main__":
    df = harmonize_all_datasets()
    print("\n" + "=" * 60)
    print("PHASE 1 INGESTION SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {df.shape[1]}")
    print(f"Districts Covered: {df['district'].nunique()} ({df['district'].unique().tolist()})")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Months per District: {len(df) // df['district'].nunique()}")
    print("\nMissing Values Check:")
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) == 0:
        print("  Zero missing values across all core features!")
    else:
        print(nulls)
    print("\nSample Rows (Head 3):")
    print(df[["district", "date", "rainfall_mm", "temperature_c", "enso_oni", "iod_dmi", "risk_level"]].head(3))
    print("\nSample Rows (Tail 3):")
    print(df[["district", "date", "rainfall_mm", "temperature_c", "enso_oni", "iod_dmi", "risk_level"]].tail(3))
    print("=" * 60)
