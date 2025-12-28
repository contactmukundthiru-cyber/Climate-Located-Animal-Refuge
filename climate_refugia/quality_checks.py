from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def gps_quality_summary(gps_df: pd.DataFrame) -> Dict[str, float]:
    df = gps_df.copy()
    summary = {
        "points": float(len(df)),
        "individuals": float(df["individual_id"].nunique()) if "individual_id" in df.columns else float("nan"),
        "species": float(df["species"].nunique()) if "species" in df.columns else float("nan"),
        "missing_lat": float(df["lat"].isna().mean()) if "lat" in df.columns else float("nan"),
        "missing_lon": float(df["lon"].isna().mean()) if "lon" in df.columns else float("nan"),
    }
    return summary


def climate_quality_summary(climate_df: pd.DataFrame) -> Dict[str, float]:
    df = climate_df.copy()
    summary = {
        "rows": float(len(df)),
        "missing_temp": float(df["temp_c"].isna().mean()) if "temp_c" in df.columns else float("nan"),
        "missing_humidity": float(df["humidity"].isna().mean()) if "humidity" in df.columns else float("nan"),
        "missing_precip": float(df["precip_mm"].isna().mean()) if "precip_mm" in df.columns else float("nan"),
        "temp_min": float(df["temp_c"].min()) if "temp_c" in df.columns else float("nan"),
        "temp_max": float(df["temp_c"].max()) if "temp_c" in df.columns else float("nan"),
    }
    return summary


def assert_quality(
    gps_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    max_missing_rate: float = 0.1,
) -> None:
    gps_summary = gps_quality_summary(gps_df)
    climate_summary = climate_quality_summary(climate_df)

    if gps_summary.get("missing_lat", 0) > max_missing_rate:
        raise ValueError("GPS latitude missing rate exceeds limit")
    if gps_summary.get("missing_lon", 0) > max_missing_rate:
        raise ValueError("GPS longitude missing rate exceeds limit")
    if climate_summary.get("missing_temp", 0) > max_missing_rate:
        raise ValueError("Climate temperature missing rate exceeds limit")
