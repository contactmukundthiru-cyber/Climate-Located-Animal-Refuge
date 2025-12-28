from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import tempfile

try:
    import cdsapi
except ImportError:  # pragma: no cover
    cdsapi = None


@dataclass
class Era5Request:
    start: datetime
    end: datetime
    area: Sequence[float]
    variables: Sequence[str]
    grid: Sequence[float]
    output_path: Path


def download_era5(request: Era5Request) -> Path:
    if cdsapi is None:
        raise RuntimeError("cdsapi is required for ERA5 downloads")

    client = cdsapi.Client()
    years = sorted({str(ts.year) for ts in pd.date_range(request.start, request.end, freq="D")})
    months = sorted({f"{ts.month:02d}" for ts in pd.date_range(request.start, request.end, freq="D")})
    days = sorted({f"{ts.day:02d}" for ts in pd.date_range(request.start, request.end, freq="D")})
    hours = [f"{hour:02d}:00" for hour in range(24)]

    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": list(request.variables),
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "area": list(request.area),
            "grid": list(request.grid),
            "format": "netcdf",
        },
        str(request.output_path),
    )
    return request.output_path


def relative_humidity_from_dewpoint(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15
    a = 17.625
    b = 243.04
    svp = 6.1094 * np.exp((a * temp_c) / (temp_c + b))
    avp = 6.1094 * np.exp((a * dew_c) / (dew_c + b))
    rh = 100.0 * (avp / svp)
    return np.clip(rh, 0.0, 100.0)


def _resolve_var(ds: xr.Dataset, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in ds:
            return name
    return None


def _is_zip(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"PK"
    except OSError:
        return False


def _open_era5_dataset(path: Path) -> xr.Dataset:
    if _is_zip(path):
        with zipfile.ZipFile(path) as zf:
            nc_files = [name for name in zf.namelist() if name.endswith(".nc")]
            if not nc_files:
                raise ValueError("ERA5 zip archive contains no NetCDF files")
            with tempfile.TemporaryDirectory() as tmpdir:
                extracted = []
                for name in nc_files:
                    extracted_path = Path(tmpdir) / Path(name).name
                    with zf.open(name) as src, extracted_path.open("wb") as dst:
                        dst.write(src.read())
                    extracted.append(extracted_path)
                datasets = []
                for item in extracted:
                    ds = xr.open_dataset(item)
                    datasets.append(ds.load())
                combined = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
                for ds in datasets:
                    ds.close()
                return combined
    return xr.open_dataset(path)


def era5_to_dataframe(path: Path) -> pd.DataFrame:
    ds = _open_era5_dataset(path)

    temp_key = _resolve_var(ds, ["t2m", "2m_temperature", "temperature_2m"])
    dew_key = _resolve_var(ds, ["d2m", "2m_dewpoint_temperature", "dewpoint_2m"])
    precip_key = _resolve_var(ds, ["tp", "total_precipitation"])

    if temp_key is None:
        raise ValueError("ERA5 dataset missing 2m temperature")

    df = ds.to_dataframe().reset_index()

    time_col = "time"
    if "time" not in df.columns:
        if "valid_time" in df.columns:
            time_col = "valid_time"
        else:
            raise ValueError("ERA5 dataset missing time coordinate")

    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        time_col: "timestamp",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["temp_c"] = df[temp_key] - 273.15

    if dew_key is not None:
        df["humidity"] = relative_humidity_from_dewpoint(df[temp_key].to_numpy(), df[dew_key].to_numpy())
    else:
        df["humidity"] = np.nan

    if precip_key is not None:
        df["precip_mm"] = df[precip_key] * 1000.0
    else:
        df["precip_mm"] = np.nan

    keep = ["timestamp", "lat", "lon", "temp_c", "humidity", "precip_mm"]
    return df[keep].dropna(subset=["lat", "lon", "temp_c"]).reset_index(drop=True)
