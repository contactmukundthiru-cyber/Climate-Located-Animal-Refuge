#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from climate_refugia.data_sources.era5 import era5_to_dataframe
from climate_refugia.data_sources.movebank import load_movebank_csv
from climate_refugia.preprocessing import clean_climate, clean_gps
from climate_refugia.quality_checks import assert_quality, climate_quality_summary, gps_quality_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPS and climate inputs")
    parser.add_argument("--gps-path", type=Path, required=True)
    parser.add_argument("--climate-path", type=Path, required=True)
    return parser.parse_args()


def load_climate(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".nc", ".netcdf"}:
        return era5_to_dataframe(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    gps_df = load_movebank_csv(args.gps_path) if args.gps_path.suffix.lower() == ".csv" else pd.read_parquet(args.gps_path)
    climate_df = load_climate(args.climate_path)

    gps_df = clean_gps(gps_df)
    climate_df = clean_climate(climate_df)

    assert_quality(gps_df, climate_df)
    print("GPS quality summary:")
    print(gps_quality_summary(gps_df))
    print("Climate quality summary:")
    print(climate_quality_summary(climate_df))


if __name__ == "__main__":
    main()
