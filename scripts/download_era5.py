#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from climate_refugia.data_sources.era5 import Era5Request, download_era5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 climate data")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--area", type=float, nargs=4, required=True, help="North West South East")
    parser.add_argument("--grid", type=float, nargs=2, default=[0.25, 0.25])
    parser.add_argument("--variables", type=str, nargs="*", default=["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.to_datetime(args.start, utc=True).to_pydatetime()
    end = pd.to_datetime(args.end, utc=True).to_pydatetime()
    request = Era5Request(
        start=start,
        end=end,
        area=args.area,
        variables=args.variables,
        grid=args.grid,
        output_path=args.output,
    )
    download_era5(request)
    print(f"Saved ERA5 data to {args.output}")


if __name__ == "__main__":
    main()
