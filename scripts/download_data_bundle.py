#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

from climate_refugia.data_sources.era5 import Era5Request, download_era5, era5_to_dataframe
from climate_refugia.data_sources.movebank import MovebankError, download_movebank_events, load_movebank_csv

MOVE_BANK_URL = "https://www.movebank.org/movebank/service/direct-read"

DEFAULT_STUDY_IDS = [
    736029750,  # ThermochronTracking Elephants Kruger 2007
    605129389,  # African elephants in Etosha National Park
    1818825,    # Forest Elephant Telemetry Programme
    208413731,  # White-bearded wildebeest movements - Kenya
    4901146318, # White-bearded wildebeest - Greater Mara Ecosystem
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Movebank and ERA5 datasets for the project")
    parser.add_argument("--study-ids", type=int, nargs="*", default=DEFAULT_STUDY_IDS)
    parser.add_argument("--output-dir", type=Path, default=Path("climate_refugia/data"))
    parser.add_argument("--buffer-deg", type=float, default=1.0)
    parser.add_argument("--grid", type=float, nargs=2, default=[0.25, 0.25])
    parser.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--skip-era5", action="store_true")
    parser.add_argument("--skip-unaccepted", action="store_true", help="Skip Movebank studies requiring license acceptance")
    return parser.parse_args()


def fetch_study_metadata(username: str, password: str) -> pd.DataFrame:
    response = requests.get(
        MOVE_BANK_URL,
        params={"entity_type": "study", "format": "csv"},
        auth=(username, password),
        timeout=120,
    )
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def build_area_bounds(df: pd.DataFrame, buffer_deg: float) -> List[float]:
    north = float(df["lat"].max() + buffer_deg)
    south = float(df["lat"].min() - buffer_deg)
    west = float(df["lon"].min() - buffer_deg)
    east = float(df["lon"].max() + buffer_deg)
    return [north, west, south, east]


def main() -> None:
    username = os.getenv("MOVEBANK_USERNAME")
    password = os.getenv("MOVEBANK_PASSWORD")
    if not username or not password:
        raise SystemExit("MOVEBANK_USERNAME and MOVEBANK_PASSWORD must be set")

    args = parse_args()
    output_dir: Path = args.output_dir
    movebank_dir = output_dir / "movebank"
    era5_dir = output_dir / "era5"
    movebank_dir.mkdir(parents=True, exist_ok=True)
    era5_dir.mkdir(parents=True, exist_ok=True)

    study_metadata = fetch_study_metadata(username, password)
    metadata_subset = study_metadata[study_metadata["id"].isin(args.study_ids)].copy()
    metadata_path = output_dir.parent / "outputs" / "movebank_study_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_subset.to_csv(metadata_path, index=False)

    gps_frames = []
    climate_frames = []
    skipped = []

    for study_id in args.study_ids:
        study_row = metadata_subset[metadata_subset["id"] == study_id]
        study_name = study_row["name"].iloc[0] if not study_row.empty else f"study_{study_id}"
        taxon_ids = study_row["taxon_ids"].iloc[0] if not study_row.empty else None
        gps_path = movebank_dir / f"study_{study_id}.csv"
        if not gps_path.exists():
            try:
                download_movebank_events(
                    output_path=gps_path,
                    study_id=study_id,
                    username=username,
                    password=password,
                )
            except MovebankError:
                try:
                    download_movebank_events(
                        output_path=gps_path,
                        study_id=study_id,
                        username=username,
                        password=password,
                        attributes=[
                            "timestamp",
                            "location_lat",
                            "location_long",
                            "individual_id",
                            "tag_id",
                        ],
                    )
                except MovebankError as exc2:
                    if args.skip_unaccepted:
                        skipped.append((study_id, study_name))
                        continue
                    raise SystemExit(f"Study {study_id} download failed: {exc2}") from exc2

        gps_df = load_movebank_csv(
            gps_path,
            require_species=False,
            species_fallback=str(taxon_ids) if taxon_ids is not None and str(taxon_ids) != "nan" else None,
        )
        if args.start or args.end:
            start = pd.to_datetime(args.start, utc=True) if args.start else gps_df["timestamp"].min()
            end = pd.to_datetime(args.end, utc=True) if args.end else gps_df["timestamp"].max()
            gps_df = gps_df[(gps_df["timestamp"] >= start) & (gps_df["timestamp"] <= end)]
            if gps_df.empty:
                skipped.append((study_id, f"{study_name} (no data in range)"))
                continue
        gps_df["study_id"] = study_id
        gps_df["study_name"] = study_name
        gps_frames.append(gps_df)

        if args.skip_era5:
            continue

        area = build_area_bounds(gps_df, args.buffer_deg)
        start_time = gps_df["timestamp"].min().strftime("%Y-%m-%d")
        end_time = gps_df["timestamp"].max().strftime("%Y-%m-%d")
        era5_path = era5_dir / f"era5_study_{study_id}.nc"

        if not era5_path.exists():
            request = Era5Request(
                start=pd.to_datetime(start_time, utc=True).to_pydatetime(),
                end=pd.to_datetime(end_time, utc=True).to_pydatetime(),
                area=area,
                variables=["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"],
                grid=args.grid,
                output_path=era5_path,
            )
            download_era5(request)

        climate_df = era5_to_dataframe(era5_path)
        climate_df["study_id"] = study_id
        climate_frames.append(climate_df)

    if not gps_frames:
        raise SystemExit("No Movebank datasets were downloaded. Check license acceptance and study IDs.")

    combined_gps = pd.concat(gps_frames, ignore_index=True)
    combined_path = output_dir / "movebank_events.parquet"
    combined_gps.to_parquet(combined_path, index=False)

    if climate_frames:
        combined_climate = pd.concat(climate_frames, ignore_index=True)
        climate_path = output_dir / "era5_combined.parquet"
        combined_climate.to_parquet(climate_path, index=False)

    data_avail_path = output_dir.parent / "outputs" / "data_availability.md"
    lines = ["# Data Availability", "", "## Movebank Studies"]
    for _, row in metadata_subset.iterrows():
        lines.append(f"- {row['name']} (Movebank study ID {row['id']}, license {row.get('license_type', 'NA')})")
        citation = str(row.get("citation", "")).strip()
        if citation and citation != "nan":
            lines.append(f"  - Citation: {citation}")
    lines.append("")
    lines.append("## ERA5")
    lines.append("ERA5 reanalysis data were obtained from the Copernicus Climate Data Store.")
    data_avail_path.write_text("\n".join(lines))

    print(f"Movebank combined data: {combined_path}")
    if climate_frames:
        print(f"ERA5 combined data: {output_dir / 'era5_combined.parquet'}")
    print(f"Study metadata: {metadata_path}")
    print(f"Data availability: {data_avail_path}")
    if skipped:
        print("Skipped studies (license not accepted):")
        for study_id, study_name in skipped:
            print(f"- {study_id}: {study_name}")


if __name__ == "__main__":
    main()
