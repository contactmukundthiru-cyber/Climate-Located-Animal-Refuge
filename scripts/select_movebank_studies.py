#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import List

import pandas as pd
import requests

MOVE_BANK_URL = "https://www.movebank.org/movebank/service/direct-read"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Movebank studies with accepted licenses")
    parser.add_argument("--studies-csv", type=Path, default=Path("outputs/movebank_studies_accessible.csv"))
    parser.add_argument("--max-studies", type=int, default=5)
    parser.add_argument("--min-individuals", type=int, default=5)
    parser.add_argument("--lat-min", type=float, default=-30.0)
    parser.add_argument("--lat-max", type=float, default=30.0)
    parser.add_argument("--lon-min", type=float, default=-20.0)
    parser.add_argument("--lon-max", type=float, default=50.0)
    parser.add_argument("--require-gps", action="store_true")
    parser.add_argument("--keywords", type=str, nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("outputs/selected_movebank_studies.csv"))
    return parser.parse_args()


def fetch_studies(username: str, password: str) -> pd.DataFrame:
    response = requests.get(
        MOVE_BANK_URL,
        params={"entity_type": "study", "format": "csv"},
        auth=(username, password),
        timeout=120,
    )
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def is_license_accepted(study_id: int, username: str, password: str, timeout: int) -> bool:
    params = {
        "entity_type": "event",
        "study_id": str(study_id),
        "format": "csv",
        "max_records": "1",
    }
    try:
        response = requests.get(
            MOVE_BANK_URL,
            params=params,
            auth=(username, password),
            timeout=(min(timeout, 3), min(timeout, 3)),
            stream=True,
        )
    except requests.RequestException:
        return False
    if response.status_code != 200:
        response.close()
        return False
    head = response.raw.read(200).decode("utf-8", errors="ignore").lower()
    response.close()
    if not head:
        return False
    first_line = head.splitlines()[0] if head.splitlines() else head
    if first_line.startswith("<html"):
        return False
    return "timestamp" in first_line and "location_lat" in first_line


def main() -> None:
    args = parse_args()
    username = os.getenv("MOVEBANK_USERNAME")
    password = os.getenv("MOVEBANK_PASSWORD")
    if not username or not password:
        raise SystemExit("MOVEBANK_USERNAME and MOVEBANK_PASSWORD must be set")

    if args.studies_csv.exists():
        studies_df = pd.read_csv(args.studies_csv)
    else:
        studies_df = fetch_studies(username, password)

    for col in ["number_of_individuals", "main_location_lat", "main_location_long"]:
        if col in studies_df.columns:
            studies_df[col] = pd.to_numeric(studies_df[col], errors="coerce")

    filtered = studies_df.copy()
    filtered = filtered[filtered["main_location_lat"].between(args.lat_min, args.lat_max, inclusive="both")]
    filtered = filtered[filtered["main_location_long"].between(args.lon_min, args.lon_max, inclusive="both")]
    if args.require_gps and "sensor_type_ids" in filtered.columns:
        filtered = filtered[filtered["sensor_type_ids"].astype(str).str.contains("GPS", case=False, na=False)]
    if args.min_individuals and "number_of_individuals" in filtered.columns:
        filtered = filtered[filtered["number_of_individuals"] >= args.min_individuals]
    if args.keywords:
        keyword_mask = pd.Series(False, index=filtered.index)
        name_lower = filtered["name"].astype(str).str.lower()
        taxon_lower = filtered.get("taxon_ids", "").astype(str).str.lower()
        for keyword in args.keywords:
            kw = keyword.lower()
            keyword_mask |= name_lower.str.contains(kw) | taxon_lower.str.contains(kw)
        filtered = filtered[keyword_mask]

    filtered = filtered.sort_values("number_of_individuals", ascending=False)
    filtered = filtered.head(args.max_candidates)

    selected_rows: List[pd.Series] = []
    for _, row in filtered.iterrows():
        study_id = int(row["id"])
        if is_license_accepted(study_id, username, password, args.timeout):
            selected_rows.append(row)
        if len(selected_rows) >= args.max_studies:
            break

    if not selected_rows:
        raise SystemExit("No studies found with accepted licenses in the filtered set")

    selected_df = pd.DataFrame(selected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(args.output, index=False)

    print(f"Selected {len(selected_df)} studies:")
    for _, row in selected_df.iterrows():
        print(f"- {int(row['id'])}: {row['name']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
