#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from climate_refugia.data_sources.movebank import download_movebank_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Movebank tracking data")
    parser.add_argument("--study-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--username", type=str, default=os.getenv("MOVEBANK_USERNAME"))
    parser.add_argument("--password", type=str, default=os.getenv("MOVEBANK_PASSWORD"))
    parser.add_argument("--attributes", type=str, nargs="*", default=None)
    parser.add_argument("--sensor-type-id", type=str, default=None)
    parser.add_argument("--individual-ids", type=int, nargs="*", default=None)
    parser.add_argument("--timestamp-start", type=str, default=None)
    parser.add_argument("--timestamp-end", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.username or not args.password:
        raise SystemExit("Movebank credentials are required via --username/--password or env vars")

    download_movebank_events(
        output_path=args.output,
        study_id=args.study_id,
        username=args.username,
        password=args.password,
        attributes=args.attributes,
        sensor_type_id=args.sensor_type_id,
        individual_ids=args.individual_ids,
        timestamp_start=args.timestamp_start,
        timestamp_end=args.timestamp_end,
    )
    print(f"Saved Movebank data to {args.output}")


if __name__ == "__main__":
    main()
