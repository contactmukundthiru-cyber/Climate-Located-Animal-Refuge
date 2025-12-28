#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from climate_refugia.reporting import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a refugia report")
    parser.add_argument("--aligned-path", type=Path, required=True)
    parser.add_argument("--events-path", type=Path, required=True)
    parser.add_argument("--clusters-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--uncertainty-path", type=Path, default=None)
    parser.add_argument("--case-studies-path", type=Path, default=None)
    parser.add_argument("--experiment", type=str, nargs="*", default=None, help="name=path pairs")
    return parser.parse_args()


def parse_experiments(arg_list: list[str] | None) -> dict[str, Path]:
    experiments = {}
    if not arg_list:
        return experiments
    for entry in arg_list:
        if "=" not in entry:
            raise SystemExit("Experiments must be name=path")
        name, path = entry.split("=", 1)
        experiments[name] = Path(path)
    return experiments


def main() -> None:
    args = parse_args()
    aligned_df = pd.read_parquet(args.aligned_path)
    events_df = pd.read_parquet(args.events_path)
    clusters_df = pd.read_parquet(args.clusters_path)

    validation_payload = json.loads(args.validation_path.read_text())
    validation_metrics = validation_payload.get("cross_validation", {})
    stats_tests = validation_payload.get("stats_tests", {})
    spatial_metrics = validation_payload.get("spatial_metrics", {})

    uncertainty_df = None
    if args.uncertainty_path:
        uncertainty_df = pd.read_parquet(args.uncertainty_path)

    experiments = parse_experiments(args.experiment)

    build_report(
        args.report_path,
        aligned_df,
        events_df,
        clusters_df,
        validation_metrics,
        stats_tests,
        spatial_metrics,
        experiments,
        uncertainty_df=uncertainty_df,
        case_studies_path=args.case_studies_path,
    )
    print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    main()
