#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from climate_refugia.config import PipelineConfig
from climate_refugia.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the climate refugia pipeline")
    parser.add_argument("--gps-path", type=Path, required=True)
    parser.add_argument("--climate-path", type=Path, required=True)
    parser.add_argument("--future-climate", type=str, nargs="*", default=None, help="scenario=path pairs")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--heat-threshold-default-c", type=float, default=35.0)
    parser.add_argument("--heat-window-hours", type=int, default=3)
    parser.add_argument("--clustering-eps-km", type=float, default=2.0)
    parser.add_argument("--clustering-min-samples", type=int, default=5)
    parser.add_argument("--model-n-estimators", type=int, default=300)
    parser.add_argument("--model-max-depth", type=int, default=12)
    parser.add_argument("--time-tolerance-minutes", type=int, default=60)
    parser.add_argument("--probability-threshold", type=float, default=0.7)
    return parser.parse_args()


def parse_future(arg_list: list[str] | None) -> dict[str, Path] | None:
    if not arg_list:
        return None
    future = {}
    for entry in arg_list:
        if "=" not in entry:
            raise SystemExit("Future climate entries must be scenario=path")
        scenario, path = entry.split("=", 1)
        future[scenario] = Path(path)
    return future


def main() -> None:
    args = parse_args()
    config = PipelineConfig.default()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.outputs_dir:
        config.outputs_dir = args.outputs_dir
    config.heat_threshold_default_c = args.heat_threshold_default_c
    config.heat_window_hours = args.heat_window_hours
    config.clustering_eps_km = args.clustering_eps_km
    config.clustering_min_samples = args.clustering_min_samples
    config.model_n_estimators = args.model_n_estimators
    config.model_max_depth = args.model_max_depth
    config.time_tolerance_minutes = args.time_tolerance_minutes

    future_paths = parse_future(args.future_climate)
    outputs = run_pipeline(
        config,
        gps_path=args.gps_path,
        climate_path=args.climate_path,
        future_climate_paths=future_paths,
        probability_threshold=args.probability_threshold,
    )

    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
