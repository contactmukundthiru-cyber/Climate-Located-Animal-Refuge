#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from climate_refugia.experiments import (
    climate_scenario_shift,
    heatwave_response_analysis,
    model_comparison_empirical_vs_climate,
    sensitivity_analysis,
)
from climate_refugia.config import PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run climate refugia experiments")
    parser.add_argument("--aligned-path", type=Path, required=True)
    parser.add_argument("--clustered-heat-path", type=Path, required=True)
    parser.add_argument("--clusters-path", type=Path, required=True)
    parser.add_argument("--future", type=str, nargs="*", default=None, help="scenario=path pairs")
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--clustering-eps-km", type=float, default=2.0)
    parser.add_argument("--clustering-min-samples", type=int, default=5)
    return parser.parse_args()


def parse_future(arg_list: list[str] | None) -> dict[str, Path] | None:
    if not arg_list:
        return None
    future = {}
    for entry in arg_list:
        if "=" not in entry:
            raise SystemExit("Future entries must be scenario=path")
        scenario, path = entry.split("=", 1)
        future[scenario] = Path(path)
    return future


def main() -> None:
    args = parse_args()
    config = PipelineConfig.default()
    if args.data_dir:
        config.data_dir = args.data_dir

    aligned_df = pd.read_parquet(args.aligned_path)
    heat_df = pd.read_parquet(args.clustered_heat_path)
    clusters_df = pd.read_parquet(args.clusters_path)

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(aligned_df["timestamp"].dt.year.unique())
    heatwave_df = heatwave_response_analysis(heat_df, years=years, eps_km=args.clustering_eps_km, min_samples=args.clustering_min_samples)
    heatwave_path = args.outputs_dir / "experiment_heatwave_response.parquet"
    heatwave_df.to_parquet(heatwave_path, index=False)

    sensitivity_df = sensitivity_analysis(
        aligned_df,
        config.load_species_thresholds(),
        default_threshold=config.heat_threshold_default_c,
        deltas=[-2.0, -1.0, 0.0, 1.0, 2.0],
        eps_km=args.clustering_eps_km,
        min_samples=args.clustering_min_samples,
    )
    sensitivity_path = args.outputs_dir / "experiment_sensitivity.parquet"
    sensitivity_df.to_parquet(sensitivity_path, index=False)

    model_comp = model_comparison_empirical_vs_climate(heat_df)
    model_comp_path = args.outputs_dir / "experiment_model_comparison.json"
    model_comp_path.write_text(pd.Series(model_comp).to_json(indent=2))

    future = parse_future(args.future)
    if future and len(future) >= 2:
        scenarios = sorted(future.keys())
        for idx in range(len(scenarios) - 1):
            scenario_a = scenarios[idx]
            scenario_b = scenarios[idx + 1]
            df_a = pd.read_parquet(future[scenario_a])
            df_b = pd.read_parquet(future[scenario_b])
            shift_df = climate_scenario_shift(df_a, df_b)
            shift_path = args.outputs_dir / f"experiment_climate_shift_{scenario_a}_vs_{scenario_b}.parquet"
            shift_df.to_parquet(shift_path, index=False)

    print(f"Outputs written to {args.outputs_dir}")


if __name__ == "__main__":
    main()
