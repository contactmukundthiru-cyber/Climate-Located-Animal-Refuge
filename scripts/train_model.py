#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from climate_refugia.config import PipelineConfig
from climate_refugia.modeling import save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refugia prediction model")
    parser.add_argument("--heat-path", type=Path, required=True)
    parser.add_argument("--clusters-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-n-estimators", type=int, default=300)
    parser.add_argument("--model-max-depth", type=int, default=12)
    parser.add_argument("--model-random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.default()
    if args.data_dir:
        config.data_dir = args.data_dir

    heat_df = pd.read_parquet(args.heat_path)
    clusters_df = pd.read_parquet(args.clusters_path)

    thresholds = config.load_species_thresholds()
    heat_points = heat_df[heat_df["heat_event_id"].notna()].copy()
    model, spec, _ = train_model(
        heat_points,
        clusters_df[clusters_df["is_refugia"]],
        thresholds,
        random_state=args.model_random_state,
        n_estimators=args.model_n_estimators,
        max_depth=args.model_max_depth,
    )
    save_model(model, spec, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
