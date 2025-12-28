from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .case_studies import build_case_studies
from .clustering import cluster_refugia
from .config import PipelineConfig
from .data_sources.era5 import era5_to_dataframe
from .data_sources.movebank import load_movebank_csv
from .experiments import (
    climate_scenario_shift,
    heatwave_response_analysis,
    model_comparison_empirical_vs_climate,
    sensitivity_analysis,
)
from .heat_events import detect_heat_events
from .metadata import build_run_metadata, write_run_metadata
from .modeling import FeatureSpec, build_features, predict_future_refugia, save_model, train_model
from .preprocessing import align_gps_climate, clean_climate, clean_gps
from .quality_checks import assert_quality, climate_quality_summary, gps_quality_summary
from .reporting import build_report
from .validation import bootstrap_uncertainty, cross_validate_model, refugia_vs_random_tests, spatial_consistency


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".nc", ".netcdf"}:
        return era5_to_dataframe(path)
    raise ValueError(f"Unsupported file type: {path}")


def run_pipeline(
    config: PipelineConfig,
    gps_path: Path,
    climate_path: Path,
    future_climate_paths: Optional[Dict[str, Path]] = None,
    probability_threshold: float = 0.7,
) -> Dict[str, Path]:
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    gps_df = load_movebank_csv(gps_path) if gps_path.suffix.lower() == ".csv" else _load_dataframe(gps_path)
    climate_df = _load_dataframe(climate_path)

    gps_df = clean_gps(gps_df)
    climate_df = clean_climate(climate_df)

    assert_quality(gps_df, climate_df)

    aligned_df = align_gps_climate(gps_df, climate_df, time_tolerance_minutes=config.time_tolerance_minutes)
    aligned_path = config.outputs_dir / "aligned_data.parquet"
    aligned_df.to_parquet(aligned_path, index=False)

    thresholds = config.load_species_thresholds()
    thresholds_path = config.outputs_dir / "species_thresholds_used.csv"
    if not thresholds:
        thresholds = (
            aligned_df.groupby("species")["temp_c"]
            .quantile(config.auto_threshold_quantile)
            .to_dict()
        )
    if thresholds:
        pd.DataFrame(
            [{"species": species, "heat_threshold_c": value} for species, value in thresholds.items()]
        ).to_csv(thresholds_path, index=False)
    heat_df, events_df = detect_heat_events(
        aligned_df,
        thresholds=thresholds,
        default_threshold=config.heat_threshold_default_c,
        heat_window_hours=config.heat_window_hours,
    )
    heat_path = config.outputs_dir / "heat_events.parquet"
    heat_df.to_parquet(heat_path, index=False)
    events_path = config.outputs_dir / "heat_event_summary.parquet"
    events_df.to_parquet(events_path, index=False)

    clustered_df, clusters_df = cluster_refugia(
        heat_df,
        eps_km=config.clustering_eps_km,
        min_samples=config.clustering_min_samples,
    )
    clustered_path = config.outputs_dir / "heat_events_with_clusters.parquet"
    clustered_df.to_parquet(clustered_path, index=False)
    clusters_path = config.outputs_dir / "refugia_clusters.parquet"
    clusters_df.to_parquet(clusters_path, index=False)

    heat_points = clustered_df[clustered_df["heat_event_id"].notna()].copy()
    if heat_points.empty:
        raise ValueError("No heat events available for model training")
    if clusters_df.empty or clusters_df["is_refugia"].sum() == 0:
        raise ValueError("No refugia clusters available for model training")
    model, spec, labeled_df = train_model(
        heat_points,
        clusters_df[clusters_df["is_refugia"]],
        thresholds,
        random_state=config.model_random_state,
        n_estimators=config.model_n_estimators,
        max_depth=config.model_max_depth,
    )
    labeled_path = config.outputs_dir / "labeled_heat_points.parquet"
    labeled_df.to_parquet(labeled_path, index=False)

    model_path = config.outputs_dir / "model.pkl"
    save_model(model, spec, model_path)

    def model_builder() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=config.model_n_estimators,
            max_depth=config.model_max_depth,
            random_state=config.model_random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    validation_metrics = cross_validate_model(labeled_df, thresholds, model_builder)
    stats_tests = refugia_vs_random_tests(labeled_df)
    spatial_metrics = spatial_consistency(clusters_df, clustered_df)

    validation_path = config.outputs_dir / "validation_metrics.json"
    validation_path.write_text(json.dumps({
        "cross_validation": validation_metrics,
        "stats_tests": stats_tests,
        "spatial_metrics": spatial_metrics,
        "gps_quality": gps_quality_summary(gps_df),
        "climate_quality": climate_quality_summary(climate_df),
    }, indent=2, default=str))

    future_outputs = {}
    scenario_predictions: Dict[str, pd.DataFrame] = {}
    if future_climate_paths:
        species_list = sorted(aligned_df["species"].unique())
        for scenario, path in future_climate_paths.items():
            future_df = _load_dataframe(path)
            future_df = clean_climate(future_df)
            future_pred = predict_future_refugia(
                future_df,
                model,
                spec,
                thresholds,
                species_list,
                probability_threshold=probability_threshold,
            )
            future_path = config.outputs_dir / f"future_refugia_{scenario}.parquet"
            future_pred.to_parquet(future_path, index=False)
            future_outputs[f"future_refugia_{scenario}"] = future_path
            scenario_predictions[scenario] = future_pred

    uncertainty_df = bootstrap_uncertainty(labeled_df, climate_df, thresholds, model_builder, spec)
    uncertainty_path = config.outputs_dir / "uncertainty.parquet"
    if not uncertainty_df.empty:
        uncertainty_df.to_parquet(uncertainty_path, index=False)

    experiment_outputs = {}
    years = sorted(aligned_df["timestamp"].dt.year.unique())
    if years:
        heatwave_df = heatwave_response_analysis(heat_df, years=years, eps_km=config.clustering_eps_km, min_samples=config.clustering_min_samples)
        heatwave_path = config.outputs_dir / "experiment_heatwave_response.parquet"
        heatwave_df.to_parquet(heatwave_path, index=False)
        experiment_outputs["heatwave_response"] = heatwave_path

    if len(scenario_predictions) >= 2:
        scenarios = sorted(scenario_predictions.keys())
        for idx in range(len(scenarios) - 1):
            scenario_a = scenarios[idx]
            scenario_b = scenarios[idx + 1]
            shift_df = climate_scenario_shift(
                scenario_predictions[scenario_a],
                scenario_predictions[scenario_b],
                probability_threshold=probability_threshold,
            )
            shift_path = config.outputs_dir / f"experiment_climate_shift_{scenario_a}_vs_{scenario_b}.parquet"
            shift_df.to_parquet(shift_path, index=False)
            experiment_outputs[f"climate_shift_{scenario_a}_vs_{scenario_b}"] = shift_path

    model_comp = model_comparison_empirical_vs_climate(clustered_df)
    model_comp_path = config.outputs_dir / "experiment_model_comparison.json"
    model_comp_path.write_text(json.dumps(model_comp, indent=2))
    experiment_outputs["model_comparison"] = model_comp_path

    sensitivity_df = sensitivity_analysis(
        aligned_df,
        thresholds,
        default_threshold=config.heat_threshold_default_c,
        deltas=[-2.0, -1.0, 0.0, 1.0, 2.0],
        eps_km=config.clustering_eps_km,
        min_samples=config.clustering_min_samples,
        heat_window_hours=config.heat_window_hours,
    )
    sensitivity_path = config.outputs_dir / "experiment_sensitivity.parquet"
    sensitivity_df.to_parquet(sensitivity_path, index=False)
    experiment_outputs["sensitivity"] = sensitivity_path

    case_study_path = config.outputs_dir / "case_studies.md"
    build_case_studies(clustered_df, events_df, clusters_df, case_study_path)

    report_path = config.outputs_dir / "report.md"
    build_report(
        report_path,
        aligned_df,
        events_df,
        clusters_df,
        validation_metrics,
        stats_tests,
        spatial_metrics,
        experiment_outputs,
        uncertainty_df=uncertainty_df,
        case_studies_path=case_study_path,
    )

    metadata_path = config.outputs_dir / "run_metadata.json"
    metadata = build_run_metadata(config, gps_path, climate_path, future_climate_paths=future_climate_paths)
    write_run_metadata(metadata_path, metadata)

    outputs = {
        "aligned_data": aligned_path,
        "heat_events": heat_path,
        "heat_event_summary": events_path,
        "heat_events_with_clusters": clustered_path,
        "refugia_clusters": clusters_path,
        "labeled_heat_points": labeled_path,
        "model": model_path,
        "validation_metrics": validation_path,
        "report": report_path,
        "run_metadata": metadata_path,
        "case_studies": case_study_path,
    }
    if thresholds_path.exists():
        outputs["species_thresholds_used"] = thresholds_path
    outputs.update(future_outputs)
    if not uncertainty_df.empty:
        outputs["uncertainty"] = uncertainty_path
    outputs.update(experiment_outputs)
    return outputs
