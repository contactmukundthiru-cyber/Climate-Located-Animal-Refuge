from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def build_report(
    report_path: Path,
    aligned_df: pd.DataFrame,
    events_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    validation_metrics: Dict[str, float],
    stats_tests: Dict[str, float],
    spatial_metrics: Dict[str, float],
    experiment_outputs: Dict[str, Path],
    uncertainty_df: Optional[pd.DataFrame] = None,
    case_studies_path: Optional[Path] = None,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Climate Refugia Report", ""]

    lines.append("## Data Summary")
    lines.append(f"- GPS points: {len(aligned_df):,}")
    lines.append(f"- Heat events: {len(events_df):,}")
    lines.append(f"- Refugia clusters: {int(clusters_df['is_refugia'].sum()) if not clusters_df.empty else 0:,}")
    lines.append("")

    lines.append("## Validation Metrics")
    for key, value in validation_metrics.items():
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")

    lines.append("## Statistical Tests")
    for key, value in stats_tests.items():
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")

    lines.append("## Spatial Consistency")
    for key, value in spatial_metrics.items():
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")

    lines.append("## Experiments")
    for name, path in experiment_outputs.items():
        lines.append(f"- {name}: {path}")
    lines.append("")

    if case_studies_path is not None:
        lines.append("## Case Studies")
        lines.append(f"- {case_studies_path}")
        lines.append("")

    if uncertainty_df is not None and not uncertainty_df.empty:
        lines.append("## Uncertainty Summary")
        lines.append(f"- Samples: {len(uncertainty_df):,}")
        lines.append(f"- Mean prediction std: {uncertainty_df['prediction_std'].mean():.4f}")
        lines.append("")

    lines.append("## Data Availability")
    lines.append("Movebank tracking data are available from the Movebank data repository under the terms of each study.")
    lines.append("ERA5 climate data are available from the Copernicus Climate Data Store.")
    lines.append("")

    lines.append("## Code Availability")
    lines.append("All analysis code used for this report is contained in this repository.")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path
