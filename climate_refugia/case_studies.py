from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def build_case_studies(
    heat_df: pd.DataFrame,
    events_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 3,
) -> Path:
    if heat_df.empty or events_df.empty:
        output_path.write_text("No heat events available for case studies.\n")
        return output_path

    heat_events = heat_df.dropna(subset=["heat_event_id"]).copy()
    event_summary = events_df.copy()
    event_summary = event_summary.sort_values(["species", "max_temp_c"], ascending=[True, False])
    top_events = event_summary.groupby("species").head(top_n)

    cluster_lookup = clusters_df.set_index("cluster_id")["is_refugia"].to_dict() if not clusters_df.empty else {}

    rows = []
    for _, event in top_events.iterrows():
        subset = heat_events[heat_events["heat_event_id"] == event["heat_event_id"]]
        cluster_ids = sorted(set(subset["cluster_id"].dropna().astype(int)))
        refugia_hits = sum(cluster_lookup.get(cid, False) for cid in cluster_ids)
        rows.append({
            "species": event["species"],
            "heat_event_id": int(event["heat_event_id"]),
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "duration_hours": event["duration_hours"],
            "mean_temp_c": event["mean_temp_c"],
            "max_temp_c": event["max_temp_c"],
            "num_points": event["num_points"],
            "cluster_ids": ",".join(str(cid) for cid in cluster_ids),
            "refugia_clusters_hit": refugia_hits,
        })

    case_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        case_df.to_csv(output_path, index=False)
    else:
        lines = ["# Case Studies", ""]
        for species, group in case_df.groupby("species"):
            lines.append(f"## {species}")
            for _, row in group.iterrows():
                lines.append(
                    f"- Event {row['heat_event_id']} ({row['start_time']} to {row['end_time']}) | "
                    f"max {row['max_temp_c']:.2f} C | duration {row['duration_hours']:.1f} h | "
                    f"clusters {row['cluster_ids']} | refugia hits {row['refugia_clusters_hit']}"
                )
            lines.append("")
        output_path.write_text("\n".join(lines))
    return output_path
