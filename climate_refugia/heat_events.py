from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_datetime


def detect_heat_events(
    aligned_df: pd.DataFrame,
    thresholds: Dict[str, float],
    default_threshold: float,
    heat_window_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = aligned_df.copy()
    df["timestamp"] = ensure_datetime(df["timestamp"])
    df["heat_threshold_c"] = df["species"].map(thresholds).fillna(default_threshold)
    df["heat_exposure"] = df["temp_c"] >= df["heat_threshold_c"]
    df = df.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)

    events = []
    df["heat_event_id"] = pd.NA
    event_counter = 0

    for individual_id, group in df.groupby("individual_id"):
        group = group.sort_values("timestamp")
        median_dt = group["timestamp"].diff().median()
        median_seconds = median_dt.total_seconds() if pd.notnull(median_dt) else heat_window_hours * 3600
        if median_seconds <= 0:
            median_seconds = heat_window_hours * 3600
        min_points = max(1, int(math.ceil(heat_window_hours * 3600 / median_seconds)))

        heat_block = (group["heat_exposure"] != group["heat_exposure"].shift()).cumsum()
        for _, block in group.groupby(heat_block):
            if not bool(block["heat_exposure"].iloc[0]):
                continue
            if len(block) < min_points:
                continue
            event_counter += 1
            df.loc[block.index, "heat_event_id"] = event_counter
            duration_hours = (block["timestamp"].max() - block["timestamp"].min()).total_seconds() / 3600
            events.append(
                {
                    "heat_event_id": event_counter,
                    "individual_id": individual_id,
                    "species": block["species"].iloc[0],
                    "start_time": block["timestamp"].min(),
                    "end_time": block["timestamp"].max(),
                    "duration_hours": duration_hours,
                    "num_points": len(block),
                    "mean_temp_c": block["temp_c"].mean(),
                    "max_temp_c": block["temp_c"].max(),
                }
            )

    events_df = pd.DataFrame(events)
    return df, events_df
