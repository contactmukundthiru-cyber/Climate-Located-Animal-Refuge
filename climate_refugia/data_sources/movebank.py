from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

MOVE_BANK_URL = "https://www.movebank.org/movebank/service/direct-read"

DEFAULT_ATTRIBUTES = [
    "timestamp",
    "location_lat",
    "location_long",
    "individual_id",
    "individual_local_identifier",
    "taxon_canonical_name",
    "sensor_type_id",
]


class MovebankError(RuntimeError):
    """Movebank API error."""


def download_movebank_events(
    output_path: Path,
    study_id: int,
    username: str,
    password: str,
    attributes: Optional[Iterable[str]] = None,
    sensor_type_id: Optional[str] = None,
    individual_ids: Optional[Iterable[int]] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    extra_params: Optional[Dict[str, str]] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params: Dict[str, str] = {
        "entity_type": "event",
        "study_id": str(study_id),
        "format": "csv",
    }
    attrs = list(attributes) if attributes is not None else DEFAULT_ATTRIBUTES
    params["attributes"] = ",".join(attrs)
    if sensor_type_id:
        params["sensor_type_id"] = str(sensor_type_id)
    if individual_ids:
        params["individual_id"] = ",".join(str(item) for item in individual_ids)
    if timestamp_start:
        params["timestamp_start"] = timestamp_start
    if timestamp_end:
        params["timestamp_end"] = timestamp_end
    if extra_params:
        params.update(extra_params)

    response = requests.get(MOVE_BANK_URL, params=params, auth=(username, password), timeout=120)
    if response.status_code != 200:
        raise MovebankError(f"Movebank request failed: {response.status_code} {response.text[:200]}")

    content = response.content
    text_head = content[:200].decode("utf-8", errors="ignore").lower()
    if "<html" in text_head or "by accepting this document the user agrees" in text_head:
        raise MovebankError(
            "Movebank license terms not accepted for this study. "
            "Log in to Movebank, accept the license terms for the study, then retry."
        )

    output_path.write_bytes(content)
    return output_path


def load_movebank_csv(
    path: Path,
    require_species: bool = True,
    species_fallback: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    column_map = {
        "location_lat": "lat",
        "location_long": "lon",
        "latitude": "lat",
        "longitude": "lon",
        "timestamp": "timestamp",
        "individual_id": "individual_id",
        "individual_local_identifier": "individual_name",
        "taxon_canonical_name": "species",
        "tag_id": "tag_id",
    }

    rename = {}
    for column in df.columns:
        if column in column_map:
            rename[column] = column_map[column]
    df = df.rename(columns=rename)

    required = {"timestamp", "lat", "lon"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise MovebankError(f"Missing required columns in Movebank data: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "species" not in df.columns:
        if require_species and not species_fallback:
            raise MovebankError("Species column missing; ensure taxon_canonical_name is included in attributes.")
        df["species"] = species_fallback if species_fallback else "Unknown"
    if "individual_id" not in df.columns:
        if "individual_name" in df.columns:
            df["individual_id"] = df["individual_name"].astype(str)
        else:
            raise MovebankError("individual_id or individual_local_identifier is required in Movebank data.")
    return df
