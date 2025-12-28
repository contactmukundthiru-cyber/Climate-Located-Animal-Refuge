from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import BallTree

from .utils import EARTH_RADIUS_KM, ensure_datetime


@dataclass
class FeatureSpec:
    columns: List[str]
    species_levels: List[str]


def build_features(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    species_levels: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, FeatureSpec]:
    data = df.copy()
    data["timestamp"] = ensure_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["dayofyear"] = data["timestamp"].dt.dayofyear
    data["heat_threshold_c"] = data["species"].map(thresholds).fillna(np.nan)

    base_features = [
        "lat",
        "lon",
        "temp_c",
        "humidity",
        "precip_mm",
        "hour",
        "dayofyear",
        "heat_threshold_c",
    ]
    for feature in base_features:
        if feature not in data.columns:
            data[feature] = np.nan

    species_levels = list(species_levels) if species_levels is not None else sorted(data["species"].unique())
    species_cat = pd.Categorical(data["species"], categories=species_levels)
    species_series = pd.Series(species_cat, index=data.index)
    species_dummies = pd.get_dummies(species_series, prefix="species")

    feature_df = pd.concat([data[base_features], species_dummies], axis=1)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.fillna(0)

    spec = FeatureSpec(columns=list(feature_df.columns), species_levels=species_levels)
    return feature_df, spec


def label_refugia_points(
    df: pd.DataFrame,
    refugia_df: pd.DataFrame,
    radius_km: float,
) -> pd.DataFrame:
    if refugia_df.empty:
        raise ValueError("Refugia clusters are required to label points")

    labeled = df.copy()
    coords = np.radians(refugia_df[["centroid_lat", "centroid_lon"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    gps_coords = np.radians(labeled[["lat", "lon"]].to_numpy())
    dist, _ = tree.query(gps_coords, k=1)
    dist_km = dist.flatten() * EARTH_RADIUS_KM
    labeled["refugia_distance_km"] = dist_km
    labeled["is_refugia_point"] = dist_km <= radius_km
    return labeled


def train_model(
    heat_df: pd.DataFrame,
    refugia_df: pd.DataFrame,
    thresholds: Dict[str, float],
    random_state: int,
    n_estimators: int,
    max_depth: int,
    radius_km: float = 3.0,
) -> Tuple[RandomForestClassifier, FeatureSpec, pd.DataFrame]:
    labeled = label_refugia_points(heat_df, refugia_df, radius_km)
    X, spec = build_features(labeled, thresholds)
    y = labeled["is_refugia_point"].astype(int)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, spec, labeled


def save_model(model: RandomForestClassifier, spec: FeatureSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": model, "spec": spec}, f)


def load_model(path: Path) -> Tuple[RandomForestClassifier, FeatureSpec]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["spec"]


def predict_future_refugia(
    climate_df: pd.DataFrame,
    model: RandomForestClassifier,
    spec: FeatureSpec,
    thresholds: Dict[str, float],
    species_list: Iterable[str],
    probability_threshold: float = 0.7,
) -> pd.DataFrame:
    predictions = []

    for species in species_list:
        threshold = thresholds.get(species)
        subset = climate_df.copy()
        subset["species"] = species
        if threshold is not None:
            subset = subset[subset["temp_c"] >= threshold]
        if subset.empty:
            continue
        X, _ = build_features(subset, thresholds, species_levels=spec.species_levels)
        X = X.reindex(columns=spec.columns, fill_value=0)
        prob = model.predict_proba(X)[:, 1]
        output = subset[["timestamp", "lat", "lon", "temp_c", "humidity", "precip_mm"]].copy()
        output["species"] = species
        output["refugia_probability"] = prob
        output["is_refugia_pred"] = prob >= probability_threshold
        predictions.append(output)

    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)
