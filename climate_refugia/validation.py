from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .modeling import FeatureSpec, build_features
from .utils import haversine_km


def cross_validate_model(
    labeled_df: pd.DataFrame,
    thresholds: Dict[str, float],
    model_builder,
    n_splits: int = 5,
) -> Dict[str, float]:
    X, spec = build_features(labeled_df, thresholds)
    y = labeled_df["is_refugia_point"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Both classes are required for cross-validation")

    metrics = {"roc_auc": [], "average_precision": [], "f1": [], "precision": [], "recall": []}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        model = model_builder()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        preds = probs >= 0.5
        metrics["roc_auc"].append(roc_auc_score(y.iloc[test_idx], probs))
        metrics["average_precision"].append(average_precision_score(y.iloc[test_idx], probs))
        metrics["f1"].append(f1_score(y.iloc[test_idx], preds, zero_division=0))
        metrics["precision"].append(precision_score(y.iloc[test_idx], preds, zero_division=0))
        metrics["recall"].append(recall_score(y.iloc[test_idx], preds, zero_division=0))

    return {key: float(np.mean(values)) for key, values in metrics.items()}


def refugia_vs_random_tests(labeled_df: pd.DataFrame) -> Dict[str, float]:
    refugia = labeled_df[labeled_df["is_refugia_point"]]
    non_refugia = labeled_df[~labeled_df["is_refugia_point"]]
    if refugia.empty or non_refugia.empty:
        raise ValueError("Need both refugia and non-refugia points for tests")

    t_stat, t_p = stats.ttest_ind(refugia["temp_c"], non_refugia["temp_c"], equal_var=False)

    species_groups = [group["temp_c"].values for _, group in labeled_df.groupby("species")]
    if len(species_groups) > 1:
        f_stat, f_p = stats.f_oneway(*species_groups)
    else:
        f_stat, f_p = np.nan, np.nan

    return {
        "temp_t_stat": float(t_stat),
        "temp_t_p": float(t_p),
        "anova_f_stat": float(f_stat) if np.isfinite(f_stat) else float("nan"),
        "anova_f_p": float(f_p) if np.isfinite(f_p) else float("nan"),
    }


def spatial_consistency(clusters_df: pd.DataFrame, events_df: pd.DataFrame) -> Dict[str, float]:
    if clusters_df.empty:
        return {"mean_centroid_shift_km": float("nan"), "median_centroid_shift_km": float("nan")}

    events = events_df.dropna(subset=["cluster_id"]).copy()
    if events.empty:
        return {"mean_centroid_shift_km": float("nan"), "median_centroid_shift_km": float("nan")}

    events["year"] = events["timestamp"].dt.year
    distances = []
    for cluster_id, group in events.groupby("cluster_id"):
        if group["year"].nunique() < 2:
            continue
        centroids = group.groupby("year").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
        for idx in range(1, len(centroids)):
            prev = centroids.iloc[idx - 1]
            curr = centroids.iloc[idx]
            distances.append(haversine_km(prev["lat"], prev["lon"], curr["lat"], curr["lon"]))

    if not distances:
        return {"mean_centroid_shift_km": float("nan"), "median_centroid_shift_km": float("nan")}

    return {
        "mean_centroid_shift_km": float(np.mean(distances)),
        "median_centroid_shift_km": float(np.median(distances)),
    }


def bootstrap_uncertainty(
    labeled_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    thresholds: Dict[str, float],
    model_builder,
    spec: FeatureSpec,
    n_bootstrap: int = 30,
) -> pd.DataFrame:
    if climate_df.empty:
        return pd.DataFrame()

    climate_sample = climate_df.sample(min(len(climate_df), 2000), random_state=42)
    climate_sample = climate_sample.copy()
    default_species = labeled_df["species"].mode().iloc[0]
    if "species" not in climate_sample.columns:
        climate_sample["species"] = default_species
    else:
        climate_sample["species"] = climate_sample["species"].fillna(default_species)
    X_pred, _ = build_features(climate_sample, thresholds, species_levels=spec.species_levels)
    X_pred = X_pred.reindex(columns=spec.columns, fill_value=0)

    preds = []
    y = labeled_df["is_refugia_point"].astype(int)
    X, _ = build_features(labeled_df, thresholds, species_levels=spec.species_levels)
    X = X.reindex(columns=spec.columns, fill_value=0)

    for idx in range(n_bootstrap):
        sample_idx = np.random.choice(len(X), size=len(X), replace=True)
        model = model_builder()
        model.fit(X.iloc[sample_idx], y.iloc[sample_idx])
        preds.append(model.predict_proba(X_pred)[:, 1])

    pred_array = np.vstack(preds)
    climate_sample["prediction_mean"] = pred_array.mean(axis=0)
    climate_sample["prediction_std"] = pred_array.std(axis=0)
    return climate_sample
