from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import importlib.metadata

from .config import PipelineConfig


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(repo_dir: Path) -> Dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        return {
            "commit": commit,
            "dirty": "yes" if status else "no",
        }
    except Exception:
        return {"commit": "unknown", "dirty": "unknown"}


def _package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def build_run_metadata(
    config: PipelineConfig,
    gps_path: Path,
    climate_path: Path,
    future_climate_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, object]:
    package_versions = _package_versions(
        [
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "statsmodels",
            "xarray",
            "netCDF4",
            "cdsapi",
            "plotly",
            "folium",
            "streamlit",
        ]
    )

    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)

    metadata: Dict[str, object] = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "config": config_dict,
        "inputs": {
            "gps_path": str(gps_path),
            "gps_sha256": _sha256(gps_path),
            "gps_size_bytes": gps_path.stat().st_size,
            "climate_path": str(climate_path),
            "climate_sha256": _sha256(climate_path),
            "climate_size_bytes": climate_path.stat().st_size,
        },
        "package_versions": package_versions,
        "git": _git_info(Path(__file__).resolve().parents[1]),
    }

    if future_climate_paths:
        future_meta = {}
        for scenario, path in future_climate_paths.items():
            future_meta[scenario] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        metadata["inputs"]["future_climate"] = future_meta

    return metadata


def write_run_metadata(path: Path, metadata: Dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
    return path
