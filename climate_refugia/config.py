from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class PipelineConfig:
    data_dir: Path
    outputs_dir: Path
    heat_threshold_default_c: float = 35.0
    heat_window_hours: int = 3
    clustering_eps_km: float = 2.0
    clustering_min_samples: int = 5
    model_random_state: int = 42
    model_n_estimators: int = 300
    model_max_depth: int = 12
    time_tolerance_minutes: int = 60
    auto_threshold_quantile: float = 0.9

    @staticmethod
    def default() -> "PipelineConfig":
        base = Path(__file__).resolve().parent
        return PipelineConfig(
            data_dir=base / "data",
            outputs_dir=base.parent / "outputs",
        )

    def load_species_thresholds(self) -> Dict[str, float]:
        thresholds_path = self.data_dir / "species_thresholds.csv"
        if not thresholds_path.exists():
            return {}
        thresholds: Dict[str, float] = {}
        for line in thresholds_path.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            species, value = line.split(",")
            thresholds[species.strip()] = float(value.strip())
        return thresholds
