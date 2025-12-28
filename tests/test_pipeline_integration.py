import os
from pathlib import Path

import pandas as pd

from climate_refugia.config import PipelineConfig
from climate_refugia.pipeline import run_pipeline


def test_pipeline_end_to_end():
    gps_path = Path(os.environ["CR_GPS_PATH"])
    climate_path = Path(os.environ["CR_CLIMATE_PATH"])

    outputs_dir = Path(os.environ.get("CR_OUTPUTS_DIR", "outputs_test"))
    data_dir = Path(os.environ.get("CR_DATA_DIR", str(PipelineConfig.default().data_dir)))

    config = PipelineConfig.default()
    config.outputs_dir = outputs_dir
    config.data_dir = data_dir

    outputs = run_pipeline(config, gps_path=gps_path, climate_path=climate_path)
    aligned_path = outputs["aligned_data"]
    assert aligned_path.exists()

    aligned_df = pd.read_parquet(aligned_path)
    assert {"lat", "lon", "temp_c", "timestamp"}.issubset(set(aligned_df.columns))
