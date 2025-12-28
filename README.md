# Climate Refugia Identification and Prediction

End-to-end system for detecting thermal refugia from animal movement data and climate reanalysis, then projecting future refugia under climate scenarios. Includes preprocessing, heat-event detection, spatiotemporal clustering, predictive modeling, statistical validation, experiments, and an interactive web application.

## Author

**Mukund Thiru**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_movebank.py --help
python3 scripts/download_era5.py --help
python3 scripts/run_pipeline.py --help
streamlit run climate_refugia/webapp/app.py
```

## Data Sources

### Movebank
The pipeline can download tracking data using the Movebank API when credentials are provided.

Required environment variables:
- `MOVEBANK_USERNAME`
- `MOVEBANK_PASSWORD`

Optional:
- `MOVEBANK_STUDY_ID` (if not provided, supply via CLI)

Note: some Movebank studies require accepting license terms in the Movebank web interface before API download.

### ERA5 (Copernicus Climate Data Store)
For ERA5 downloads, configure `~/.cdsapirc` (see CDS API docs). The downloader requests hourly 2m temperature, humidity, and precipitation.
ERA5 downloads require accepting CDS license terms for the ERA5 dataset in the CDS portal.

### Species Heat Thresholds
Provide `species_thresholds.csv` in `climate_refugia/data/` with columns `species,heat_threshold_c`. Species not listed fall back to the default heat threshold.
If no thresholds file is provided, the pipeline derives thresholds from the configured quantile of observed temperatures and writes `species_thresholds_used.csv` to `outputs/`.

## Project Structure

- `climate_refugia/` core library
- `climate_refugia/data/` project datasets
- `scripts/` CLI entry points
- `tests/` unit tests
- `docs/` methods and reproducibility
- `outputs/` pipeline outputs (created at runtime)

## Pipeline Overview

1. Download/ingest Movebank GPS data
2. Download/ingest ERA5 climate data
3. Clean and align data spatiotemporally
4. Detect heat events using species thresholds
5. Cluster refugia locations across events
6. Train predictive model and project future refugia
7. Validate with statistics and uncertainty analysis
8. Visualize in interactive web app

## Scripts

- `scripts/run_pipeline.py` end-to-end pipeline
- `scripts/download_movebank.py` Movebank download
- `scripts/download_era5.py` ERA5 download
- `scripts/download_data_bundle.py` curated Movebank + ERA5 bundle download
- `scripts/download_data_bundle.py` supports `--start`, `--end`, and `--grid` to keep ERA5 requests within CDS limits
- `scripts/select_movebank_studies.py` choose Movebank studies with accepted licenses
- `scripts/train_model.py` train and persist model
- `scripts/run_experiments.py` run experiments suite
- `scripts/build_report.py` generate a report (Markdown)

## Outputs

Pipeline outputs are written to `outputs/` including:
- `aligned_data.parquet`
- `heat_events.parquet`
- `heat_events_with_clusters.parquet`
- `refugia_clusters.parquet`
- `model.pkl`
- `future_refugia_*.parquet`
- `validation_metrics.json`
- `report.md`
- `run_metadata.json`
- `case_studies.md`
- `species_thresholds_used.csv`

## Testing

Tests run against real datasets and require environment variables:
- `CR_GPS_PATH`
- `CR_CLIMATE_PATH`
- `CR_OUTPUTS_DIR` (optional)
- `CR_DATA_DIR` (optional)

## Documentation

- `docs/methods.md`
- `docs/reproducibility.md`
- `docs/data_availability.md`

## Notes

- Provide Movebank and ERA5 data for full analyses and adjust spatial/temporal resolution in config.
