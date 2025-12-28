# Reproducibility

## Environment
- Python 3.10+
- Install dependencies with `pip install -r requirements.txt`

## Data Acquisition
### Movebank
Download GPS data for the target study using `scripts/download_movebank.py`. Required environment variables:
- `MOVEBANK_USERNAME`
- `MOVEBANK_PASSWORD`

For available parameters and required fields:
```bash
python3 scripts/download_movebank.py --help
```

Some studies require accepting license terms in the Movebank web interface before API download. If you see a license acceptance error, log in to Movebank and accept the study license, then retry the download.

### ERA5
Configure `~/.cdsapirc` for the Copernicus Climate Data Store and download hourly data. For available parameters:
```bash
python3 scripts/download_era5.py --help
```

The ERA5 dataset requires accepting CDS license terms for each dataset. If you see a 403 with “required licences not accepted”, log in to the CDS portal and accept the license for ERA5 single levels.

### Species Thresholds
Provide `climate_refugia/data/species_thresholds.csv` with columns `species,heat_threshold_c`.

## Pipeline
Run the full pipeline with your downloaded datasets:
```bash
python3 scripts/run_pipeline.py --help
```

## Validation
Validate data quality before running the pipeline:
```bash
python3 scripts/validate_data.py --help
```

## ERA5 Request Size
The CDS API enforces request size limits. If you see a “cost limits exceeded” error, reduce the spatial extent, shorten the date range, or run smaller time windows. The `scripts/download_data_bundle.py` script supports `--start`, `--end`, and `--grid` to limit request size.

## Outputs
Outputs are written to `outputs/` and include aligned data, heat event summaries, refugia clusters, trained model, validation metrics, uncertainty estimates, and a report.
