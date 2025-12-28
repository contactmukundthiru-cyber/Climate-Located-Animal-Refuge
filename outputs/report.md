# Climate Refugia Report

## Data Summary
- GPS points: 528
- Heat events: 7
- Refugia clusters: 1

## Validation Metrics
- roc_auc: 0.9429
- average_precision: 0.8667
- f1: 0.6667
- precision: 0.7000
- recall: 0.7000

## Statistical Tests
- temp_t_stat: 0.6080
- temp_t_p: 0.5587
- anova_f_stat: nan
- anova_f_p: nan

## Spatial Consistency
- mean_centroid_shift_km: nan
- median_centroid_shift_km: nan

## Experiments
- heatwave_response: outputs/experiment_heatwave_response.parquet
- model_comparison: outputs/experiment_model_comparison.json
- sensitivity: outputs/experiment_sensitivity.parquet

## Case Studies
- outputs/case_studies.md

## Uncertainty Summary
- Samples: 2,000
- Mean prediction std: 0.0896

## Data Availability
Movebank tracking data are available from the Movebank data repository under the terms of each study.
ERA5 climate data are available from the Copernicus Climate Data Store.

## Code Availability
All analysis code used for this report is contained in this repository.
