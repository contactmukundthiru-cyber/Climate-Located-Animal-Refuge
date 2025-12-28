# Methods

## Overview
This project identifies thermal refugia from animal movement data and hourly climate reanalysis, then predicts future refugia under climate scenarios. The workflow follows eight stages: data collection, preprocessing and alignment, heat event detection, spatiotemporal clustering, predictive modeling, statistical validation, experiments, and visualization.

## Animal Movement Data
Tracking data are acquired through the Movebank direct-read API. Events are filtered to include timestamps, geographic coordinates, individual identifiers, and species. GPS points are quality controlled by removing invalid coordinates, enforcing a minimum fix interval, and excluding unrealistic speeds. Movement metrics use haversine distances on a spherical Earth approximation.

## Climate Data
Hourly ERA5 single-level variables are retrieved from the Copernicus Climate Data Store. The pipeline uses 2 m temperature, 2 m dewpoint temperature (for relative humidity), and total precipitation. ERA5 fields are converted to a tidy tabular format with temperature in Celsius, humidity as percent, and precipitation in millimeters.

## Spatiotemporal Alignment
Animal positions are aligned to climate grid points by nearest-neighbor matching on latitude and longitude using a haversine BallTree. Climate time series are matched to each GPS record using nearest-neighbor temporal alignment within a user-defined tolerance.

## Heat Event Detection
Species-specific heat thresholds are applied to aligned temperature observations. Continuous sequences exceeding the threshold are grouped into heat events. Event duration, mean temperature, and maximum temperature are summarized for each animal and event.

## Refugia Identification
Heat-exposed positions are clustered using DBSCAN with a haversine distance metric. Clusters with repeated use across individuals and events are labeled refugia. Cluster centroids provide spatial summaries for mapping and downstream modeling.

## Predictive Modeling
A Random Forest classifier predicts whether a location is a refugia point. Features include latitude, longitude, temperature, humidity, precipitation, hour of day, day of year, and species identifiers. The model is trained on heat-exposed points labeled by proximity to observed refugia clusters.

## Validation
Model performance is assessed using stratified cross-validation with ROC AUC, average precision, F1, precision, and recall. Statistical tests compare refugia and non-refugia conditions using Welch’s t-test and species-level ANOVA. Spatial consistency is assessed by centroid shifts across years. Bootstrap resampling quantifies predictive uncertainty.

## Experiments
Experiments quantify heatwave response across years, compare climate scenarios, evaluate empirical versus climate-defined refugia overlap, and assess sensitivity to thermal thresholds.

## Visualization
Interactive maps display heat exposure, refugia clusters, and future predictions. The web application supports species filtering and scenario selection.
