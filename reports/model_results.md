# Model Results

## Baseline

Model: Linear Regression

MAE: 548.2261
RMSE: 2786.0597

## Improved Model

Model: Random Forest Regressor

Features:
- Pickup hour
- Pickup day
- Pickup month
- Pickup weekday
- Haversine distance
- Passenger count
- Pickup/dropoff coordinates
- Vendor
- Store and forward flag

MAE: 456.6057
RMSE: 2973.5571

## Observation

The improved model reduced MAE from 548.2261 to 456.6057.
However, RMSE increased, indicating that extreme prediction errors remain.
