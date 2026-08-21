from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="ProdML Taxi Duration API",
    version="0.1.0",
)

MODEL_PATH = Path("models/taxi_model.joblib")

model = joblib.load(MODEL_PATH)


class TripRequest(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str
    pickup_hour: int
    pickup_day: int
    pickup_month: int
    pickup_weekday: int
    distance_km: float


class PredictionResponse(BaseModel):
    trip_duration: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TripRequest) -> PredictionResponse:
    data = pd.DataFrame([request.model_dump()])

    prediction = model.predict(data)[0]

    return PredictionResponse(
        trip_duration=float(prediction)
    )