from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        return json.dumps(log)


logger = logging.getLogger("prodml")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())

if not logger.handlers:
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


app = FastAPI(
    title="ProdML Taxi Duration API",
    version="0.1.0",
)

MODEL_PATH = Path("models/taxi_model.joblib")

logger.info("Loading model")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")


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
    logger.info("Health check requested")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TripRequest) -> PredictionResponse:
    logger.info("Prediction request received")

    data = pd.DataFrame([request.model_dump()])
    prediction = model.predict(data)[0]
    result = float(prediction)

    logger.info("Prediction completed")

    return PredictionResponse(trip_duration=result)
