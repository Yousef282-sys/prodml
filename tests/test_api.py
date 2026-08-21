from fastapi.testclient import TestClient

from prodml.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "vendor_id": 1,
        "passenger_count": 1,
        "pickup_longitude": -73.985,
        "pickup_latitude": 40.748,
        "dropoff_longitude": -73.975,
        "dropoff_latitude": 40.758,
        "store_and_fwd_flag": "N",
        "pickup_hour": 12,
        "pickup_day": 15,
        "pickup_month": 8,
        "pickup_weekday": 5,
        "distance_km": 1.4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "trip_duration" in data
    assert isinstance(data["trip_duration"], float)
    assert data["trip_duration"] > 0
