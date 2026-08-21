import streamlit as st
import requests

st.title("?? Taxi Trip Duration Predictor")

vendor_id = st.text_input("Vendor ID", "1")
store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["N", "Y"])
passenger_count = st.number_input("Passenger Count", min_value=1, value=1)

pickup_longitude = st.number_input("Pickup Longitude", value=-73.985)
pickup_latitude = st.number_input("Pickup Latitude", value=40.748)
dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.985)
dropoff_latitude = st.number_input("Dropoff Latitude", value=40.748)

pickup_hour = st.number_input("Pickup Hour", min_value=0, max_value=23, value=12)
pickup_day = st.number_input("Pickup Day", min_value=1, max_value=31, value=15)
pickup_month = st.number_input("Pickup Month", min_value=1, max_value=12, value=6)
pickup_weekday = st.number_input("Pickup Weekday", min_value=0, max_value=6, value=2)
distance_km = st.number_input("Distance (km)", min_value=0.0, value=2.0)

if st.button("Predict"):
    payload = {
        "vendor_id": vendor_id,
        "store_and_fwd_flag": store_and_fwd_flag,
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "pickup_hour": pickup_hour,
        "pickup_day": pickup_day,
        "pickup_month": pickup_month,
        "pickup_weekday": pickup_weekday,
        "distance_km": distance_km,
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=30,
        )

        if response.ok:
            prediction = response.json()
            st.success(
                f"Predicted trip duration: "
                f"{prediction['trip_duration']:.1f} seconds"
            )
        else:
            st.error(f"API error: {response.text}")

    except requests.RequestException as e:
        st.error(f"Connection error: {e}")
