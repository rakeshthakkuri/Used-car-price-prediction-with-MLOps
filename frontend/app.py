import streamlit as st
import requests
import json

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")

st.markdown("Enter the details of the car below to get the predicted price:")

# Input fields
form_data = {
    "Location": st.text_input("Location", "Mumbai"),
    "Kilometers_Driven": st.number_input("Kilometers Driven", min_value=0, value=50000),
    "Fuel_Type": st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"]),
    "Transmission": st.selectbox("Transmission", ["Manual", "Automatic"]),
    "Owner_Type": st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"]),
    "Mileage": st.number_input("Mileage (km/l)", min_value=0.0, value=18.0),
    "Engine": st.number_input("Engine (CC)", min_value=500.0, value=1200.0),
    "Power": st.number_input("Power (BHP)", min_value=20.0, value=90.0),
    "Seats": st.selectbox("Seats", list(range(2, 9)), index=3),
    "Manufacturer": st.text_input("Manufacturer", "BMW 3 Series"),
    "Year Used": st.number_input("Years Used", min_value=0, value=5)
}

# Prediction trigger
if st.button("Predict Price"):
    with st.spinner("Contacting model..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=form_data)
            if response.status_code == 200:
                price = response.json().get("predicted_price", "N/A")
                st.success(f"💰 Predicted Car Price: ₹{price} Lakhs")
            else:
                st.error(f"Server returned error: {response.status_code}\n{response.text}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
