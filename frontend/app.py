import streamlit as st
import requests

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")

st.markdown("Fill in the car details below to predict its price:")

# Input fields with blank or neutral defaults
form_data = {
    "Location": st.text_input("Location", ""),
    "Kilometers_Driven": st.number_input("Kilometers Driven", min_value=0, step=1000, value=0),
    "Fuel_Type": st.selectbox("Fuel Type", ["", "Petrol", "Diesel", "CNG", "LPG", "Electric"]),
    "Transmission": st.selectbox("Transmission", ["", "Manual", "Automatic"]),
    "Owner_Type": st.selectbox("Owner Type", ["", "First", "Second", "Third", "Fourth & Above"]),
    "Mileage": st.number_input("Mileage (km/l)", min_value=0.0, step=0.1, value=0.0),
    "Engine": st.number_input("Engine (CC)", min_value=0.0, step=100.0, value=0.0),
    "Power": st.number_input("Power (BHP)", min_value=0.0, step=5.0, value=0.0),
    "Seats": st.selectbox("Seats", ["", 2, 3, 4, 5, 6, 7, 8]),
    "Manufacturer": st.text_input("Manufacturer", ""),
    "Year Used": st.number_input("Years Used", min_value=0, value=0)
}

# Predict button
if st.button("Predict Price"):
    # Basic validation
    if (
        "" in form_data.values() or
        any(v == 0 or v == 0.0 for k, v in form_data.items() if isinstance(v, (int, float)) and k not in ["Year Used"])
    ):
        st.warning("⚠️ Please fill in all fields with valid values.")
    else:
        with st.spinner("Contacting model..."):
            try:
                response = requests.post("http://api:8000/predict", json=form_data)
                if response.status_code == 200:
                    price = response.json().get("predicted_price", "N/A")
                    st.success(f"💰 Predicted Car Price: ₹{price} Lakhs")
                else:
                    st.error(f"❌ Server error: {response.status_code}\n{response.text}")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")
