import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model_path = os.path.join("model", "model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="🏠 NZ Property Valuer", layout="wide")
st.title("📍 Trade Me NZ Property Price Estimator")

# Input fields
address = st.text_input("Enter address (optional):")
bedrooms = st.slider("Number of bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of bathrooms", 1, 5, 2)
floor_area = st.number_input("Floor area (sqm)", 30, 1000, 100)
land_area = st.number_input("Land area (sqm)", 100, 3000, 500)

if st.button("💰 Estimate Price"):
    features = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor_area": floor_area,
        "land_area": land_area
    }])

    prediction = model.predict(features)[0]
    st.success(f"🏷️ Estimated Market Price: NZD ${int(prediction):,}")
