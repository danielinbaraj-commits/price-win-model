import streamlit as st
import pandas as pd
import os
import base64
import joblib

# Load the pre-trained model


# Decode the base64 model file
with open("trained_model_streamlit.b64.txt", "rb") as f:
    model_bytes = base64.b64decode(f.read())

# Save to a temporary file
with open("temp_model.pkl", "wb") as f:
    f.write(model_bytes)


# model_path = os.path.join(os.path.dirname(__file__), "trained_model_streamlit.pkl")
try:
    model = joblib.load("temp_model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop

# Streamlit app
st.title("Price-Win Probability Model")

# User inputs
price = st.number_input("Price", min_value=1.0)
customer_type = st.selectbox("Customer Type", options=['New', 'Existing'])
region = st.selectbox("Region", options=['North', 'South', 'East', 'West'])
deal_size = st.selectbox("Deal Size", options=['Small', 'Medium', 'Large'])

# Predict button
if st.button("Predict Win Probability"):
    input_df = pd.DataFrame({
        'Price': [price],
        'CustomerType': [customer_type],
        'Region': [region],
        'DealSize': [deal_size]
    })

    try:
        probability = model.predict_proba(input_df)[0][1]
        st.success(f"Predicted Win Probability: {probability:.2%}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
