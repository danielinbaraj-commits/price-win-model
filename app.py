import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
try:
    model = joblib.load("trained_model.pkl")
except FileNotFoundError:
    st.error("Model file 'trained_model.pkl' not found. Please ensure it is in the correct directory.")
    st.stop()

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
