import streamlit as st
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



model = joblib.load("trained_model.pkl")

# Sample training data
data = pd.DataFrame({
    'Price': [100, 150, 200, 250, 300, 120, 180, 220],
    'CustomerType': ['New', 'Existing', 'New', 'Existing', 'New', 'Existing', 'New', 'Existing'],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
    'DealSize': ['Small', 'Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium'],
    'Won': [1, 0, 1, 0, 1, 0, 1, 0]
})

# Features and target
X = data[['Price', 'CustomerType', 'Region', 'DealSize']]
y = data['Won']

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['CustomerType', 'Region', 'DealSize'])
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
model.fit(X, y)

# Streamlit app
st.title("Price-Win Probability Model")

# User inputs
price = st.number_input("Price", min_value=0.0)
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
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted Win Probability: {probability:.2%}")
