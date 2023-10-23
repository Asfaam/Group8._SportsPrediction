import pickle
import streamlit as st
import numpy as np

# Load the trained RandomForestRegressor model
model = pickle.load(open('player_rating_predictor.pkl', 'rb'))

# Function to predict the player's rating and calculate confidence
def predict_rating(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    
    # Calculate confidence (standard deviation in this example)
    confidence = np.std(model.predict(features))
    
    return prediction[0], confidence

# Streamlit app
st.title("FIFA Player Rating Predictor")

# Input fields for user to enter player attributes
st.write("Enter Player Attributes:")
features = []

# Define input fields for relevant features from the dataset
age = st.number_input("Age", min_value=16, max_value=45, value=20)
features.append(age)

overall = st.number_input("Overall Rating", min_value=40, max_value=100, value=70)
features.append(overall)

potential = st.number_input("Potential Rating", min_value=40, max_value=100, value=80)
features.append(potential)

value_eur = st.number_input("Value (in Euros)", min_value=0, value=500000)
features.append(value_eur)

wage_eur = st.number_input("Wage (in Euros)", min_value=0, value=5000)
features.append(wage_eur)

# Make prediction when the "Predict" button is clicked
if st.button("Predict Rating"):
    prediction, confidence = predict_rating(features)
    st.write(f"Predicted Player Rating: {prediction:.2f}")
    st.write(f"Confidence Score (Standard Deviation): {confidence:.2f}")

st.write("Note: The model's prediction and confidence score may not be accurate for every player.")
