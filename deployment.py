# Import necessary libraries
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained RandomForestRegressor model using joblib
model = joblib.load('player_rating_predictor.joblib')

# Create a Streamlit web app
st.title("FIFA Player Rating Predictor")

# Create input elements for user to provide player profile data
st.header("Enter Player Profile Data")
age = st.slider("Age", 16, 45, 25)
overall = st.slider("Overall Rating", 50, 100, 75)
potential = st.slider("Potential Rating", 50, 100, 80)

# Create a button to predict the rating
if st.button("Predict Rating"):
    # Prepare input data for prediction
    input_data = np.array([age, overall, potential]).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the predicted rating
    st.subheader("Predicted Player Rating:")
    st.write(prediction[0])

# Add an explanation about the model and how to use the app
st.write("This web app uses a RandomForestRegressor model to predict a player's rating based on their profile data.")
st.write("You can adjust the player's age, overall rating, and potential rating using the sliders above, and click the 'Predict Rating' button to see the predicted rating.")

# Optionally, display model confidence score if available
st.write("Confidence scores can be displayed if provided by the model.")

