import streamlit as st
import pickle
import pandas as pd

# Load the trained RandomForestRegressor model
model = pickle.load(open('player_rating_predictor.pkl', 'rb'))

# Create a function to make predictions
def predict_player_rating(input_data):
    return model.predict(input_data)

# Streamlit UI
st.title("FIFA Player Rating Predictor")

# Create input fields for user to enter player attributes
st.sidebar.header("Enter Player Attributes")

# Define input fields for relevant features (modify as needed)
features = ['movement_reactions', 'mentality_composure', 'passing', 'potential', 'release_clause_eur', 'dribbling', 'wage_eur', 'power_shot_power', 'value_eur', 'mentality_vision', 'attacking_short_passing']
input_data = []

for feature in features:
    input_value = st.sidebar.number_input(f"Enter {feature}", value=0)
    input_data.append(input_value)

# Create a button to make predictions
if st.sidebar.button("Predict"):
    input_data = [input_data]
    prediction = predict_player_rating(input_data)
    st.write(f"Predicted Player Rating: {prediction[0]:.2f}")



# Deploy the Streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="FIFA Player Rating Predictor", page_icon="⚽")
    st.sidebar.markdown("By Your Name")
    st.write("Welcome to the FIFA Player Rating Predictor!")
    st.write("Enter player attributes on the left sidebar to predict the player's rating.")


