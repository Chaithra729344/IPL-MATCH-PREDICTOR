import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the dataset
df = pd.read_csv('dataset.csv')

# Extract unique team names and cities from the dataset
teams = df['batting_team'].unique()
cities = df['city'].unique()

# Load the pre-trained model
pipe = pickle.load(open('ra_pipe.pkl', 'rb'))

# Function to predict win and loss probabilities for the batting team
def predict_probabilities(batting_team, bowling_team, selected_city, target, score, balls_left, wickets):
    runs_left = target - score
    wickets_remaining = 10 - wickets
    overs_completed = (120 - balls_left) / 6
    crr = score / overs_completed
    rrr = runs_left / (balls_left / 6)

    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_remaining': [wickets_remaining],
        'total_run_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_data)
    win_probability = round(result[0][1] * 100, 2)
    loss_probability = round(result[0][0] * 100, 2)

    return win_probability, loss_probability

# Streamlit app
st.title('IPL Match Predictor')

# User inputs
batting_team = st.selectbox('Select Batting Team', teams)
bowling_team = st.selectbox('Select Bowling Team', teams)
selected_city = st.selectbox('Select City', cities)
target = st.number_input('Target Score', min_value=0)
score = st.number_input('Current Score', min_value=0)
balls_left = st.number_input('Balls Left', min_value=0)
wickets = st.number_input('Wickets Fallen', min_value=0)

# Predict button
if st.button('Predict'):
    win_prob_batting, loss_prob_batting = predict_probabilities(batting_team, bowling_team, selected_city, target, score, balls_left, wickets)
    st.write(f'Win Probability for {batting_team}: {win_prob_batting}%')
    st.write(f'Loss Probability for {batting_team}: {loss_prob_batting}%')
