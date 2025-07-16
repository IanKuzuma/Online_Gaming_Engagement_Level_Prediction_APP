#Import library
import streamlit as st
import pandas as pd
import pickle

def run():
  # Write title
  st.title("ONLINE GAMING ENGAGEMENT LEVEL PREDICTION - APP")

  # Load the best model's file
  with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

  # Create the inferencing form
  with st.form('player_form'):
    st.subheader('Enter Player Info')

    # Variables for the player data input
    player = st.number_input("Player ID", min_value=9000, step=1)
    age = st.slider("Age", min_value=15, max_value=70, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Location", ["USA", "Asia", "Europe", "Other"])
    genre = st.selectbox("Favorite Game Genre", ["Action", "Strategy", "RPG", "Sports", "Simulation"])
    hours = st.number_input("Total Play Time (hours)", min_value=0.0, step=1)
    purch_label = st.selectbox("In-Game Purchases?", ["No", "Yes"])  # UI labels
    purch = 1 if purch_label == "Yes" else 0  # convert to binary
    diff = st.selectbox("Preferred Game Difficulty", ["Easy", "Medium", "Hard"])
    sesh = st.number_input("Sessions Per Week", min_value=0, step=1)
    minutes = st.number_input("Avg Session Duration (minutes)", min_value=0, step=1)
    level = st.slider("Player Level", min_value=1, max_value=99, value=5)
    achv = st.number_input("Achievements Unlocked", min_value=0, step=1)

    # Submit the player data input button
    submit = st.form_submit_button("Predict Now!")

  # Create the new player data dictionary
  data_player = {
    'PlayerID': player,
    'Age': age,
    'Gender': gender,
    'Location': location,
    'GameGenre': genre,
    'PlayTimeHours': hours,
    'InGamePurchases': purch,
    'GameDifficulty': diff,
    'SessionsPerWeek': sesh,
    'AvgSessionDurationMinutes': minutes,
    'PlayerLevel': level,
    'AchievementsUnlocked': achv
  }

  # Turn them into a dataframe
  data_player = pd.DataFrame([data_player])

  # Show prpediction results
  if submit:
    pred = model.predict(data_player)[0]
    if pred == 0:
      st.markdown('''### Predicted Class is : :blue-background[High Engagement Level]''')
    elif pred == 1:
      st.markdown('''### Predicted Class is : :blue-background[Low Engagement Level]''')
    elif pred == 2:
      st.markdown('''### Predicted Class is : :blue-background[Medium Engagement Level]''')

if __name__ == '__main__':
  run()