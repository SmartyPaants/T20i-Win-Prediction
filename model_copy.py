import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load dataset for dropdown options
df = pd.read_csv('t20_data_clean.csv.zip', compression="zip")
venue_stats = pd.read_csv("venue_stats.csv")

# Extract unique teams and venues dynamically
teams = sorted(set(df['batting_team'].unique()) | set(df['bowling_team'].unique()))
venues = sorted(df['venue'].dropna().unique())

# Load dual-model bundle
model_bundle = pickle.load(open('pipe_new.pkl', 'rb'))
model_inn1 = model_bundle['model_inn1']
model_inn2 = model_bundle['model_inn2']
label_encoders = model_bundle['label_encoders']
features_inn1 = model_bundle['features_inn1']
features_inn2 = model_bundle['features_inn2']

st.title('🏏 T20I Win Predictor')

# User inputs
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_options = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox('Select the bowling team', bowling_options)

selected_venue = st.selectbox('Select venue', venues)

col3, col4, col5 = st.columns(3)
with col3:
    toss_winner = st.selectbox('Toss Winner', [batting_team, bowling_team])
with col4:
    toss_decision = st.selectbox('Toss Decision', ['bat', 'field'])
with col5:
    innings = st.selectbox('Innings', [1, 2])

# Target only if innings 2
if innings == 2:
    target_runs = st.number_input('Target runs', min_value=1)
else:
    target_runs = -1  # placeholder for innings 1

col6, col7, col8 = st.columns(3)
with col6:
    runs_so_far = st.number_input('Runs scored so far', min_value=0)
with col7:
    overs = st.number_input('Overs completed', min_value=0.0, step=0.1, format="%.1f", max_value=20.0)
with col8:
    wickets_so_far = st.number_input('Wickets lost so far', min_value=0, max_value=10)

if st.button('Predict Probability'):
    whole_num = int(overs)
    fractional_num = overs - whole_num
    balls_faced = whole_num * 6 + round(fractional_num * 10)
    # print(balls_faced)
    run_rate = runs_so_far / overs if overs > 0 else 0
    balls_left = 120 - balls_faced
    runs_left = target_runs - runs_so_far if innings == 2 else -1
    required_run_rate = (runs_left * 6) / balls_left if innings == 2 and balls_left > 0 else -1

    year = datetime.now().year

    projected_score = runs_so_far + (runs_so_far / balls_faced) * (120 - balls_faced) if balls_faced > 0 else runs_so_far

    # Create input dataframe
    input_dict = {
        'venue': [selected_venue],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'innings': [innings],
        'runs_so_far': [runs_so_far],
        'wickets_so_far': [wickets_so_far],
        'balls_faced': [balls_faced],
        'run_rate': [run_rate],
        'target_runs': [target_runs],
        'required_run_rate': [required_run_rate],
        'year': [year]
}

    input_df = pd.DataFrame(input_dict)
    input_df = input_df.merge(venue_stats, on="venue", how="left")
    input_df["score_vs_avg"] = (input_df["runs_so_far"] - input_df["avg_1st_score"])
    input_df["projected_score"] = projected_score
    input_df["proj_vs_avg"] = (input_df["projected_score"] - input_df["avg_1st_score"])

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in input_df.columns:
            val = input_df.at[0, col]
            if val not in le.classes_:
                st.error(f"Value '{val}' not recognized in '{col}' encoder.")
                st.stop()
            input_df[col] = le.transform(input_df[col])

    # Pick model and features based on innings
    if innings == 1:
        input_df = input_df[features_inn1]
        model = model_inn1
    else:
        input_df = input_df[features_inn2]
        model = model_inn2

    # Predict
    result = model.predict_proba(input_df)
    loss_prob = result[0][0]
    win_prob = result[0][1]

    # Handle case where all wickets are lost and target not reached
    if innings == 2 and wickets_so_far == 10 and runs_so_far < target_runs:
        win_prob = 0.0
        loss_prob = 1.0
    # Handle case where target is reached
    elif innings == 2 and runs_so_far >= target_runs:
        win_prob = 1.0
        loss_prob = 0.0

    # Output
    st.markdown("### Win Probability")
    colA, colB = st.columns(2)
    with colA:
        st.write(f"**{batting_team}**")
        st.progress(win_prob)
        st.write(f"{win_prob*100:.2f}%")
    with colB:
        st.write(f"**{bowling_team}**")
        st.progress(loss_prob)
        st.write(f"{loss_prob*100:.2f}%")
