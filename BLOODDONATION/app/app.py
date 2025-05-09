import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import xgboost as xgb

# Define the model path
model_path = os.path.join('..', 'model', 'blood_donor_xgb_model.pkl')

# Function to train and save model
def train_and_save_model(save_path):
    # Create dummy data for training
    data = {
        'months_since_last_donation': np.random.randint(0, 12, 500),
        'number_of_donations': np.random.randint(1, 10, 500),
        'total_volume_donated': np.random.randint(250, 5000, 500),
        'months_since_first_donation': np.random.randint(6, 120, 500),
    }
    df = pd.DataFrame(data)
    df['made_donation'] = np.random.choice([0, 1], size=500)  # Random target (0 or 1)

    X = df[['months_since_last_donation', 'number_of_donations', 'total_volume_donated', 'months_since_first_donation']]
    y = df['made_donation']

    model = xgb.XGBClassifier()
    model.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(model, open(save_path, 'wb'))

    return model

# Load or train model
if not os.path.exists(model_path):
    st.warning(f"Model file not found at {model_path}. Training a new model...")
    model = train_and_save_model(model_path)
    st.success("✅ Model trained and saved successfully!")
else:
    try:
        model = pickle.load(open(model_path, 'rb'))
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

# Streamlit App
st.title("Blood Donor Prediction System 🚑")
st.markdown("### Enter Donor Details Below:")

# Input fields
months_since_last_donation = st.number_input('Months Since Last Donation', min_value=0, max_value=120, value=6)
number_of_donations = st.number_input('Number of Donations', min_value=0, max_value=50, value=3)
total_volume_donated = st.number_input('Total Volume Donated (in c.c.)', min_value=0, max_value=20000, value=1500)
months_since_first_donation = st.number_input('Months Since First Donation', min_value=0, max_value=500, value=24)

# Predict Button
if st.button('Predict Donation Possibility'):
    input_data = np.array([[months_since_last_donation, number_of_donations, total_volume_donated, months_since_first_donation]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success('✅ Likely to Donate Blood Again!')
    else:
        st.error('⚠️ Might Not Donate Blood Again.')
