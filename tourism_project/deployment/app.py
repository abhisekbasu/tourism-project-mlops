import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "abhisekbasu/tourism-project-model")
MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_tourism_model_v1.joblib")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

st.title("Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package (ProdTaken: 0/1).")

# Input fields (match training columns)
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=35.0, step=1.0)
typeofcontact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("CityTier", [1, 2, 3])
durationofpitch = st.number_input("DurationOfPitch", min_value=0.0, max_value=300.0, value=15.0, step=1.0)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=20, value=2, step=1)
num_followups = st.number_input("NumberOfFollowups", min_value=0.0, max_value=20.0, value=3.0, step=1.0)
productpitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King", "Other"])
preferred_star = st.number_input("PreferredPropertyStar", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
marital = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried", "Other"])
num_trips = st.number_input("NumberOfTrips", min_value=0.0, max_value=50.0, value=2.0, step=1.0)
passport = st.selectbox("Passport", [0, 1])
pitch_score = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5])
owncar = st.selectbox("OwnCar", [0, 1])
num_children = st.number_input("NumberOfChildrenVisiting", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"])
monthly_income = st.number_input("MonthlyIncome", min_value=0.0, max_value=10000000.0, value=30000.0, step=1000.0)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": int(citytier),
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(num_person),
    "NumberOfFollowups": num_followups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital,
    "NumberOfTrips": num_trips,
    "Passport": int(passport),
    "PitchSatisfactionScore": int(pitch_score),
    "OwnCar": int(owncar),
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(input_df)[:, 1][0])

    if int(pred) == 1:
        st.success(f"Prediction: WILL purchase (ProdTaken = 1)")
    else:
        st.warning(f"Prediction: will NOT purchase (ProdTaken = 0)")

    if proba is not None:
        st.info(f"Purchase probability (class 1): {proba:.3f}")

st.caption(f"Model source: {HF_MODEL_REPO}/{MODEL_FILENAME}")
