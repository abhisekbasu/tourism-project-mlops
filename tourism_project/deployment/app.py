"""
app.py — Streamlit Inference App
==================================
Loads the best model from HF Model Hub, collects customer inputs
through a web form, and predicts the probability of package purchase.

The decision threshold is read from model_summary.json (uploaded
alongside the model) — there is no hard-coded threshold in this file.
"""
import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

MODEL_REPO     = os.environ.get("HF_MODEL_REPO",      "abhisekbasu/tourism-project-model")
MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME",  "best_tourism_model_v1.joblib")
SUMMARY_FILE   = "model_summary.json"

@st.cache_resource(show_spinner="Loading model from Hugging Face Hub ...")
def load_model_and_threshold():
    try:
        model_path   = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
        summary_path = hf_hub_download(repo_id=MODEL_REPO, filename=SUMMARY_FILE,   repo_type="model")
        model        = joblib.load(model_path)
        with open(summary_path) as fh:
            summary = json.load(fh)
        threshold = float(summary.get("threshold", 0.45))
        return model, threshold
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

model, THRESHOLD = load_model_and_threshold()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🌴 Tourism Package Prediction")
st.markdown(
    "Predict whether a customer will purchase the **Wellness Tourism Package** "
    f"(`ProdTaken = 1`).  Decision threshold: `{THRESHOLD}`"
)

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age                      = st.number_input("Age",                        min_value=18,   max_value=100, value=35)
    city_tier                = st.selectbox("CityTier",                      [1, 2, 3])
    duration_of_pitch        = st.number_input("DurationOfPitch (min)",      min_value=0.0,  max_value=120.0, value=15.0)
    number_of_person_visiting = st.number_input("NumberOfPersonVisiting",    min_value=1,    max_value=10,  value=2)
    number_of_followups      = st.number_input("NumberOfFollowups",          min_value=0.0,  max_value=10.0, value=2.0)
    preferred_property_star  = st.number_input("PreferredPropertyStar",      min_value=1.0,  max_value=5.0, value=3.0)
    number_of_trips          = st.number_input("NumberOfTrips",              min_value=0.0,  max_value=20.0, value=2.0)
    passport                 = st.selectbox("Passport (0=No, 1=Yes)",        [0, 1])
    pitch_satisfaction_score = st.slider("PitchSatisfactionScore",           1, 5, 3)
    own_car                  = st.selectbox("OwnCar (0=No, 1=Yes)",          [0, 1])
    number_of_children       = st.number_input("NumberOfChildrenVisiting",   min_value=0.0,  max_value=5.0, value=0.0)
    monthly_income           = st.number_input("MonthlyIncome (₹)",          min_value=0.0,  value=30000.0, step=1000.0)

with col2:
    type_of_contact  = st.selectbox("TypeofContact",  ["Self Enquiry", "Company Invited"])
    occupation       = st.selectbox("Occupation",     ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender           = st.selectbox("Gender",         ["Male", "Female"])
    product_pitched  = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    marital_status   = st.selectbox("MaritalStatus",  ["Single", "Married", "Divorced", "Unmarried"])
    designation      = st.selectbox("Designation",    ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Build input DataFrame — column order must match training data
input_df = pd.DataFrame([{
    "Age":                      age,
    "TypeofContact":            type_of_contact,
    "CityTier":                 city_tier,
    "DurationOfPitch":          duration_of_pitch,
    "Occupation":               occupation,
    "Gender":                   gender,
    "NumberOfPersonVisiting":   number_of_person_visiting,
    "NumberOfFollowups":        number_of_followups,
    "ProductPitched":           product_pitched,
    "PreferredPropertyStar":    preferred_property_star,
    "MaritalStatus":            marital_status,
    "NumberOfTrips":            number_of_trips,
    "Passport":                 passport,
    "PitchSatisfactionScore":   pitch_satisfaction_score,
    "OwnCar":                   own_car,
    "NumberOfChildrenVisiting": number_of_children,
    "Designation":              designation,
    "MonthlyIncome":            monthly_income,
}])

st.markdown("---")
st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("🔍 Predict", use_container_width=True):
    proba = model.predict_proba(input_df)[0, 1]
    pred  = int(proba >= THRESHOLD)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.metric("Purchase Probability", f"{proba:.1%}")
    col_b.metric("Prediction (ProdTaken)", "✅ Will Buy" if pred == 1 else "❌ Will Not Buy")

    if pred == 1:
        st.success("High likelihood of purchase — recommend proactive outreach.")
    else:
        st.info("Low likelihood of purchase — consider deprioritising for this campaign.")
