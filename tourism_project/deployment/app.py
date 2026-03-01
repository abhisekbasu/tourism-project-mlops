import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

# Reading model repo and filename from environment variables
# Defaults point to the correct HF Model Hub repo if no env vars are set
MODEL_REPO     = os.environ.get("HF_MODEL_REPO",     "abhisekbasu/tourism-project-model")
MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_tourism_model_v1.joblib")
SUMMARY_FILE   = "model_summary.json"

@st.cache_resource(show_spinner="Loading model from Hugging Face Hub ...")
def load_model_and_threshold():
    # Downloading the model and summary file from Hugging Face Model Hub
    # The threshold is stored in model_summary.json so there is only one place to update it
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

st.title("Tourism Package Prediction")
st.markdown(
    "Predicting whether a customer will purchase the **Wellness Tourism Package** "
    f"(`ProdTaken = 1`).  Decision threshold: `{THRESHOLD}`"
)

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age                       = st.number_input("Age",                    min_value=18,  max_value=100, value=35, step=1)
    city_tier                 = st.selectbox("CityTier",                  [1, 2, 3])
    duration_of_pitch         = st.number_input("DurationOfPitch (min)",  min_value=0.0, max_value=120.0, value=15.0)
    number_of_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1,   max_value=10,  value=2, step=1)
    # Using selectbox for whole-number fields to prevent invalid decimal inputs
    number_of_followups       = st.selectbox("NumberOfFollowups",         [1, 2, 3, 4, 5, 6])
    preferred_property_star   = st.selectbox("PreferredPropertyStar",     [1, 2, 3, 4, 5])
    number_of_trips           = st.selectbox("NumberOfTrips",             [1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22])
    passport                  = st.selectbox("Passport",                  [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score  = st.slider("PitchSatisfactionScore",       1, 5, 3)
    own_car                   = st.selectbox("OwnCar",                    [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children        = st.selectbox("NumberOfChildrenVisiting",  [0, 1, 2, 3])
    monthly_income            = st.number_input("MonthlyIncome",          min_value=0.0, value=30000.0, step=1000.0)

with col2:
    type_of_contact = st.selectbox("TypeofContact",  ["Self Enquiry", "Company Invited"])
    occupation      = st.selectbox("Occupation",     ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender          = st.selectbox("Gender",         ["Male", "Female"])
    product_pitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    marital_status  = st.selectbox("MaritalStatus",  ["Single", "Married", "Divorced", "Unmarried"])
    designation     = st.selectbox("Designation",    ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Building the input DataFrame — column names must exactly match the training data
input_df = pd.DataFrame([{
    "Age":                      age,
    "TypeofContact":            type_of_contact,
    "CityTier":                 city_tier,
    "DurationOfPitch":          duration_of_pitch,
    "Occupation":               occupation,
    "Gender":                   gender,
    "NumberOfPersonVisiting":   number_of_person_visiting,
    "NumberOfFollowups":        float(number_of_followups),
    "ProductPitched":           product_pitched,
    "PreferredPropertyStar":    float(preferred_property_star),
    "MaritalStatus":            marital_status,
    "NumberOfTrips":            float(number_of_trips),
    "Passport":                 passport,
    "PitchSatisfactionScore":   pitch_satisfaction_score,
    "OwnCar":                   own_car,
    "NumberOfChildrenVisiting": float(number_of_children),
    "Designation":              designation,
    "MonthlyIncome":            monthly_income,
}])

st.markdown("---")
st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict", use_container_width=True):
    proba = model.predict_proba(input_df)[0, 1]
    pred  = int(proba >= THRESHOLD)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.metric("Purchase Probability", f"{proba:.1%}")
    col_b.metric("Prediction (ProdTaken)", "Will Buy" if pred == 1 else "Will Not Buy")

    if pred == 1:
        st.success("High likelihood of purchase. Recommend proactive outreach.")
    else:
        st.info("Low likelihood of purchase. Consider lower priority for this campaign.")
