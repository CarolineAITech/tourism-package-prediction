"""
app.py
-------
Streamlit front end for the Wellness Tourism Package predictor.

Loads the registered model from the Hugging Face model hub, collects customer
attributes from the user, assembles them into a single-row DataFrame whose
columns match the training schema, and returns a purchase prediction.
"""

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = "CarolineBuildsAI"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"
MODEL_FILE = "best_tourism_model.joblib"

st.set_page_config(page_title="Wellness Tourism Package Predictor",
                   page_icon="🧳", layout="centered")


# ----------------------------------------------------------------------
# Load the saved model from the Hugging Face model hub (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


model = load_model()

st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details below to predict whether they are likely to "
    "purchase the Wellness Tourism Package, before a salesperson contacts them."
)

# ----------------------------------------------------------------------
# Collect the inputs
# ----------------------------------------------------------------------
st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact",
                                   ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation",
                              ["Salaried", "Small Business",
                               "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status",
                                  ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation",
                               ["Executive", "Manager", "Senior Manager",
                                "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000,
                                     max_value=100000, value=23000, step=500)

with col2:
    num_persons = st.number_input("Number of Persons Visiting",
                                  min_value=1, max_value=10, value=3)
    num_children = st.number_input("Number of Children Visiting (under 5)",
                                   min_value=0, max_value=5, value=1)
    num_trips = st.number_input("Number of Trips per Year",
                                min_value=0, max_value=30, value=3)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    passport = st.selectbox("Holds a Passport", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car", ["No", "Yes"])

st.subheader("Interaction details")
col3, col4 = st.columns(2)
with col3:
    product_pitched = st.selectbox("Product Pitched",
                                   ["Basic", "Deluxe", "Standard",
                                    "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)",
                                        min_value=1, max_value=60, value=15)
with col4:
    num_followups = st.number_input("Number of Follow-ups",
                                    min_value=0, max_value=10, value=4)
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# ----------------------------------------------------------------------
# Assemble the inputs into a DataFrame with the training column order
# ----------------------------------------------------------------------
input_df = pd.DataFrame([{
    "Age": float(age),
    "TypeofContact": type_of_contact,
    "CityTier": int(city_tier),
    "DurationOfPitch": float(duration_of_pitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(num_persons),
    "NumberOfFollowups": float(num_followups),
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": float(preferred_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": float(num_trips),
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": int(pitch_satisfaction),
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": float(num_children),
    "Designation": designation,
    "MonthlyIncome": float(monthly_income),
}])

with st.expander("Review the input sent to the model"):
    st.dataframe(input_df)

# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
if st.button("Predict", type="primary"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the package "
                   f"(probability {probability:.1%})")
        st.write("Recommended action: prioritise this customer for outreach.")
    else:
        st.error(f"Unlikely to purchase the package "
                 f"(probability {probability:.1%})")
        st.write("Recommended action: deprioritise, or target with a "
                 "lower-tier offer.")

    st.progress(float(probability))
    st.caption(f"Predicted purchase probability: {probability:.2%}")
