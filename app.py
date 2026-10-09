"""
app.py
-------
Streamlit web interface for the Employee Attrition Prediction model.

Before running this, run train_model.py once to generate:
    attrition_model.pkl, scaler.pkl, feature_info.pkl

Then start the app with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    meta = joblib.load("feature_info.pkl")
    return model, scaler, meta


try:
    model, scaler, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate attrition_model.pkl, scaler.pkl and feature_info.pkl."
    )
    st.stop()

feature_info = meta["feature_info"]
numerical_columns = meta["numerical_columns"]
columns = meta["columns"]

st.title("🧑‍💼 Employee Attrition Prediction")
st.write(
    "Fill in the employee's details below to predict whether they are "
    "likely to leave the company."
)

user_input = {}

with st.form("attrition_form"):
    st.subheader("Employee Details")

    for col in columns:
        info = feature_info[col]
        if info["type"] == "numeric":
            default_val = round(info["mean"])
            user_input[col] = st.number_input(
                label=col,
                min_value=float(info["min"]),
                max_value=float(info["max"]) * 1.5,  # allow a bit of headroom
                value=float(default_val),
            )
        else:
            user_input[col] = st.selectbox(col, info["options"])

    submitted = st.form_submit_button("Predict Attrition")

if submitted:
    input_df = pd.DataFrame([user_input])
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(
            f"⚠️ This employee is likely to **LEAVE** the company.\n\n"
            f"Probability of leaving: **{probability:.1%}**"
        )
    else:
        st.success(
            f"✅ This employee is likely to **STAY** with the company.\n\n"
            f"Probability of leaving: **{probability:.1%}**"
        )

    st.progress(min(max(probability, 0.0), 1.0))
