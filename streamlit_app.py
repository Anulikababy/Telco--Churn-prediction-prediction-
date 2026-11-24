import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecommunications customer is likely to leave or stay using tenure, contract type, and monthly charges.")

try:
    logistic_model = joblib.load("logistic_churn_model.pkl")
    forest_model = joblib.load("forest_churn_model.pkl")
    eval_results = joblib.load("eval_results.pkl")
except Exception as e:
    st.error("Unable to load saved model files. Make sure .pkl files are in the same folder as streamlit_app.py.")
    st.error(f"Details: {e}")
    st.stop()

st.subheader("Model Performance Comparison")
for name in ["Logistic Regression", "Random Forest"]:
    m = eval_results[name]
    st.write(f"### {name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.2%}")
    c2.metric("Precision", f"{m['precision']:.2%}")
    c3.metric("Recall (Churn)", f"{m['recall']:.2%}")
    st.dataframe(
        pd.DataFrame(
            m["cm"],
            index=["Actual Stay", "Actual Churn"],
            columns=["Predicted Stay", "Predicted Churn"]
        )
    )

st.subheader("Predict Customer Churn")
with st.form("prediction_form"):
    tenure = st.number_input("Customer Tenure (months)", min_value=0, max_value=120, value=12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0)
    submit = st.form_submit_button("Predict Churn")

if submit:
    customer = pd.DataFrame([{
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly_charges
    }])
    pred = int(logistic_model.predict(customer)[0])
    prob = float(logistic_model.predict_proba(customer)[0][1])
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"Likely to leave — Churn Probability: {prob:.2%}")
    else:
        st.success(f"Likely to stay — Stay Probability: {(1 - prob):.2%} (Churn Probability: {prob:.2%})")

st.caption("Developed by Nwikwu Anulika Oluchukwu | Machine Learning Project 1")
