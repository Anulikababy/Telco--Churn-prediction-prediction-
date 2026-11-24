# Project 1: Customer Churn Prediction

Predict whether a telecom customer may leave or stay using tenure, contract type and monthly charges.

## Dataset
Public Telco Customer Churn CSV. The app downloads the data, converts `TotalCharges` to numeric and removes invalid entries.

## Models
- Logistic Regression (primary prediction model)
- Random Forest (model comparison)

## Metrics
Accuracy, precision, recall for the churn class, and confusion matrix, calculated on a held-out test set.

## Run
Install dependencies: `pip install -r requirements.txt`
Run the app: `streamlit run app.py`

## Features in the app
Tenure (months), contract type, monthly charges. The app displays the estimated churn probability.

## Tools
Python, Pandas, Scikit-learn, Streamlit, GitHub, Streamlit Community Cloud.
