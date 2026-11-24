# Customer Churn Prediction

This project predicts whether a telecommunications customer is likely to leave or stay based on customer tenure, contract type, and monthly charges. It compares Logistic Regression and Random Forest models using recall and accuracy, saves the trained models for fast inference, and serves predictions through an interactive Streamlit app.

## Dataset
Telco Customer Churn dataset (7,043 rows and 21 columns).

## Repository Contents
- `streamlit_app.py`: Streamlit web application loading saved `.pkl` models.
- `train_churn.py`: Training script for Logistic Regression and Random Forest models.
- `logistic_churn_model.pkl`, `forest_churn_model.pkl`, `eval_results.pkl`: Saved models and evaluation metrics.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project overview and instructions.
