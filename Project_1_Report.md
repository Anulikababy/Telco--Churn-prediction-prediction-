# Project 1 Report: Customer Churn Prediction

## Objective
Develop a supervised Machine Learning model that estimates whether a telecommunications customer is likely to leave or stay, using tenure, contract type and monthly charges.

## Problem Statement
Telecommunications businesses may lose revenue when customers discontinue services. Identifying customers who may churn can help businesses explore retention strategies. This project uses classification models to estimate churn from selected customer attributes.

## Dataset
Public Telco Customer Churn dataset. The app loads the CSV, converts `TotalCharges` to numeric and removes rows with invalid values.

## Tools, Technologies and Platforms
Python; Pandas; Scikit-learn; Logistic Regression; Random Forest; StandardScaler; OneHotEncoder; Streamlit; GitHub; Streamlit Community Cloud.

## Methodology
1. Load the Telco Customer Churn dataset.
2. Convert `TotalCharges` to numeric and remove invalid rows.
3. Select tenure, contract type and monthly charges.
4. Scale numeric variables and encode contract categories.
5. Split data into training and test sets.
6. Train Logistic Regression and Random Forest models.
7. Evaluate accuracy, precision, recall for churn and confusion matrix.
8. Create a Streamlit app for individual customer predictions.

## Evaluation
The app calculates evaluation metrics from the held-out test set at runtime.

## Demonstration
Live app link: insert after deployment.
Video link: insert after recording and uploading.

## Conclusion
This project demonstrates a supervised classification workflow including data cleaning, preprocessing, model comparison, evaluation and deployment. Predictions are estimates, not guarantees.
