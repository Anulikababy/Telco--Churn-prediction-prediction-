import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

features = ["tenure", "Contract", "MonthlyCharges"]
X = df[features]
y = df["Churn"].map({"No": 0, "Yes": 1})

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), ["tenure", "MonthlyCharges"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Contract"]),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
]).fit(X_train, y_train)

forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
]).fit(X_train, y_train)

eval_results = {}
for name, model in [("Logistic Regression", logistic_model), ("Random Forest", forest_model)]:
    preds = model.predict(X_test)
    eval_results[name] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "cm": confusion_matrix(y_test, preds),
    }

joblib.dump(logistic_model, "logistic_churn_model.pkl")
joblib.dump(forest_model, "forest_churn_model.pkl")
joblib.dump(eval_results, "eval_results.pkl")
print("Models saved successfully.")
