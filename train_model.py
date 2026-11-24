import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
URL = 'https://raw.githubusercontent.com/IBM/Customer-Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(URL)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
X = df[['tenure', 'Contract', 'MonthlyCharges']]
y = (df['Churn'].str.strip().str.lower() == 'yes').astype(int)
prep = ColumnTransformer([('num', StandardScaler(), ['tenure', 'MonthlyCharges']), ('cat', OneHotEncoder(handle_unknown='ignore'), ['Contract'])])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
for name, model in [('Logistic Regression', LogisticRegression(max_iter=1000)), ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))]:
    pipe = Pipeline([('prep', prep), ('model', model)]); pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
    print('\n', name); print(classification_report(y_test, pred)); print(confusion_matrix(y_test, pred))
