import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title='Customer Churn Prediction', page_icon='📉')
st.title('📉 Customer Churn Prediction')
st.write('Predict whether a telecom customer may leave or stay using tenure, contract type and monthly charges.')
DATA_URL = 'https://raw.githubusercontent.com/IBM/Customer-Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df.dropna(subset=['TotalCharges'])

@st.cache_resource
def train_models(df):
    X = df[['tenure', 'Contract', 'MonthlyCharges']]
    y = (df['Churn'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    prep = ColumnTransformer([('num', StandardScaler(), ['tenure', 'MonthlyCharges']), ('cat', OneHotEncoder(handle_unknown='ignore'), ['Contract'])])
    lr = Pipeline([('prep', prep), ('model', LogisticRegression(max_iter=1000))])
    rf = Pipeline([('prep', prep), ('model', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    lr.fit(X_train, y_train); rf.fit(X_train, y_train)
    results = {}
    for name, model in [('Logistic Regression', lr), ('Random Forest', rf)]:
        pred = model.predict(X_test)
        results[name] = {'accuracy': accuracy_score(y_test, pred), 'precision': precision_score(y_test, pred, zero_division=0), 'recall': recall_score(y_test, pred, zero_division=0), 'matrix': confusion_matrix(y_test, pred).tolist()}
    return lr, rf, results

try:
    df = load_data()
except Exception:
    st.error('Could not download the dataset. Please refresh or check your internet connection.')
    st.stop()
st.caption('Public Telco Customer Churn dataset')
st.write(f'Rows after cleaning: {len(df):,}')
lr, rf, metrics = train_models(df)
st.subheader('Model Evaluation on Test Data')
for name, values in metrics.items():
    st.markdown(f'**{name}**')
    c1, c2, c3 = st.columns(3)
    c1.metric('Accuracy', f"{values['accuracy']:.1%}")
    c2.metric('Precision (churn)', f"{values['precision']:.1%}")
    c3.metric('Recall (churn)', f"{values['recall']:.1%}")
    st.write('Confusion matrix [TN, FP; FN, TP]:', values['matrix'])
st.divider(); st.subheader('Try a Customer Prediction')
tenure = st.slider('Tenure (months)', 0, 72, 12)
contract = st.selectbox('Contract type', ['Month-to-month', 'One year', 'Two year'])
monthly = st.number_input('Monthly charges ($)', min_value=0.0, max_value=200.0, value=70.0, step=1.0)
if st.button('Predict Customer Churn', type='primary'):
    sample = pd.DataFrame([{'tenure': tenure, 'Contract': contract, 'MonthlyCharges': monthly}])
    prob = float(lr.predict_proba(sample)[0][1])
    if prob >= .5: st.warning(f'Likely to leave. Estimated churn probability: {prob:.1%}')
    else: st.success(f'Likely to stay. Estimated probability of staying: {(1-prob):.1%}')
    st.caption('This prediction is an estimate, not a guarantee.')
