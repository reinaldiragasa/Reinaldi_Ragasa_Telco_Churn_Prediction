import streamlit as st
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, pipeline, threshold=0.208):
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas > self.threshold).astype(int)

st.title('Survival Rate Predictor')
st.text('This web can be used to predict customer churn rate')

st.sidebar.header("Please input your features")

def create_user_input():
    tenure = st.sidebar.slider('tenure', min_value=0.0, max_value=72.0, value=20.0, step=0.1)
    MonthlyCharges = st.sidebar.slider('MonthlyCharges', min_value=18.8, max_value=118.65, value=25.0, step=0.1)

    Dependents = st.sidebar.radio('Dependents', ['No', 'Yes'])
    OnlineSecurity = st.sidebar.radio('OnlineSecurity', ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.sidebar.radio('OnlineBackup', ['Yes', 'No', 'No internet service'])
    InternetService = st.sidebar.radio('InternetService', ['DSL', 'Fiber optic', 'No'])
    DeviceProtection = st.sidebar.radio('DeviceProtection', ['Yes', 'No', 'No internet service'])
    TechSupport = st.sidebar.radio('TechSupport', ['Yes', 'No', 'No internet service'])
    Contract = st.sidebar.radio('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.radio('PaperlessBilling', ['Yes', 'No'])

    user_data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'Dependents': Dependents,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'InternetService': InternetService,
        'DeviceProtection':DeviceProtection,
        'TechSupport':TechSupport,
        'Contract':Contract,
        'PaperlessBilling':PaperlessBilling
    }

    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

data_customer = create_user_input()
data_customer.index = ['Value']

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

with open('best_model_lgbm_threshold.sav', 'rb') as f:
    model_loaded = pickle.load(f)

classification = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]

with col2:
    st.subheader('Prediction Result')
    if classification == 1:
        st.write('Class 1: This customer will Churn')
    else:
        st.write('Class 2: This customer will not Churn')

    st.write(f"Probability of Churn: {probability[1]:.2f}")

