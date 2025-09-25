import streamlit as st
import numpy as np
import pandas as pd
import pickle

# load model
with open("xgboost_tuned_model.pkl", "rb") as file:
    model = pickle.load(file)

#generate feature dataframe based on user input
def create_input_df(age, income, gender, emp_exp, loan_amnt, int_rate, percent_income, cred_hist_length,
                    education, loan_intent, home_ownership, previous_default, credit_score):

    # Replicate LabelEncoder-style integer encoding used during training (sorted classes)
    education_map = {
        'Associate': 0,
        'Bachelor': 1,
        'Doctorate': 2,
        'High School': 3,
        'Master': 4
    }
    loan_intent_map = {
        'Debt Consolidation': 0,
        'Education': 1,
        'Home Improvement': 2,
        'Medical': 3,
        'Personal': 4,
        'Venture': 5
    }
    home_ownership_map = {
        'Mortgage': 0,
        'Own': 1,
        'Rent': 2
    }

    # Assume training used lowercased gender with LabelEncoder: female=0, male=1
    gender_code = 1 if gender == 'Male' else 0

    # Map previous defaults: assume No=0, Yes=1
    prev_default_code = 1 if previous_default == 'Yes' else 0

    input_dict = {
        'person_age': [age],
        'person_gender': [gender_code],
        'person_education': [education_map[education]],
        'person_income': [income],
        'person_emp_exp': [emp_exp],
        'person_home_ownership': [home_ownership_map[home_ownership]],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent_map[loan_intent]],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [percent_income],
        'cb_person_cred_hist_length': [cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [prev_default_code]
    }

    return pd.DataFrame(input_dict)

# streamlit UI
st.title("Loan Default Prediction")

col1, col2 = st.columns(2)

with col1:
    person_age = st.slider("Age", 18, 100, 30)
    person_income = st.number_input("Annual Income ($)", 1000, step=500, value=50000)
    person_home_ownership = st.selectbox("Home Ownership", ['Mortgage', 'Own', 'Rent'])
    person_education = st.selectbox("Education Level", ['Associate', 'Bachelor', 'Doctorate', 'High School', 'Master'])
    person_gender = st.selectbox("Gender", ['Male', 'Female'])
    cred_hist_length = st.slider('Credit History Length (years)', 0, 30, 5)

with col2:
    person_emp_exp = st.slider("Employment Experience (years)", 0, 80, 5)
    loan_intent = st.selectbox("Loan Intent", ['Debt Consolidation', 'Education', 'Home Improvement', 'Medical', 'Personal', 'Venture'])
    loan_amnt = st.number_input("Loan Amount ($)", 500, step=100, value=10000)
    loan_int_rate = st.slider("Interest Rate (%)", 0.0, 50.0, 12.0, step=0.1)
    loan_percent_income = st.slider("Loan % of Income", 0.0, 2.0, 0.2, step=0.01)
    previous_loan_defaults = st.selectbox("Previous Loan Defaults", ['Yes', 'No'])
    credit_score = st.number_input("Credit Score", 300, 850, 650)

# button to predict
if st.button("Predict Loan Approval"):
    input_data = create_input_df(person_age, person_income, person_gender, person_emp_exp, loan_amnt,
                                 loan_int_rate, loan_percent_income, cred_hist_length,
                                 person_education, loan_intent, person_home_ownership, previous_loan_defaults, credit_score)

    prediction = model.predict(input_data)[0]
    result_text = "**Loan is likely to be APPROVED.**" if prediction == 1 else "**Loan is likely to be REJECTED.**"

    st.subheader("Prediction Result")
    st.write(result_text)
    st.info("This prediction is based on historical data and created model.")