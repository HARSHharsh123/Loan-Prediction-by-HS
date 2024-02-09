import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt

st.title("Loan Approval Prediction")

# Taking Input from the User
no_of_dependents = st.selectbox("Enter No of Dependents", options=["0", "1", "2", "3", "4", "5"])
education = st.selectbox("Enter Education", options=['Graduate', 'Not Graduate'])
self_employed = st.selectbox("self_employed", options=['Yes', 'No'])
income_annum = st.number_input("Enter Annual Income of the Applicant")
loan_amount = st.number_input("Enter Loan Amount of the Applicant")
loan_term = st.number_input("Enter Loan Term in years")
cibil_score = st.number_input("Enter Credit Score of the Applicant")
residential_assets_value = st.number_input("Enter Residential assets of the Applicant")
commercial_assets_value = st.number_input("Enter Commercial assets of the Applicant")
luxury_assets_value = st.number_input("Enter Luxury assets of the Applicant")
bank_asset_value = st.number_input("Enter Bank Asset of the Applicant")

# Load the pre-trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

def prediction(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
               residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value):
    input_test = np.array(
        [no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
         residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value],
        dtype=object).reshape(1, 11)
    columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term',
               'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
               'bank_asset_value']
    input_test_df = pd.DataFrame(input_test, columns=columns)

    result = pipe.predict(input_test_df)
    feature_importance = pipe.named_steps['Prediction'].feature_importances_
    feature_importance_dict = dict(zip(columns, feature_importance))

    return result[0], feature_importance_dict


## Predict Button

pre = st.button("Predict")

## Prediction Logic if result == 0 -> loan to be given and if result == 1 then no loan to be given
if pre:
    result, feature_importance_dict = prediction(no_of_dependents, education, self_employed, income_annum, loan_amount,
                                                 loan_term, cibil_score, residential_assets_value,
                                                 commercial_assets_value, luxury_assets_value, bank_asset_value)

    if result == 0:
        st.write("Congratulations 🎉🎊 !!! , Your Loan has been Approved by the Bank")
    else:
        st.write("Sorry to Say 🥲🥲 but you are not eligible for this loan")

        # Sort features by importance in descending order
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        st.subheader("Feature Importance : (Model Interpretability) ")

        # Display the top features that contributed to the rejection
        st.write("Top features contributing to rejection:")
        for feature, importance in sorted_features[:3]:  # Display top 3 features
            st.write(f"{feature}: {importance}")