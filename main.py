import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle

# load the saved model
st.set_page_config(page_title="Telecom Churn Rate Predictor", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")

def main():
    def prediction_output(transformed_data):
        with open('new_model.pkl', 'rb') as file:
            model = pickle.load(file)
            
        prediction = model.predict(transformed_data)
        if prediction[0] > 0.5:
            st.error("User is likely to churn.")
        else:
            st.success("User is likely to continue.")

    # functions for transforming data
    def transform_data(data):
        df = pd.DataFrame(data)
        
        # one hot encoding
        cols_to_encode = ["contract", "internetservice", "paymentmethod"]
        encoder = OneHotEncoder(categories=[
            ["Month-to-month", "One year", "Two year"],
            ["DSL", "Fiber optic", "No"],
            ["Bank transfer (automatic)", "Credit card (automatic)","Electronic check", "Mailed check"]
        ], handle_unknown="ignore", sparse_output=False)
        encoded_features = encoder.fit_transform(df[cols_to_encode])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(input_features=cols_to_encode))
        df = pd.concat([df.drop(cols_to_encode, axis=1), encoded_df], axis=1)
        
        # scaling
        cols_to_scale = ["tenure", "monthlycharges", "totalcharges"]
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        # rounding to 7 decimal places
        df[cols_to_scale] = df[cols_to_scale].round(7)
        
        # Ensure all columns are visible
        pd.set_option('display.max_columns', 26)
        
        return df

    st.title("Telecom Churn :red[Rate Predictor :] ")
    st.write("**This app predicts the likelihood of a user to churn based on the input features.**")

    st.info("""
    ### About this web app:
    - **This web app is a churn rate predictor**.
    - **It uses a trained deep learning model to predict the likelihood of a user to churn**.
    - **The model was trained on the Telco Customer Churn dataset**.
    - **Made by Bhuvan, Vaishnavi, Gowtham, Shreya.**
    """, icon="ℹ️")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Select Gender:", options=["Male", "Female"])
        with col2:
            senior_citizen = st.selectbox("Senior Citizen:", options=["Yes", "No"])

        col1, col2 = st.columns(2)
        with col1:
            partner = st.selectbox("Partner (Does user have a partner):", options=["Yes", "No"])
        with col2:
            dependents = st.selectbox("Dependents (Does user depend on anyone):", options=["Yes", "No"])

        tenure = st.number_input("Tenure (How long user associated with the company):", min_value=1, max_value=100, value=1, step=1)

    with st.container(border=True):
        col3, col4 = st.columns(2)
        with col3:
            phone_service = st.selectbox("Phone Service:", options=["Yes", "No"])
        with col4:
            multiple_lines = st.selectbox("Multiple Lines:", options=["Yes", "No"])

        internet_service = st.selectbox("Internet Service:", options=["DSL", "Fiber optic", "No"])

    with st.container(border=True):
        col5, col6, col7 = st.columns(3)
        with col5:
            online_security = st.selectbox("Online Security:", options=["Yes", "No"])
        with col6:
            online_backup = st.selectbox("Online Backup:", options=["Yes", "No"])
        with col7:
            device_protection = st.selectbox("Device Protection:", options=["Yes", "No"])

        col8, col9, col10 = st.columns(3)
        with col8:
            tech_support = st.selectbox("Tech Support:", options=["Yes", "No"])
        with col9:
            streaming_tv = st.selectbox("Streaming TV:", options=["Yes", "No"])
        with col10:
            streaming_movies = st.selectbox("Streaming Movies:", options=["Yes", "No"])

        col11, col12, col13 = st.columns(3)
        with col11:
            contract = st.selectbox("Contract (Billing Plan):", options=["Month-to-month", "One year", "Two year"])
        with col12:
            paperless_billing = st.selectbox("Paperless Billing:", options=["Yes", "No"])
        with col13:
            payment_method = st.selectbox("Payment Method:", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    with st.container(border=True):
        col14, col15 = st.columns(2)
        with col14:
            monthly_charges = st.number_input("Monthly Charges (1.0 - 100.0):", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
        with col15:
            total_charges = st.number_input("Total Charges (1.0 - 1000.0):", min_value=1.0, max_value=1000.0, value=1.0, step=0.1)
        

    data = {
        "gender": [1 if gender == "Male" else 0],
        "seniorcitizen": [1 if senior_citizen == "Yes" else 0],
        "partner": [1 if partner == "Yes" else 0],
        "dependents": [1 if dependents == "Yes" else 0],
        "tenure": [tenure],
        "phoneservice": [1 if phone_service == "Yes" else 0],
        "multiplelines": [1 if multiple_lines == "Yes" else 0],
        "internetservice": [internet_service],
        "onlinesecurity": [1 if online_security == "Yes" else 0],
        "onlinebackup": [1 if online_backup == "Yes" else 0],
        "deviceprotection": [1 if device_protection == "Yes" else 0],
        "techsupport": [1 if tech_support == "Yes" else 0],
        "streamingtv": [1 if streaming_tv == "Yes" else 0],
        "streamingmovies": [1 if streaming_movies == "Yes" else 0],
        "contract": [contract],
        "paperlessbilling": [1 if paperless_billing == "Yes" else 0],
        "paymentmethod": [payment_method],
        "monthlycharges": [monthly_charges],
        "totalcharges": [total_charges]
    }

    transformed_data = transform_data(data)
    # print("Transformed Data:", transformed_data)
    if st.button("Predict"):
        prediction_output(transformed_data)
        # st.dataframe(transformed_data)

if __name__ == "__main__":
    main()
    