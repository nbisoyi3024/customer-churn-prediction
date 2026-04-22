import streamlit as st
import pandas as pd
from src.preprocessing import create_features
import joblib

# Load model and columns
model = joblib.load("models/pipeline.pkl")
#st.write(f"Model used: {best_model}")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn")

# ---------------- INPUTS ---------------- #

CreditScore = st.number_input("Credit Score", 300, 900, 650)
Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider("Age", 18, 100, 40)
Tenure = st.slider("Tenure", 0, 10, 5)
Balance = st.number_input("Balance", 0, 250000, 50000)
NumOfProducts = st.slider("Number of Products", 1, 4, 2)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", 0, 200000, 50000)
CardType = st.selectbox("Card Type", ["Silver", "Gold", "Platinum", "Diamond"])
PoinEarned = st.number_input("Point Earned",)
SatisfactionScore = st.number_input("Satisfaction Score",0,5)

# ---------------- PREDICT ---------------- #

if st.button("Predict Churn"):

    # Raw input dictionary
    input_data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Satisfaction Score":SatisfactionScore,
        "Card Type": CardType
    }
    input_df = pd.DataFrame([input_data])

    # apply feature engineering
    #input_df = create_features(input_df)
    #align columns
    #input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")