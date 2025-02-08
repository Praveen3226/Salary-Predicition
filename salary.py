import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load(r"C:\Users\viper\Desktop\NOTES\AI ML DS\salary_prediction.pkl")


df = pd.read_csv(r"C:\Users\viper\Desktop\NOTES\AI ML DS\sl.csv")


le_edu = LabelEncoder()
df["Education Level"] = df["Education Level"].fillna("Bachelor's")
df["Education Level Encoded"] = le_edu.fit_transform(df["Education Level"])

le_job = LabelEncoder()
df["Job Title"] = df["Job Title"].fillna("Software Engineer")
df["Job Title Encoded"] = le_job.fit_transform(df["Job Title"])


st.title("Salary Prediction App 💰")


education_levels = df["Education Level"].unique()
education_input = st.selectbox("Select Education Level:", education_levels)
education_encoded = le_edu.transform([education_input])[0]


job_titles = df["Job Title"].unique()
job_input = st.selectbox("Select Job Title:", job_titles)
job_encoded = le_job.transform([job_input])[0]


experience = st.number_input("Enter Years of Experience:", min_value=0, max_value=50, value=1)


if st.button("Predict Salary 💵"):
    prediction = model.predict([[education_encoded, job_encoded, experience]])
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f} /Mon")

# python -m streamlit run salary.py
