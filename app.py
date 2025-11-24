import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Model
# -------------------------------
model_path = "/mnt/data/employee_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload employee_model.pkl")
    st.stop()

# -------------------------------
# Load Dataset (For Department Options)
# -------------------------------
data_path = "/mnt/data/Employee_clean_data.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("❌ Data file not found. Please upload Employee_clean_data.csv")
    st.stop()

# Extract unique departments
departments = sorted(df["Department"].unique())

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Employee Feedback Prediction App")

st.write("Enter employee details below to predict Feedback:")

age = st.number_input("Age", min_value=18, max_value=70, step=1)
experience = st.number_input("Experience (years)", min_value=0, max_value=40, step=1)
salary = st.number_input("Salary", min_value=5000, max_value=500000, step=1000)

department = st.selectbox("Department", departments)

# Convert department to numeric (same encoding as your dataset)
try:
    dept_mapping = {dept: i for i, dept in enumerate(departments)}
    dept_encoded = dept_mapping[department]
except:
    st.error("Department encoding failed.")
    st.stop()

# Prepare input for model
input_data = [[age, experience, salary, dept_encoded]]

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Feedback"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Feedback: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
