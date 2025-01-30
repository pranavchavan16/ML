import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load trained model and scaler
rf = joblib.load("rice_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for feature selection (same as training data)
data = pd.read_excel("Rice_Cammeo_Osmancik.xlsx")
X_train = data.drop(columns=["Class"])
y_train = data["Class"]

# Fit feature selector
selector = SelectKBest(f_classif, k=5)
selector.fit(X_train, y_train)

# Calculate accuracy of the model on training data
train_predictions = rf.predict(scaler.transform(selector.transform(X_train)))
accuracy = accuracy_score(y_train, train_predictions)

# Streamlit UI
st.title("🌾 Rice Classification App")
st.write("This application classifies rice varieties between **Cammeo** and **Osmancik**.")
st.write(f"### Model Accuracy: **{accuracy * 100:.2f}%**")

# File Upload Option
uploaded_file = st.file_uploader("📥 Upload your rice dataset (Excel file)", type=["xlsx"])

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file)

    # Remove "Class" column if present
    if "Class" in input_df.columns:
        input_df = input_df.drop(columns=["Class"])

    # Ensure correct feature selection and scaling
    input_features = selector.transform(input_df)
    input_scaled = scaler.transform(input_features)

    # Predict
    predictions = rf.predict(input_scaled)
    input_df["Predicted Class"] = predictions

    st.write("### Predictions from File Upload:")
    st.dataframe(input_df)

# Manual Input Form
st.write("OR Enter Values Manually:")
manual_input = {}
for col in X_train.columns:
    manual_input[col] = st.number_input(f"Enter value for **{col}**", value=float(X_train[col].mean()))

# Predict button
if st.button("🔍 Predict Manually"):
    manual_df = pd.DataFrame([manual_input])
    manual_features = selector.transform(manual_df)
    manual_scaled = scaler.transform(manual_features)
    manual_prediction = rf.predict(manual_scaled)

    st.write(f"### Predicted Class: **{manual_prediction[0]}**")