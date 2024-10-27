import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("/content/cleaned_financial_inclusion dataset.csv")

# Define features and target
X = data.drop(columns='bank_account')
y = data['bank_account']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app code
st.title("Financial Inclusion Prediction App")
st.write("This app predicts the likelihood of owning a bank account based on user input.")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.2f}")

# Generate user input fields based on features
st.header("Enter the following features:")

user_input = {}
for column in X.columns:
    # Dynamically create input fields based on column type
    if X[column].dtype == 'object':  # Categorical feature
        user_input[column] = st.selectbox(f"{column}:", X[column].unique())
    else:  # Numerical feature
        user_input[column] = st.number_input(f"{column}:", min_value=float(X[column].min()), max_value=float(X[column].max()))

# Predict button
if st.button("Predict"):
    # Convert user input to DataFrame for prediction
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    
    # Display prediction result
    if prediction == 1:
        st.write("The model predicts that the individual **owns a bank account**.")
    else:
        st.write("The model predicts that the individual **does not own a bank account**.")
