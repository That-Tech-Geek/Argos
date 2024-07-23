import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Set title of the Streamlit app
st.title("Stock Risk Prediction")

# Create a file uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False, key="fileUploader")

# Load the uploaded file
if uploaded_file is not None:
    stock_data = pd.read_csv(uploaded_file)

    # Display the data
    st.write("Data:")
    st.write(stock_data)

    # Split the data into features and target
    X = stock_data.drop(["Risk"], axis=1)
    y = stock_data["Risk"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a neural network model
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display the predictions
    st.write("Predictions:")
    st.write(y_pred)
