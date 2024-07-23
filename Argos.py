import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# RiskManagement class to calculate volatility
class RiskManagement:
    def __init__(self, data):
        self.data = data

    def calculate_volatility(self):
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=20).std()

    def plot_volatility(self):
        fig = px.line(self.data, x=self.data.index, y='Volatility', title='Volatility Over Time')
        st.plotly_chart(fig)

# RiskPrediction class for risk prediction
class RiskPrediction:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.model = None
        self.prediction = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        self.data['Price_Direction'] = np.where(self.data['Close'].shift(1) > self.data['Close'], 1, 0)
        self.data.dropna(inplace=True)
        self.X = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.y = self.data['Price_Direction']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        self.prediction = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, self.prediction)
        st.write(f"Accuracy: {accuracy:.3f}")
        st.write("Classification Report:")
        report = classification_report(self.y_test, self.prediction, output_dict=True)
        st.write("Precision:")
        st.write(f"Class 0: {report['0']['precision']:.3f}")
        st.write(f"Class 1: {report['1']['precision']:.3f}")
        st.write("Recall:")
        st.write(f"Class 0: {report['0']['recall']:.3f}")
        st.write(f"Class 1: {report['1']['recall']:.3f}")
        st.write("F1 Score:")
        st.write(f"Class 0: {report['0']['f1-score']:.3f}")
        st.write(f"Class 1: {report['1']['f1-score']:.3f}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(self.y_test, self.prediction))

    def save_plot(self):
        fig = px.line(self.X_test, x=self.X_test.index, y=self.prediction, title='Price Direction Prediction')
        st.plotly_chart(fig)

# Streamlit App
st.title("Argos: Algorithmic Trading Risk Analysis System")

# File uploader
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    # List the columns
    st.header("Dataset Columns:")
    st.write(data.columns)

    # Create RiskManagement instance
    risk_management = RiskManagement(data)
    risk_management.calculate_volatility()
    st.header("Volatility Over Time")
    risk_management.plot_volatility()

    # Create RiskPrediction instance
    risk_prediction = RiskPrediction(data)
    risk_prediction.prepare_data()
    risk_prediction.train_model()
    st.header("Model Evaluation")
    risk_prediction.evaluate_model()
    st.header("Price Direction Prediction")
    risk_prediction.save_plot()
else:
    st.write("Please upload a CSV file.")
