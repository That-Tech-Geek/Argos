import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Utilizing Machine Learning for Risk Prediction
class RiskPrediction:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.model = None
        self.prediction = None
        self.y_test = None

    def evaluate_model(self):
        accuracy = accuracy_score(self.y_test, self.prediction)
        st.write(f"Accuracy: {accuracy:.3f}")
        st.write("Classification Report:")
        st.write(classification_report(self.y_test, self.prediction))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(self.y_test, self.prediction))
        st.write("Error Metrics:")
        st.write(f"Precision: {precision_score(self.y_test, self.prediction):.3f}")
        st.write(f"Recall: {recall_score(self.y_test, self.prediction):.3f}")
        st.write(f"F1 Score: {f1_score(self.y_test, self.prediction):.3f}")

    def train_model(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.prediction = self.model.predict(self.X_test)

    def evaluate_model(self):
        accuracy = accuracy_score(self.y_test, self.prediction)
        st.write(f"Accuracy: {accuracy:.3f}")
        st.write("Classification Report:")
        st.write(classification_report(self.y_test, self.prediction))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(self.y_test, self.prediction))

    def save_plot(self):
        fig = px.line(self.X_test, x=self.X_test.index, y=self.prediction, title='Price Direction Prediction')
        st.plotly_chart(fig)

# Streamlit App
st.title("Argos: Algorithmic Trading Safeguard System")

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
