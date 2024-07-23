# Importing necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    def prepare_data(self):
        self.data['Price_Direction'] = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)
        self.data.dropna(inplace=True)
        self.X = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.y = self.data['Price_Direction']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.prediction = self.model.predict(self.X_test)

    def evaluate_model(self):
        accuracy = accuracy_score(self.y_test, self.prediction)
        st.write(f"Accuracy: {accuracy}")

    def save_plot(self):
        fig = px.line(self.X_test, x=self.X_test.index, y=self.prediction, title='Price Direction Prediction')
        st.plotly_chart(fig)

# Streamlit App
st.title("Argos: Algorithmic Trading Safeguard System")

# Load data
data = pd.read_csv('stock_data.csv')

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
