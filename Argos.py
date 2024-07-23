import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Set title of the Streamlit app
st.title("Stock Risk Prediction")

# Create a file uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False, key="fileUploader")

# Load the uploaded file
if uploaded_file is not None:
    stock_data = pd.read_csv(uploaded_file)

    # Preprocess data
    scaler = MinMaxScaler()
    columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Average Total Assets', 'Asset Turnover Ratio', 
                        'EBIT', 'Interest Rate', 'Corporate Tax', 'Debt-to-Equity Ratio', 'Current Ratio', 'Interest Coverage Ratio', 
                        'Debt-to-Capital Ratio', 'Price-to-Earnings Ratio', 'Price-to-Book Ratio', 'Return on Equity (ROE)', 
                        'Return on Assets (ROA)', 'Earnings Yield', 'Dividend Yield', 'Price-to-Sales Ratio', 
                        'Enterprise Value-to-EBITDA Ratio', 'Inventory Turnover Ratio', 'Receivables Turnover Ratio', 
                        'Payables Turnover Ratio', 'Cash Conversion Cycle', 'Debt Service Coverage Ratio', 'Return on Invested Capital (ROIC)', 
                        'Return on Common Equity (ROCE)', 'Gross Margin Ratio', 'Operating Margin Ratio', 'Net Profit Margin Ratio', 
                        'Debt to Assets Ratio', 'Equity Ratio', 'Financial Leverage Ratio', 'Proprietary Ratio', 'Capital Gearing Ratio', 
                        'DSCR', 'Gross Profit Ratio', 'Net Profit Ratio', 'ROI', 'EBITDA Margin', 'Fixed Asset Turnover Ratio', 
                        'Capital Turnover Ratio']
    stock_data[columns_to_scale] = scaler.fit_transform(stock_data[columns_to_scale])

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(stock_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train model
    model.fit(stock_data, epochs=50, batch_size=1, verbose=2)

    # Make predictions
    predictions = model.predict(stock_data)

    # Display predictions
    st.write("Predictions:")
    st.write(predictions)
