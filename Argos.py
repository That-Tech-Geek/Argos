import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate financial ratios
def calculate_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.financials
    ratios = {}
    
    # Example financial ratios
    ratios['PE Ratio'] = stock.info.get('forwardPE', np.nan)  # Forward P/E ratio
    ratios['ROE'] = stock.info.get('returnOnEquity', np.nan)  # Return on Equity
    ratios['Debt to Equity'] = stock.info.get('debtToEquity', np.nan)  # Debt to Equity ratio
    ratios['Current Ratio'] = stock.info.get('currentRatio', np.nan)  # Current Ratio
    
    # You can add more ratios here based on available financial data
    return ratios

# Function to perform candlestick analysis
def perform_candlestick_analysis(stock_data):
    # For simplicity, let's analyze the last few candlestick patterns
    patterns = {
        'Bullish Engulfing': (stock_data['Open'].iloc[-2] > stock_data['Close'].iloc[-2] and 
                             stock_data['Close'].iloc[-1] > stock_data['Open'].iloc[-1] and 
                             stock_data['Open'].iloc[-1] < stock_data['Close'].iloc[-2]),
        'Bearish Engulfing': (stock_data['Open'].iloc[-2] < stock_data['Close'].iloc[-2] and 
                              stock_data['Close'].iloc[-1] < stock_data['Open'].iloc[-1] and 
                              stock_data['Open'].iloc[-1] > stock_data['Close'].iloc[-2]),
    }

    return {k: v for k, v in patterns.items() if v}

# Main function to fetch data, calculate metrics, and return a stock score
def get_stock_score(ticker):
    # Download stock data
    stock_data = yf.download(ticker, period="1y", interval="1d")

    # Ensure data is not empty
    if stock_data.empty:
        return "No data available for ticker"
    
    # Calculate Rolling Means
    stock_data['7d_open'] = stock_data['Open'].rolling(window=7).mean()
    stock_data['50d_open'] = stock_data['Open'].rolling(window=50).mean()
    stock_data['200d_open'] = stock_data['Open'].rolling(window=200).mean()
    stock_data['7d_close'] = stock_data['Close'].rolling(window=7).mean()
    stock_data['50d_close'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200d_close'] = stock_data['Close'].rolling(window=200).mean()

    # Fetch VIX data for different time frames
    vix_data = {
        '1y': yf.download('^VIX', period='1y', interval='1d'),
        '1m': yf.download('^VIX', period='1mo', interval='1d'),
        '1d': yf.download('^VIX', period='1d', interval='5m'),
        '1h': yf.download('^VIX', period='1d', interval='1h'),
        '1min': yf.download('^VIX', period='1d', interval='1m')
    }

    # Calculate RSI
    stock_data['RSI'] = calculate_rsi(stock_data)
    overbought = len(stock_data[stock_data['RSI'] > 70]) / len(stock_data) * 100
    oversold = len(stock_data[stock_data['RSI'] < 30]) / len(stock_data) * 100

    # Financial ratios and candlestick analysis
    financial_ratios = calculate_financial_ratios(ticker)
    candlestick_analysis = perform_candlestick_analysis(stock_data)

    # Scoring logic
    score_open = (stock_data['7d_open'].iloc[-1] + stock_data['50d_open'].iloc[-1] + stock_data['200d_open'].iloc[-1]) / 3
    score_close = (stock_data['7d_close'].iloc[-1] + stock_data['50d_close'].iloc[-1] + stock_data['200d_close'].iloc[-1]) / 3
    score_vix = np.mean([vix_data['1y']['Close'].iloc[-1], vix_data['1m']['Close'].iloc[-1], vix_data['1d']['Close'].iloc[-1], 
                         vix_data['1h']['Close'].iloc[-1], vix_data['1min']['Close'].iloc[-1]])

    # Combine the scores with weights
    overall_score = 0.3 * score_open + 0.3 * score_close + 0.2 * score_vix + 0.1 * (overbought - oversold) + 0.1 * len(candlestick_analysis)

    return overall_score, financial_ratios, candlestick_analysis

# Streamlit interface
st.title("Stock Scoring System")

# Input for ticker symbol
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):")

if st.button("Get Stock Score"):
    if ticker:
        score, financial_ratios, candlestick_analysis = get_stock_score(ticker)
        st.write(f"Overall score for {ticker}: {score}")
        st.write("Financial Ratios:")
        for key, value in financial_ratios.items():
            st.write(f"{key}: {value}")
        st.write("Recent Candlestick Patterns:")
        for pattern in candlestick_analysis:
            st.write(pattern)
    else:
        st.error("Please enter a valid ticker symbol.")
