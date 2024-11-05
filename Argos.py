import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# Custom RSI calculation function with proper indexing
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Ensure the index of the stock data is used in the Series
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Placeholder financial ratios calculation function
def calculate_financial_ratios(ticker):
    # This function would normally fetch financial data for the stock
    # and calculate various financial ratios. For this example, we'll return a fixed score.
    return 50  # Dummy score for financial ratios

# Placeholder candlestick analysis function
def perform_candlestick_analysis(stock_data):
    # Quantitative analysis of candlestick patterns would be implemented here
    # We'll return a fixed score as a placeholder.
    return 50  # Dummy score for candlestick patterns

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
    
    # Fetch VIX data for different time frames (1 year, 1 month, 1 day, 1 hr, 1 min)
    vix_data = {
        '1y': yf.download('^VIX', period='1y', interval='1d'),
        '1m': yf.download('^VIX', period='1mo', interval='1d'),
        '1d': yf.download('^VIX', period='1d', interval='5m'),
        '1h': yf.download('^VIX', period='1d', interval='1h'),
        '1min': yf.download('^VIX', period='1d', interval='1m')
    }

    # Calculate RSI using the custom function
    stock_data['RSI'] = calculate_rsi(stock_data)
    overbought = len(stock_data[stock_data['RSI'] > 70]) / len(stock_data) * 100
    oversold = len(stock_data[stock_data['RSI'] < 30]) / len(stock_data) * 100
    
    # Financial ratios and candlestick analysis (using placeholder functions)
    financial_ratios = calculate_financial_ratios(ticker)
    candlestick_analysis = perform_candlestick_analysis(stock_data)

    # Scoring logic (for simplicity, assign each component equal weight)
    score_open = (stock_data['7d_open'].iloc[-1] + stock_data['50d_open'].iloc[-1] + stock_data['200d_open'].iloc[-1]) / 3
    score_close = (stock_data['7d_close'].iloc[-1] + stock_data['50d_close'].iloc[-1] + stock_data['200d_close'].iloc[-1]) / 3
    score_vix = np.mean([vix_data['1y']['Close'].iloc[-1], vix_data['1m']['Close'].iloc[-1], vix_data['1d']['Close'].iloc[-1], 
                         vix_data['1h']['Close'].iloc[-1], vix_data['1min']['Close'].iloc[-1]])
    
    # Final score calculation
    overall_score = 0.3 * score_open + 0.3 * score_close + 0.2 * score_vix + 0.1 * (overbought - oversold) + 0.1 * (financial_ratios + candlestick_analysis)

    return overall_score

# Streamlit interface
st.title("Stock Scoring Application")

# Ticker input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, etc.):")

if ticker:
    score = get_stock_score(ticker)
    st.write(f"Overall score for {ticker}: {score}")
