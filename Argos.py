import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate financial ratios
def calculate_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    ratios = {}

    # Example financial ratios
    ratios['PE Ratio'] = stock.info.get('forwardPE', np.nan)  # Forward P/E ratio
    ratios['ROE'] = stock.info.get('returnOnEquity', np.nan)  # Return on Equity
    ratios['Debt to Equity'] = stock.info.get('debtToEquity', np.nan)  # Debt to Equity ratio
    ratios['Current Ratio'] = stock.info.get('currentRatio', np.nan)  # Current Ratio

    return ratios

# Function to perform candlestick analysis
def perform_candlestick_analysis(stock_data):
    patterns = {}

    # Check for Bullish Engulfing pattern
    if len(stock_data) >= 2:
        if (
            (stock_data['Open'].iloc[-2] > stock_data['Close'].iloc[-2]).all() and
            (stock_data['Close'].iloc[-1] > stock_data['Open'].iloc[-1]).all() and
            (stock_data['Open'].iloc[-1] < stock_data['Close'].iloc[-2]).all()
        ):
            patterns['Bullish Engulfing'] = True

        # Check for Bearish Engulfing pattern
        if (
            (stock_data['Open'].iloc[-2] < stock_data['Close'].iloc[-2]).all() and
            (stock_data['Close'].iloc[-1] < stock_data['Open'].iloc[-1]).all() and
            (stock_data['Open'].iloc[-1] > stock_data['Close'].iloc[-2]).all()
        ):
            patterns['Bearish Engulfing'] = True

    return patterns

# Main function to fetch data, calculate metrics, and return a stock score
def get_stock_score(ticker):
    # Download stock data
    stock_data = yf.download(ticker, period="1y", interval="1d")

    # Ensure data is not empty
    if stock_data.empty:
        return "No data available for ticker", {}, {}

    # Calculate Rolling Means
    stock_data['7d_open'] = stock_data['Open'].rolling(window=7).mean()
    stock_data['50d_open'] = stock_data['Open'].rolling(window=50).mean()
    stock_data['200d_open'] = stock_data['Open'].rolling(window=200).mean()
    stock_data['7d_close'] = stock_data['Close'].rolling(window=7).mean()
    stock_data['50d_close'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200d_close'] = stock_data['Close'].rolling(window=200).mean()

    # Calculate RSI
    stock_data['RSI'] = calculate_rsi(stock_data)

    # Calculate VIX data for scoring (example using VIX)
    vix_data = yf.download('^VIX', period='1y', interval='1d')

    # Scoring logic
    score_open = (stock_data['7d_open'].iloc[-1] + stock_data['50d_open'].iloc[-1] + stock_data['200d_open'].iloc[-1]) / 3
    score_close = (stock_data['7d_close'].iloc[-1] + stock_data['50d_close'].iloc[-1] + stock_data['200d_close'].iloc[-1]) / 3
    score_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else np.nan

    # Calculate RSI impact on score (normalized)
    rsi_overbought = len(stock_data[stock_data['RSI'] > 70])
    rsi_oversold = len(stock_data[stock_data['RSI'] < 30])
    rsi_impact = (rsi_overbought - rsi_oversold) / len(stock_data) if len(stock_data) > 0 else 0

    # Combine the scores with weights
    overall_score = (
        0.3 * score_open +
        0.3 * score_close +
        0.2 * score_vix +
        0.1 * rsi_impact
    )

    # Ensure overall_score is a scalar value
    overall_score = float(overall_score) if isinstance(overall_score, pd.Series) else overall_score

    return overall_score, calculate_financial_ratios(ticker), perform_candlestick_analysis(stock_data)

    # Calculate Rolling Means
    stock_data['7d_open'] = stock_data['Open'].rolling(window=7).mean()
    stock_data['50d_open'] = stock_data['Open'].rolling(window=50).mean()
    stock_data['200d_open'] = stock_data['Open'].rolling(window=200).mean()
    stock_data['7d_close'] = stock_data['Close'].rolling(window=7).mean()
    stock_data['50d_close'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200d_close'] = stock_data['Close'].rolling(window=200).mean()

    # Calculate RSI
    stock_data['RSI'] = calculate_rsi(stock_data)

    # Calculate VIX data for scoring (example using VIX)
    vix_data = yf.download('^VIX', period='1y', interval='1d')

    # Scoring logic
    score_open = (stock_data['7d_open'].iloc[-1] + stock_data['50d_open'].iloc[-1] + stock_data['200d_open'].iloc[-1]) / 3
    score_close = (stock_data['7d_close'].iloc[-1] + stock_data['50d_close'].iloc[-1] + stock_data['200d_close'].iloc[-1]) / 3
    score_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else np.nan

    # Calculate RSI impact on score (normalized)
    rsi_overbought = len(stock_data[stock_data['RSI'] > 70])
    rsi_oversold = len(stock_data[stock_data['RSI'] < 30])
    rsi_impact = (rsi_overbought - rsi_oversold) / len(stock_data) if len(stock_data) > 0 else 0

    # Combine the scores with weights
    overall_score = (
        0.3 * score_open +
        0.3 * score_close +
        0.2 * score_vix +
        0.1 * rsi_impact
    )

    return overall_score, calculate_financial_ratios(ticker), perform_candlestick_analysis(stock_data)

# Streamlit interface
st.title("Stock Scoring System")

# Input for ticker symbol
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):")

if st.button("Get Stock Score"):
    if ticker:
        score, financial_ratios, candlestick_analysis = get_stock_score(ticker)
        st.write(f"Overall score for {ticker}: {score:.2f}")
        st.write("Financial Ratios:")
        for key, value in financial_ratios.items():
            st.write(f"{key}: {value}")
        st.write("Recent Candlestick Patterns:")
        for pattern in candlestick_analysis:
            st.write(pattern)
    else:
        st.error("Please enter a valid ticker symbol.")
