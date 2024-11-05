import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
import streamlit as st

# Define a function to fetch data, calculate rolling means, VIX data, RSI, and analyze financial ratios and candlestick patterns
def get_stock_score(ticker):
    # Download stock data
    stock_data = yf.download(ticker, period="1y", interval="1d")
    
    # Ensure data is not empty
    if stock_data.empty:
        st.error("No data available for this ticker")
        return None
    
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

    # Calculate RSI and % of time in overbought/oversold regions
    stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
    overbought = len(stock_data[stock_data['RSI'] > 70]) / len(stock_data) * 100
    oversold = len(stock_data[stock_data['RSI'] < 30]) / len(stock_data) * 100
    
    # Calculate Financial Ratios
    financial_ratios = calculate_financial_ratios(ticker)
    
    # Perform Candlestick Analysis
    candlestick_analysis = perform_candlestick_analysis(stock_data)

    # Scoring logic
    score_open = (stock_data['7d_open'][-1] + stock_data['50d_open'][-1] + stock_data['200d_open'][-1]) / 3
    score_close = (stock_data['7d_close'][-1] + stock_data['50d_close'][-1] + stock_data['200d_close'][-1]) / 3
    score_vix = np.mean([vix_data['1y']['Close'][-1], vix_data['1m']['Close'][-1], vix_data['1d']['Close'][-1], 
                         vix_data['1h']['Close'][-1], vix_data['1min']['Close'][-1]])
    
    # Overall Score
    overall_score = (0.3 * score_open + 0.3 * score_close + 0.2 * score_vix + 
                     0.1 * (overbought - oversold) + 0.1 * (financial_ratios + candlestick_analysis))

    return overall_score

# Function to calculate financial ratios
def calculate_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    try:
        pe_ratio = stock.info['trailingPE']
        pb_ratio = stock.info['priceToBook']
        roe = stock.info['returnOnEquity']
        debt_to_equity = stock.info['debtToEquity']
        current_ratio = stock.info['currentRatio']
        
        # Calculate a weighted average score of financial ratios
        score = (pe_ratio + pb_ratio + roe + current_ratio) - debt_to_equity
        return score
    except Exception as e:
        st.warning("Could not fetch financial ratios. Using default score.")
        return 50  # Fallback score if ratios are unavailable

# Function to perform candlestick analysis using TA-Lib
def perform_candlestick_analysis(stock_data):
    # TA-Lib functions for common patterns
    doji = ta.CDLDOJI(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    hammer = ta.CDLHAMMER(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    engulfing = ta.CDLENGULFING(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    
    # Count the occurrence of patterns in the last year
    pattern_score = np.sum(doji) + np.sum(hammer) + np.sum(engulfing)
    
    # Normalize the score (based on the occurrence of patterns)
    normalized_score = pattern_score / len(stock_data)
    
    return normalized_score * 100  # Return score on a scale of 100

# Streamlit interface
def main():
    st.title("Stock Scoring Application")

    # Ticker input
    ticker = st.text_input("Enter Stock Ticker", value='AAPL')
    
    if st.button("Get Stock Score"):
        if ticker:
            score = get_stock_score(ticker)
            if score:
                st.success(f"Overall score for {ticker}: {score}")
        else:
            st.error("Please enter a valid ticker symbol.")

if __name__ == "__main__":
    main()
