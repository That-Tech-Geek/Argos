import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize

# Function to fetch historical stock data
def fetch_data(tickers):
    data = yf.download(tickers, period="1y", interval="1d")
    
    # Check if columns are multi-level (this happens when multiple tickers are involved)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close']  # Extract only the adjusted close price for each ticker
    else:
        data = data[['Adj Close']]  # For a single ticker, extract the adjusted close price directly
        
    return data

# Function to calculate daily returns
def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

# Function to calculate the expected portfolio return and volatility
def portfolio_metrics(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Function to calculate the Sharpe Ratio
def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return, portfolio_volatility = portfolio_metrics(weights, mean_returns, cov_matrix)
    return - (portfolio_return - risk_free_rate) / portfolio_volatility  # Negative for minimization

# Function to get the optimal portfolio
def get_optimal_portfolio(tickers):
    # Fetch the data
    data = fetch_data(tickers)

    # Calculate returns
    returns = calculate_returns(data)

    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Number of assets
    num_assets = len(tickers)

    # Initial guess for the weights (equal distribution)
    initial_weights = np.ones(num_assets) / num_assets

    # Bounds for the weights (each weight between 0 and 1)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Constraints: sum of weights should be 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Minimize the negative Sharpe ratio
    result = minimize(sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    portfolio_return, portfolio_volatility = portfolio_metrics(optimal_weights, mean_returns, cov_matrix)

    return optimal_weights, portfolio_return, portfolio_volatility

# Streamlit interface
st.title("Stock Portfolio Optimization")

# Input for ticker symbols
tickers_input = st.text_input("Enter stock tickers (comma separated, e.g., AAPL, MSFT, TSLA):")
if tickers_input:
    tickers = [ticker.strip() for ticker in tickers_input.split(",")]

    if st.button("Optimize Portfolio"):
        optimal_weights, portfolio_return, portfolio_volatility = get_optimal_portfolio(tickers)

        st.write(f"Optimal Weights for Portfolio:")
        for ticker, weight in zip(tickers, optimal_weights):
            st.write(f"{ticker}: {weight * 100:.2f}%")

        st.write(f"Expected Portfolio Return: {portfolio_return * 100:.2f}%")
        st.write(f"Expected Portfolio Volatility (Risk): {portfolio_volatility * 100:.2f}%")
else:
    st.error("Please enter valid tickers.")
