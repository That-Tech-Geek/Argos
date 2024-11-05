# Argos: Stock Scoring Application

This is a **Stock Scoring Application** built using **Streamlit**. The app fetches stock market data and evaluates a stock ticker based on various metrics like rolling means, VIX volatility data, RSI (Relative Strength Index), financial ratios, and candlestick pattern analysis to provide an overall score for the stock.

## Features

- **Rolling Means**: Calculates 7-day, 50-day, and 200-day rolling averages for both the opening and closing prices of the stock.
- **VIX Data**: Fetches the VIX (Volatility Index) for multiple timeframes including 1 year, 1 month, 1 day, 1 hour, and 1 minute.
- **RSI (Relative Strength Index)**: Determines the percentage of time the stock was overbought or oversold using a custom RSI calculation (without TA-Lib).
- **Financial Ratios**: Fetches financial ratios such as Price-to-Earnings (P/E), Price-to-Book (P/B), Return on Equity (ROE), Debt-to-Equity, and Current Ratio to evaluate the stock's financial health.
- **Candlestick Pattern Analysis**: Performs quantitative analysis of candlestick patterns to assess trends and patterns in stock prices.

## Installation

1. Clone the repository or download the source code.

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Required libraries:
    - [Streamlit](https://streamlit.io/)
    - [yfinance](https://pypi.org/project/yfinance/)
    - [Pandas](https://pandas.pydata.org/)
    - [NumPy](https://numpy.org/)

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run stock_scoring_app.py
    ```

2. Once the app starts, enter the stock ticker in the input box (e.g., `AAPL` for Apple) and click the **Get Stock Score** button.

3. The app will fetch stock data, perform calculations, and display the overall score for the selected ticker.

## Application Logic

The stock score is calculated using the following components:

1. **Rolling Means**:
    - `7d`, `50d`, `200d` averages of both the opening and closing prices.
  
2. **VIX (Volatility Index)**:
    - VIX values are fetched for multiple intervals (1 year, 1 month, 1 day, 1 hour, 1 minute) and averaged.

3. **RSI (Relative Strength Index)**:
    - A custom function is used to calculate RSI from stock data, and the percentage of time the stock is in an overbought (RSI > 70) or oversold (RSI < 30) condition is considered.

4. **Financial Ratios**:
    - Key financial ratios like P/E, P/B, ROE, Debt-to-Equity, and Current Ratio are fetched from Yahoo Finance.

5. **Candlestick Patterns**:
    - Quantitative analysis of candlestick patterns is performed to provide insights into the stock’s price trends and potential reversals.

The scoring formula combines these metrics to produce an overall score. Each component is weighted to reflect its importance.

## Example

To get a stock score for **Apple (AAPL)**:
- Enter `AAPL` in the input box and click **Get Stock Score**.
- The app will display the overall score based on its market performance and financial metrics.

## Screenshots

![App Screenshot](assets/stock_scoring_screenshot.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Note

- This application provides a competitive score based on available data and technical analysis. It is recommended to conduct further research before making investment decisions.

---

## Contact

For any queries, feel free to contact the developer.
