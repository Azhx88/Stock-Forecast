import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return pd.DataFrame()
        data.reset_index(inplace=True)
        # Ensure required columns exist
        if 'Close' not in data.columns:
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
