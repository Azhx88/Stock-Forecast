# Stock Forecasting


A Streamlit-based **Stock Prediction & Investment Analysis** app.

## Features
- Fetch historical stock data (via `yfinance`)
- Forecast future prices using Prophet
- Visualize actual vs predicted prices (Plotly)
- Simple investment suggestion based on predicted growth
- Ready to extend with risk metrics (Sharpe, volatility) & portfolio optimization

## Quickstart

```bash
# 1) Create & activate venv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

## Notes
- Symbols: Use Yahoo tickers (e.g., `AAPL` for Apple, `TCS.NS` for TCS on NSE).
- Prophet may download a model the first time; allow a moment on first run.
- This app predicts based on historical prices only; **not financial advice**.

## Extend
- Add risk metrics & Sharpe ratio via `pypfopt`.
- Add multi-stock portfolio suggestion.
- Add news sentiment with `transformers` for better signals.
