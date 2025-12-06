from prophet import Prophet
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

def _detect_cols(df: pd.DataFrame):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([c for c in col if c]).strip() for col in df.columns.values]
    cols = [str(c) for c in df.columns]
    date_col = next((c for c in cols if 'date' in c.lower()), None)
    close_col = next((c for c in cols if 'close' in c.lower()), None)
    if date_col is None or close_col is None:
        raise KeyError(f"Expected a 'Date' and 'Close' column, got {df.columns.tolist()}")
    return date_col, close_col

def train_prophet_model(df):
    try:
        date_col, close_col = _detect_cols(df)
        data = df[[date_col, close_col]].copy()
        data.rename(columns={date_col:'ds', close_col:'y'}, inplace=True)
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
        data['y']  = pd.to_numeric(data['y'], errors='coerce')
        data = data.dropna(subset=['ds','y']).sort_values('ds')
        if data.empty:
            raise ValueError("No valid data after cleaning.")

        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(data)
        m.model_type = "prophet"
        return m
    except Exception as e:
        print(f"[Prophet failed -> Holt-Winters] {e}")
        return train_holt_winters_model(df)

def train_holt_winters_model(df):
    # Smooth, non-linear fallback
    date_col, close_col = _detect_cols(df)
    s = df[[date_col, close_col]].dropna().sort_values(date_col)
    s['ds'] = pd.to_datetime(s[date_col])
    s.set_index('ds', inplace=True)
    y = pd.to_numeric(s[close_col], errors='coerce').dropna()
    # Daily trend, no seasonality to avoid overfit; raise if too short
    if len(y) < 20:
        # still create a trivial model that echoes last value
        model = {"type":"naive", "last": float(y.iloc[-1]), "last_index": y.index[-1]}
        return model
    hw = ExponentialSmoothing(y, trend='add', seasonal=None).fit(optimized=True)
    return {"type":"holtwinters", "fit": hw, "last_index": y.index[-1]}

def forecast_future(model, periods):
    # Prophet path
    if hasattr(model, 'model_type') and model.model_type == "prophet":
        future = model.make_future_dataframe(periods=periods, freq='D')
        fc = model.predict(future)[['ds','yhat','yhat_lower','yhat_upper']]
        return fc

    # Holt-Winters path
    if isinstance(model, dict) and model.get("type") == "holtwinters":
        hw = model["fit"]
        last = model["last_index"]
        future_idx = pd.date_range(last, periods=periods+1, freq='D')[1:]  # strictly future
        yhat = hw.forecast(periods)
        # build band from residual std
        resid = hw.resid
        sigma = float(np.nanstd(resid)) if resid is not None else 0.0
        yhat_lower = yhat - 1.96 * sigma
        yhat_upper = yhat + 1.96 * sigma
        return pd.DataFrame({"ds": future_idx, "yhat": yhat.values,
                             "yhat_lower": yhat_lower.values, "yhat_upper": yhat_upper.values})

    # Naive fallback
    if isinstance(model, dict) and model.get("type") == "naive":
        last = model["last_index"]
        future_idx = pd.date_range(last, periods=periods+1, freq='D')[1:]
        yhat = pd.Series([model["last"]] * periods, index=future_idx)
        return pd.DataFrame({"ds": yhat.index, "yhat": yhat.values,
                             "yhat_lower": yhat.values, "yhat_upper": yhat.values})

    raise ValueError("Unknown model type for forecasting.")
