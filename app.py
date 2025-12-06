import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from dateutil.relativedelta import relativedelta

from src.data_loader import fetch_stock_data
from src.predictor import train_prophet_model, forecast_future

# ---------------- Page setup ----------------
st.set_page_config(page_title="StockVisionAI", layout="wide")
st.title("📈 StockVisionAI — Predict • Analyze • Invest")

# ---------------- Helpers ----------------
def months_to_days(months: int) -> int:
    return int(round(months * 30.44))  # avg month length

def years_to_days(years: int) -> int:
    return int(round(years * 365.25))

def detect_cols(df: pd.DataFrame):
    """Detect date and close columns robustly for yfinance output variants."""
    if df is None or df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([c for c in col if c]).strip() for col in df.columns.values]
    cols = [str(c) for c in df.columns]
    date_col  = next((c for c in cols if 'date'  in c.lower()), None)
    close_col = next((c for c in cols if 'close' in c.lower()), None)
    return date_col, close_col

def nearest_row(forecast: pd.DataFrame, target_dt: pd.Timestamp):
    if forecast is None or forecast.empty:
        return None
    fc = forecast.sort_values('ds')
    target_dt = pd.to_datetime(target_dt)
    if target_dt <= fc['ds'].iloc[0]:  return fc.iloc[[0]]
    if target_dt >= fc['ds'].iloc[-1]: return fc.iloc[[-1]]
    idx = (fc['ds'] - target_dt).abs().idxmin()
    return fc.loc[[idx]]

def band_width_ratio(row):
    if row is None or row.empty:
        return np.nan
    r = row.iloc[0]
    yhat = float(r.get("yhat", np.nan))
    yhL  = float(r.get("yhat_lower", np.nan))
    yhU  = float(r.get("yhat_upper", np.nan))
    if not np.isfinite(yhat) or not np.isfinite(yhL) or not np.isfinite(yhU) or yhat == 0:
        return np.nan
    return abs(yhU - yhL) / abs(yhat)

def evaluate_horizons_for_stock(
    df: pd.DataFrame,
    date_col: str,
    close_col: str,
    model,
    max_months: int,
    principal: float,
    rank_mode: str = "Total Return"
) -> dict:
    if df is None or df.empty:
        return {"best_months": None, "expected_value": np.nan, "return_pct": np.nan,
                "principal": principal, "breakdown": []}

    df = df.dropna(subset=[date_col, close_col]).copy().sort_values(date_col)
    current_price = float(df[close_col].iloc[-1])
    now_dt = pd.to_datetime(df[date_col].iloc[-1])

    # Generate a dynamic set of horizons to evaluate
    if max_months <= 12:
        horizons_months = list(range(1, max_months + 1))
    else:
        horizons_months = sorted(list(set(
            list(range(1, 13)) + [m for m in range(18, max_months + 1, 6)]
        )))
    horizons_months = [m for m in horizons_months if m <= max_months]

    breakdown = []
    max_days = months_to_days(max(horizons_months))
    full_forecast = forecast_future(model, max_days)

    for mo in horizons_months:
        target_dt = now_dt + relativedelta(months=mo)
        row = nearest_row(full_forecast, target_dt)

        pred_price = float(row['yhat'].iloc[0]) if row is not None else np.nan
        if not np.isfinite(pred_price) or current_price <= 0:
            growth_pct = np.nan
            exp_val = np.nan
            score = -np.inf
        else:
            growth = (pred_price / current_price) - 1.0
            growth_pct = growth * 100.0
            exp_val = principal * (1.0 + growth)
            score = growth_pct

        breakdown.append({
            "months": mo,
            "predicted_price": pred_price,
            "growth_pct": growth_pct,
            "expected_value": exp_val,
            "score": score
        })

    if not breakdown:
        return {"best_months": None, "expected_value": np.nan, "return_pct": np.nan,
                "principal": principal, "breakdown": []}

    best_horizon = max(breakdown, key=lambda x: x["score"])
    return {
        "best_months": best_horizon["months"],
        "expected_value": best_horizon["expected_value"],
        "return_pct": best_horizon["growth_pct"],
        "principal": principal,
        "breakdown": breakdown
    }

def summarize_best_choices(evals: list[dict]) -> list[dict]:
    rows = []
    for e in evals:
        if "symbol" not in e:
            continue
        rows.append({
            "symbol": e["symbol"],
            "best_months": e.get("best_months"),
            "expected_value": e.get("expected_value"),
            "return_pct": e.get("return_pct"),
            "principal": e.get("principal"),
        })
    
    # sort descending by expected value (highest return first)
    df = pd.DataFrame(rows)
    if not df.empty and "expected_value" in df.columns:
        df = df.sort_values("expected_value", ascending=False)
    
    return df.to_dict(orient="records")

# ---------- Pretty UI helpers ----------
def apply_theme(theme: str):
    """Inject CSS for light/dark and return plotly template name."""
    if theme == "Dark":
        template = "plotly_dark"
        css = """
        <style>
        :root { --card-bg:#111418; --card-text:#e6e6e6; --accent:#22c55e; --muted:#94a3b8; }
        .sv-card{background:var(--card-bg); color:var(--card-text); border:1px solid #23262b;
                 padding:16px; border-radius:16px; box-shadow: 0 1px 12px rgba(0,0,0,.35);}
        .sv-kv{display:flex; gap:8px; color:var(--muted); font-size:12px;}
        .sv-big{font-size:28px; font-weight:700; margin:6px 0;}
        .sv-delta{color:#16a34a; font-weight:700;}
        </style>
        """
    else:
        template = "plotly_white"
        css = """
        <style>
        :root { --card-bg:#ffffff; --card-text:#0f172a; --accent:#16a34a; --muted:#64748b; }
        .sv-card{background:var(--card-bg); color:var(--card-text); border:1px solid #e2e8f0;
                 padding:16px; border-radius:16px; box-shadow: 0 1px 10px rgba(2,6,23,.06);}
        .sv-kv{display:flex; gap:8px; color:var(--muted); font-size:12px;}
        .sv-big{font-size:28px; font-weight:700; margin:6px 0;}
        .sv-delta{color:#16a34a; font-weight:700;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)
    return template

def render_card(symbol, best_months, expected_value, return_pct, principal):
    exp_txt = "—" if not np.isfinite(expected_value) else f"₹{expected_value:,.0f}"
    delta_txt = "—" if not np.isfinite(return_pct) else f"{return_pct:.2f}%"
    principal_txt = f"₹{principal:,.0f}"

    if np.isfinite(return_pct) and return_pct < 0:
        delta_html = f'<div class="sv-delta" style="color: #ef4444;">↓ {delta_txt}</div>'
    else:
        delta_html = f'<div class="sv-delta">↑ {delta_txt}</div>'

    html = f"""
    <div class="sv-card">
      <div class="sv-kv">{symbol} &nbsp;—&nbsp; Best: {best_months} mo</div>
      <div class="sv-big">Expected {exp_txt}</div>
      {delta_html}
      <div class="sv-kv">Principal {principal_txt}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Parameters")
    theme = st.radio("Theme", ["Dark", "Light"], horizontal=True, index=0)
    plotly_template = apply_theme(theme)

    tickers = st.text_area(
        "Stock Symbols (comma-separated, Yahoo Finance format)",
        value="RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, SBIN.NS, BAJFINANCE.NS, BHARTIARTL.NS",
        help="Examples: RELIANCE.NS, TCS.NS (NSE); AAPL, MSFT (US)"
    )
    symbols = [s.strip() for s in tickers.split(",") if s.strip()]

    start_date = st.date_input("Start Date", date(2020, 1, 1))
    end_date   = st.date_input("End Date", date.today())

    unit = st.radio("Forecast horizon unit", ["Months", "Years"], horizontal=True, index=0)
    horizon_value = st.number_input(
        f"Forecast Horizon ({unit})",
        min_value=1, max_value=60 if unit == "Months" else 10,
        value=12 if unit == "Months" else 1
    )

    rank_mode = "Total Return"

    st.caption("The app evaluates horizons up to your selected maximum and chooses the best per stock.")
    amount = st.number_input("Investment amount (principal)", min_value=1000, value=100000, step=1000)

    run = st.button("Analyze & Predict")

# ---------------- Main ----------------
if run:
    if not symbols:
        st.error("Please enter at least one stock symbol.")
        st.stop()

    if unit == "Months":
        forecast_days = months_to_days(horizon_value)
        max_months = horizon_value
    else:
        forecast_days = years_to_days(horizon_value)
        max_months = horizon_value * 12

    results = []
    per_stock_figs = []

    for sym in symbols:
        with st.spinner(f"Fetching & modeling: {sym}"):
            df = fetch_stock_data(sym, start_date, end_date)
            if df is None or df.empty:
                st.warning(f"{sym}: No data for the selected range.")
                continue

            date_col, close_col = detect_cols(df)
            if not date_col or not close_col:
                st.warning(f"{sym}: Could not detect Date/Close columns. Columns={df.columns.tolist()}")
                continue

            model = train_prophet_model(df)
            forecast = forecast_future(model, forecast_days)

            # --------- Build chart: candlestick + attached forecast line ---------
            fig = go.Figure()

            x_actual = pd.to_datetime(df[date_col])

            has_ohlc = (
                any('open'  in c.lower() for c in df.columns) and
                any('high'  in c.lower() for c in df.columns) and
                any('low'   in c.lower() for c in df.columns) and
                any('close' in c.lower() for c in df.columns)
            )
            if has_ohlc:
                def find_col(name):
                    return next((c for c in df.columns if name in c.lower()), None)
                o = find_col('open'); h = find_col('high'); l = find_col('low'); c_ = find_col('close')
                fig.add_trace(go.Candlestick(
                    x=x_actual, open=df[o], high=df[h], low=df[l], close=df[c_],
                    name=f"{sym} (Actual)"
                ))
                last_y = float(df[c_].iloc[-1])
            else:
                fig.add_trace(go.Scatter(
                    x=x_actual, y=df[close_col], name=f"{sym} (Actual Close)"
                ))
                last_y = float(df[close_col].iloc[-1])

            last_x = x_actual.iloc[-1]

            # Confidence band (if available)
            if set(['yhat_lower','yhat_upper']).issubset(forecast.columns):
                fig.add_trace(go.Scatter(
                    x=pd.concat([pd.Series([last_x]), forecast['ds']]),
                    y=pd.concat([pd.Series([last_y]), forecast['yhat_upper']]),
                    mode="lines", name=f"{sym} Pred +CI", line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=pd.concat([pd.Series([last_x]), forecast['ds']]),
                    y=pd.concat([pd.Series([last_y]), forecast['yhat_lower']]),
                    mode="lines", name=f"{sym} Pred -CI", fill="tonexty",
                    line=dict(width=0), showlegend=False
                ))

            # Prediction line continuing from last real point
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series([last_x]), forecast['ds']]),
                y=pd.concat([pd.Series([last_y]), forecast['yhat']]),
                name=f"{sym} (Predicted)", mode="lines"
            ))

            fig.update_layout(
                template=plotly_template,
                title=f"{sym} — Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            per_stock_figs.append((sym, fig))

            # --------- Evaluate horizons & choose best for tiles ----------
            stock_eval = evaluate_horizons_for_stock(
                df=df,
                date_col=date_col,
                close_col=close_col,
                model=model,
                max_months=max_months,
                principal=amount,
                rank_mode=rank_mode
            )
            stock_eval["symbol"] = sym
            results.append(stock_eval)

    # ----- Tiles: best choice per stock -----
    if results:
        st.subheader("🎯 Best Holding Period per Stock (for your amount)")
        rows = summarize_best_choices(results)  # already sorted desc by expected_value

        # Card grid
        cols = st.columns(min(4, max(1, len(rows))))
        for i, row in enumerate(rows):
            with cols[i % len(cols)]:
                render_card(
                    symbol=row['symbol'],
                    best_months=row['best_months'],
                    expected_value=row['expected_value'],
                    return_pct=row['return_pct'],
                    principal=row['principal']
                )

        # Sorted table for quick scan
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ----- Charts -----
    if per_stock_figs:
        st.subheader("📊 Charts")
        for sym, fig in per_stock_figs:
            st.plotly_chart(fig, use_container_width=True)

    if not results and not per_stock_figs:
        st.info("No results to show. Try a different symbol or date range.")
