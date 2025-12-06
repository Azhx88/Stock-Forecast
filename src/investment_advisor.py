import pandas as pd
from dateutil.relativedelta import relativedelta
from src.predictor import forecast_future
import numpy as np

def _months_to_days(m):  # ~30.44 days / month
    return int(round(m * 30.44))

def _nearest_row(forecast: pd.DataFrame, target_dt: pd.Timestamp):
    if forecast is None or forecast.empty:
        return None
    fc = forecast.sort_values('ds')
    target_dt = pd.to_datetime(target_dt)
    if target_dt <= fc['ds'].iloc[0]:
        return fc.iloc[[0]]
    if target_dt >= fc['ds'].iloc[-1]:
        return fc.iloc[[-1]]
    idx = (fc['ds'] - target_dt).abs().idxmin()
    return fc.loc[[idx]]

def _band_width_ratio(row):
    """Relative uncertainty: (upper-lower)/|yhat|. Lower is better."""
    if row is None or row.empty:
        return np.nan
    r = row.iloc[0]
    yhat = float(r.get("yhat", np.nan))
    yhL = float(r.get("yhat_lower", np.nan))
    yhU = float(r.get("yhat_upper", np.nan))
    if not np.isfinite(yhat) or not np.isfinite(yhL) or not np.isfinite(yhU) or yhat == 0:
        return np.nan
    return abs(yhU - yhL) / abs(yhat)

def evaluate_horizons_for_stock(
    df: pd.DataFrame,
    date_col: str,
    close_col: str,
    model,
    horizons_months: list[int],
    principal: float,
    rank_mode: str = "Total Return",   # "Total Return" | "CAGR" | "Risk-Adjusted"
    risk_penalty: float = 0.5          # used only for Risk-Adjusted
) -> dict:
    if df is None or df.empty:
        return {"best_months": None, "expected_value": np.nan, "return_pct": np.nan,
                "principal": principal, "breakdown": []}

    df = df.dropna(subset=[date_col, close_col]).copy().sort_values(date_col)
    current_price = float(df[close_col].iloc[-1])
    now_dt = pd.to_datetime(df[date_col].iloc[-1])

    breakdown = []
    for mo in horizons_months:
        days = _months_to_days(mo)
        fc = forecast_future(model, days)
        target_dt = now_dt + relativedelta(months=mo)
        row = _nearest_row(fc, target_dt)

        pred_price = float(row['yhat'].iloc[0]) if row is not None else np.nan
        if not np.isfinite(pred_price) or current_price <= 0:
            growth = np.nan; growth_pct = np.nan; cagr_pct = np.nan
            exp_val = np.nan; risk = np.nan; score = np.nan
        else:
            growth = (pred_price / current_price) - 1.0
            growth_pct = growth * 100.0
            years = mo / 12.0
            cagr = (pred_price / current_price) ** (1.0 / years) - 1.0
            cagr_pct = cagr * 100.0
            exp_val = principal * (1.0 + growth)
            # risk proxy = relative band width; fall back to NaN (ignored) if missing
            risk = _band_width_ratio(row)

            # --- scoring ---
            if rank_mode == "Total Return":
                score = exp_val
            elif rank_mode == "CAGR":
                score = cagr_pct
            else:  # Risk-Adjusted
                # higher return, lower risk; if risk missing, treat as 0 penalty
                penalty = (risk_penalty * risk) if np.isfinite(risk) else 0.0
                score = (cagr_pct if np.isfinite(cagr_pct) else -1e9) - 100.0 * penalty

        breakdown.append({
            "months": mo,
            "predicted_price": round(pred_price, 4) if np.isfinite(pred_price) else np.nan,
            "growth_pct": round(growth_pct, 4) if np.isfinite(growth_pct) else np.nan,
            "cagr_pct": round(cagr_pct, 4) if np.isfinite(cagr_pct) else np.nan,
            "expected_value": round(exp_val, 2) if np.isfinite(exp_val) else np.nan,
            "risk_bw_ratio": round(risk, 5) if np.isfinite(risk) else np.nan,
            "score": score if np.isfinite(score) else -1e12
        })

    valid = [b for b in breakdown if np.isfinite(b.get("score", np.nan))]
    if not valid:
        return {"best_months": None, "expected_value": np.nan, "return_pct": np.nan,
                "principal": principal, "breakdown": breakdown}

    best = max(valid, key=lambda x: x["score"])
    return {
        "best_months": int(best["months"]),
        "expected_value": float(best["expected_value"]) if np.isfinite(best["expected_value"]) else np.nan,
        "return_pct": float(best["growth_pct"]) if np.isfinite(best["growth_pct"]) else np.nan,
        "principal": float(principal),
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
    return rows
