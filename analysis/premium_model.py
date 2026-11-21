"""Market premium model: calibrate on history, then judge today's close and the next quarter.

Part 1, calibration. The ground truth is the trailing P/E the market actually paid: close / reported EPS of the last
four quarters. The valuation model's fundamental P/E says what growth, gross margin, the yield and risk justify (on
next year's earnings; times one year of growth on trailing ones). The premium is the gap between the two, in logs:

    premium = ln(actual trailing P/E / justified trailing P/E)

The premium is explained by fundamentals known on each day (revenue growth, its change of pace, the gross-margin
trend, the net margin, EPS growth) with a ridge regression, fitted walk-forward: on each month-end the model only
sees the months before it, so the history shows how it would have done in real time.

Part 2, judgement and forecast. Today's premium against the one the fundamentals explain says whether the close
sits on the rich (bearish) or cheap (bullish) side. The next quarter's price is the trailing EPS a quarter on, times
the justified P/E, times the premium partly drifting back to the explained one at the speed seen in the history.
The bull and bear prices are the 80th and 20th percentiles of the walk-forward errors of that same forecast.
"""

import numpy as np
import pandas as pd

FEATURES = {
    "growth": "Revenue growth, latest year %",
    "acceleration": "Growth change of pace (latest year - history) %",
    "gross_margin_change": "Gross margin change over a year, points",
    "net_margin": "Net margin %",
    "eps_growth": "EPS growth over a year % (within -100..200)",
}
MIN_MONTHS = 24          # months of history before the first walk-forward fit
HORIZON_MONTHS = 3       # one quarter ahead
RIDGE = 0.1              # shrinkage on the standardised features, per sample (keeps a short history from overfitting)


def _held(quarterly, column, days):
    series = quarterly[column].dropna() if column in quarterly else pd.Series(dtype=float)
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        return pd.Series(np.nan, index=days, dtype=float)
    return series.reindex(series.index.union(days)).ffill().reindex(days)


def samples(result):
    """Month-end samples of the actual and justified trailing P/E, the premium and its fundamental features."""

    daily, quarterly = result["daily"], result["quarterly"]
    if daily.empty or quarterly.empty or "fundamental_pe" not in daily:
        return pd.DataFrame()
    month_ends = daily["close"].dropna().groupby(daily["close"].dropna().index.to_period("M")).tail(1).index
    frame = pd.DataFrame(index=month_ends)
    frame["close"] = daily["close"].reindex(month_ends)
    ttm = _held(quarterly, "ttm_eps", month_ends)
    frame["ttm_eps"] = ttm
    releases = quarterly[["ttm_eps", "gross_margin"]].copy() if "gross_margin" in quarterly else quarterly[["ttm_eps"]].copy()
    releases = releases[~releases.index.duplicated(keep="last")]
    # A year back by release: four releases earlier.
    year_ago = releases.shift(4)
    eps_growth = ((releases["ttm_eps"] / year_ago["ttm_eps"].where(year_ago["ttm_eps"] > 0) - 1) * 100).clip(-100, 200)
    frame["growth"] = _held(quarterly, "revenue_yoy", month_ends)
    frame["acceleration"] = frame["growth"] - _held(quarterly, "revenue_cagr", month_ends)
    if "gross_margin" in releases:
        change = releases["gross_margin"] - year_ago["gross_margin"]
        frame["gross_margin_change"] = _held(pd.DataFrame({"x": change}), "x", month_ends)
    else:
        frame["gross_margin_change"] = np.nan
    frame["net_margin"] = _held(quarterly, "net_margin", month_ends)
    frame["eps_growth"] = _held(pd.DataFrame({"x": eps_growth}), "x", month_ends)
    growth = daily["rule_growth"].reindex(month_ends) if "rule_growth" in daily else frame["growth"]
    justified_forward = daily["fundamental_pe"].reindex(month_ends)
    frame["justified_pe"] = justified_forward * (1 + growth.fillna(0) / 100)
    frame["actual_pe"] = (frame["close"] / ttm.where(ttm > 0)).replace([np.inf, -np.inf], np.nan)
    frame["premium"] = np.log(frame["actual_pe"] / frame["justified_pe"])
    return frame


def _design(frame, columns):
    return frame[columns].astype(float)


def fit(frame):
    """Ridge fit of the premium on the features that have data. Returns a dict for ``predict`` (None if too few)."""

    usable = [column for column in FEATURES if frame[column].notna().mean() >= 0.8]
    data = frame.dropna(subset=usable + ["premium"])
    # A feature that never moves explains nothing (and its tiny rounding noise would get a huge weight).
    usable = [column for column in usable if data[column].std() > 1e-6 * max(1.0, data[column].abs().mean())]
    if len(data) < 12:
        return None
    x = _design(data, usable)
    mean, scale = x.mean(), x.std().replace(0, 1).fillna(1)
    z = ((x - mean) / scale).to_numpy()
    y = data["premium"].to_numpy()
    intercept = y.mean()
    beta = (np.linalg.solve(z.T @ z + RIDGE * len(z) * np.eye(z.shape[1]), z.T @ (y - intercept)) if usable
            else np.array([]))
    fitted = intercept + z @ beta if usable else np.full(len(y), intercept)
    residual = y - fitted
    return {"columns": usable, "mean": mean, "scale": scale, "intercept": intercept, "beta": beta,
            "residual_std": float(np.std(residual)), "n": len(data)}


def predict(model, frame):
    """Explained premium for each row of ``frame`` (NaN where a feature is missing)."""

    if model is None:
        return pd.Series(np.nan, index=frame.index)
    if not model["columns"]:
        return pd.Series(model["intercept"], index=frame.index)
    z = (_design(frame, model["columns"]) - model["mean"]) / model["scale"]
    return model["intercept"] + z @ pd.Series(model["beta"], index=model["columns"])


def _reversion(residuals, lag):
    """How much of a premium gap is left ``lag`` months on (0: gone, 1: stays), from the residuals' autocorrelation."""

    residuals = residuals.dropna()
    if len(residuals) <= lag + 6:
        return 0.5
    now, later = residuals.iloc[:-lag].to_numpy(), residuals.iloc[lag:].to_numpy()
    if np.std(now) == 0:
        return 0.5
    return float(np.clip(np.polyfit(now, later, 1)[0], 0, 1))


def forecast_price(row, explained, keep, growth):
    """Next-quarter price from a month-end row: EPS a quarter on x justified P/E x the premium drifting back."""

    eps = row["ttm_eps"] * (1 + growth / 100) ** (HORIZON_MONTHS / 12)
    premium = explained + keep * (row["premium"] - explained)
    return eps * row["justified_pe"] * np.exp(premium)


def walk_forward(frame):
    """Real-time history: on each month-end fit on the months before it, explain the premium and forecast the price
    a quarter on. Adds explained_premium, forecast (made then, for 3 months later) and the realised close."""

    out = frame.copy()
    out["explained_premium"], out["forecast"], out["keep"] = np.nan, np.nan, np.nan
    usable = out.dropna(subset=["premium"])
    for position, day in enumerate(out.index):
        before = usable[usable.index < day]
        if len(before) < MIN_MONTHS:
            continue
        model = fit(before)
        if model is None:
            continue
        row = out.loc[day]
        explained = predict(model, out.loc[[day]]).iloc[0]
        out.at[day, "explained_premium"] = explained
        residuals = before["premium"] - predict(model, before)
        keep = _reversion(residuals, HORIZON_MONTHS)
        out.at[day, "keep"] = keep
        growth = row["growth"] if np.isfinite(row["growth"]) else 0.0
        if np.isfinite(explained) and np.isfinite(row["premium"]) and row["ttm_eps"] > 0:
            out.at[day, "forecast"] = forecast_price(row, explained, keep, growth)
    out["realised"] = out["close"].shift(-HORIZON_MONTHS)
    out["error"] = np.log(out["realised"] / out["forecast"])
    return out


def calibrate(result):
    """Part 1 and Part 2 for a model run. Returns a dict (empty when the history is too short or has no EPS):

    history: month-end frame with the walk-forward columns; model: the fit on all history; quality: walk-forward
    error, direction hit rate against 'no change', band coverage; today: actual and explained premium, the gap, the
    side (rich / cheap) and next quarter's base, bull and bear prices; coefficients: each feature's effect.
    """

    frame = samples(result)
    if frame.empty or frame["premium"].notna().sum() < MIN_MONTHS + 6:
        return {}
    history = walk_forward(frame)
    tested = history.dropna(subset=["forecast", "realised"])
    errors = tested["error"]
    low, high = (errors.quantile(0.2), errors.quantile(0.8)) if len(errors) >= 8 else (np.nan, np.nan)
    quality = {
        "months": int(len(tested)),
        "premium_error_pct": float((np.exp((history["premium"] - history["explained_premium"]).abs().median()) - 1) * 100)
        if history["explained_premium"].notna().any() else np.nan,
        "price_error_pct": float((np.exp(errors.abs().median()) - 1) * 100) if len(errors) else np.nan,
        "naive_error_pct": float((np.exp(np.log(tested["realised"] / tested["close"]).abs().median()) - 1) * 100)
        if len(tested) else np.nan,
        "direction_hit": float(((tested["forecast"] > tested["close"]) == (tested["realised"] > tested["close"])).mean())
        if len(tested) else np.nan,
        "band_coverage": float(((tested["realised"] >= tested["forecast"] * np.exp(low))
                                & (tested["realised"] <= tested["forecast"] * np.exp(high))).mean())
        if len(tested) and np.isfinite(low) else np.nan,
    }
    model = fit(history.dropna(subset=["premium"]))
    last = history.dropna(subset=["premium"]).iloc[-1]
    explained = predict(model, history.loc[[last.name]]).iloc[0]
    residuals = history["premium"] - predict(model, history)
    keep = _reversion(residuals, HORIZON_MONTHS)
    growth = last["growth"] if np.isfinite(last["growth"]) else 0.0
    base = forecast_price(last, explained, keep, growth)
    gap = float(last["premium"] - explained)
    today = {
        "date": last.name, "close": float(last["close"]), "actual_pe": float(last["actual_pe"]),
        "justified_pe": float(last["justified_pe"]), "premium_pct": float((np.exp(last["premium"]) - 1) * 100),
        "explained_premium_pct": float((np.exp(explained) - 1) * 100),
        "gap_pct": float((np.exp(gap) - 1) * 100), "side": "rich (bearish side)" if gap > 0 else "cheap (bullish side)",
        "fair_price": float(last["close"] * np.exp(-gap)), "keep": keep,
        "base": float(base), "bull": float(base * np.exp(high)) if np.isfinite(high) else np.nan,
        "bear": float(base * np.exp(low)) if np.isfinite(low) else np.nan,
        "horizon": last.name + pd.DateOffset(months=HORIZON_MONTHS),
    }
    coefficients = pd.DataFrame(
        [{"Feature": FEATURES[column], "Today": float(last[column]),
          "Effect of +1 st. dev. on premium %": float((np.exp(beta) - 1) * 100)}
         for column, beta in zip(model["columns"], model["beta"])]) if model and model["columns"] else pd.DataFrame()
    return {"history": history, "model": model, "quality": quality, "today": today, "coefficients": coefficients}
