import numpy as np
import pandas as pd
import pytest

from analysis.premium_model import calibrate, fit, predict, samples, walk_forward
from analysis.valuation_model import model_params, quarterly_inputs, run_model


def cyclical_stock(sensitivity=0.03, quarters=40):
    """Ten years of results with revenue growth cycling 2%..22% a year, and a market that pays a P/E rising with
    the latest year's growth: P/E = 15 x exp(sensitivity x growth %)."""

    dates = pd.date_range(end="2026-10-28", periods=quarters, freq="91D")
    growth = 0.12 + 0.10 * np.sin(np.arange(quarters) / 4)
    revenue = 1000 * np.cumprod((1 + growth) ** 0.25)
    results = pd.DataFrame({"Revenue": revenue, "Net_Income": revenue * 0.2, "EPS": revenue * 0.2 / 100,
                            "Gross_Profit": revenue * 0.5}, index=dates)
    inputs = quarterly_inputs(results, 3)
    days = pd.bdate_range(dates[0], "2026-11-10")
    ttm = inputs["ttm_eps"].reindex(inputs.index.union(days)).ffill().reindex(days)
    yoy = inputs["revenue_yoy"].reindex(inputs.index.union(days)).ffill().reindex(days).fillna(12)
    close = (ttm * 15 * np.exp(sensitivity * yoy)).bfill()
    return results, close


def run(results, close):
    return run_model(results, close, None, model_params({"anchor_source": "Fundamentals", "peer_weight": 1.0}))


def test_samples_use_the_trailing_p_e_the_market_paid():
    results, close = cyclical_stock()
    frame = samples(run(results, close)).dropna(subset=["premium"])
    assert len(frame) > 80
    row = frame.iloc[-1]
    assert row["actual_pe"] == pytest.approx(row["close"] / row["ttm_eps"])
    assert row["premium"] == pytest.approx(np.log(row["actual_pe"] / row["justified_pe"]))
    # Month-ends only, one per month.
    assert frame.index.to_period("M").is_unique


def test_walk_forward_explains_the_premium_and_forecasts_the_next_quarter():
    results, close = cyclical_stock()
    calibration = calibrate(run(results, close))
    quality, today, history = calibration["quality"], calibration["today"], calibration["history"]
    assert quality["months"] > 40
    # The market's P/E follows growth here, so the explained premium tracks the actual one closely.
    assert quality["premium_error_pct"] < 15
    assert 0 <= quality["direction_hit"] <= 1 and 0 <= quality["band_coverage"] <= 1
    # Next quarter: the close grown by a quarter of the yearly growth; no pull of the premium.
    assert today["base"] == pytest.approx(today["close"] * (1 + today["growth"] / 100) ** 0.25)
    assert today["bear"] <= today["base"] <= today["bull"]
    assert today["side"] in ("rich", "cheap")
    assert today["fair_price"] == pytest.approx(today["close"] / (1 + today["gap_pct"] / 100))
    assert "Revenue growth, latest year %" in calibration["coefficients"]["Feature"].tolist()
    # Walk-forward: a month's explained premium comes from a fit on earlier months only.
    day = history.dropna(subset=["explained_premium"]).index[5]
    before = history[history.index < day].dropna(subset=["premium"])
    assert predict(fit(before), history.loc[[day]]).iloc[0] == pytest.approx(history.at[day, "explained_premium"])


def test_the_band_uses_only_errors_known_at_the_time():
    results, close = cyclical_stock()
    history = calibrate(run(results, close))["history"]
    banded = history.dropna(subset=["bear"])
    day = banded.index[3]
    settled = history["error"][(history.index + pd.DateOffset(months=3)) <= day].dropna()
    assert banded.at[day, "bear"] == pytest.approx(history.at[day, "forecast"] * np.exp(settled.quantile(0.2)))
    # The first band needs a year of settled errors, so it starts well after the first forecast.
    assert banded.index[0] > history.dropna(subset=["forecast"]).index[0] + pd.DateOffset(months=14)


def test_the_valuation_record_compares_cheap_and_rich_months():
    results, close = cyclical_stock()
    record = calibrate(run(results, close))["record"].set_index("Horizon")
    assert "3 months" in record.index and "12 months" in record.index
    assert -1 <= record.loc["3 months", "Cheapness vs return (corr)"] <= 1
    assert record.loc["3 months", "Months"] >= 25


def test_too_little_history_gives_nothing():
    results, close = cyclical_stock(quarters=10)
    assert calibrate(run(results, close)) == {}
