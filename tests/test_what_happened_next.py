import numpy as np
import pandas as pd

from analysis.what_happened_next import (
    check_dates,
    non_overlapping_count,
    observations,
    scatter_figure,
    summary_table,
    walkthrough_figure,
)


def make_inputs(days=2600):
    index = pd.bdate_range("2015-01-01", periods=days)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, days))), index=index)
    eps = pd.Series(np.repeat(np.linspace(3, 6, days // 63 + 1), 63)[:days], index=index)
    valuation = pd.DataFrame({"Close": close, "TTM_Reported_EPS": eps, "Trailing_PE": close / eps})
    benchmark = pd.Series(100 * np.exp(np.arange(days) * 0.0003), index=index)
    return valuation, close, benchmark


def test_check_dates_take_the_last_trading_day_of_each_period():
    index = pd.bdate_range("2024-01-01", "2024-03-15")
    months = check_dates(index, "Month")
    assert list(months) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-15")]
    weeks = check_dates(index, "Week")
    assert all(day.dayofweek == 4 for day in weeks[:-1]) and weeks[-1] == index[-1]
    assert len(check_dates(index, "Day")) == len(index)


def test_observations_use_only_the_past_for_the_percentile_and_the_future_for_the_outcome():
    valuation, close, benchmark = make_inputs()
    obs = observations(valuation, close, benchmark, "Month", "3 months", lookback_years=5)
    row = obs.dropna(subset=["percentile", "return"]).iloc[0]
    when = obs.dropna(subset=["percentile", "return"]).index[0]
    window = valuation["Trailing_PE"].loc[when - pd.Timedelta(days=round(365.25 * 5)):when]
    assert row["percentile"] == (window < valuation["Trailing_PE"].loc[when]).mean() * 100
    end = close.loc[:when + pd.DateOffset(months=3)].iloc[-1]
    assert np.isclose(row["return"], (end / close.loc[when] - 1) * 100)
    assert np.isclose(row["excess"], row["return"] - row["benchmark_return"])
    # The look-ahead of the last rows has not finished: no outcome yet, but today still has a percentile.
    assert np.isnan(obs["return"].iloc[-1]) and np.isfinite(obs["percentile"].iloc[-1])
    # Too little history for a 5-year percentile at the start.
    assert np.isnan(obs["percentile"].iloc[0])


def test_daily_checks_count_far_fewer_non_overlapping_periods():
    valuation, close, benchmark = make_inputs()
    obs = observations(valuation, close, benchmark, "Day", "12 months", lookback_years=3)
    table = summary_table(obs, "12 months")
    first = table.iloc[0]
    assert first["Checks"] > 500 and first["Non-overlapping"] <= first["Checks"] / 200
    dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2021-02-01"])
    assert non_overlapping_count(dates, "12 months") == 2
    assert non_overlapping_count(dates, "1 week") == 3


def test_figures_build_for_a_past_check():
    valuation, close, benchmark = make_inputs()
    obs = observations(valuation, close, benchmark, "Week", "1 month", lookback_years=3)
    example = obs.dropna(subset=["excess", "percentile"]).index[-10]
    scatter = scatter_figure(obs, "TEST", "1 month", "SPY", example=example)
    assert "explained below" in str(scatter.layout.annotations)
    walk = walkthrough_figure(valuation, close, benchmark, obs, example, "TEST", "1 month", "SPY", lookback_years=3)
    assert "the dot's x" in str(walk.layout.annotations) and "the dot's y" in str(walk.layout.annotations)
