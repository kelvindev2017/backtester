import numpy as np
import pandas as pd

from analysis.valuation_chart import pe_summary, pe_summary_text, revenue_estimate_quarters, revenue_pe_figure


def test_estimates_are_placed_after_the_last_reported_quarter():
    estimates = pd.DataFrame({"avg": [110e9, 125e9, 400e9]}, index=pd.Index(["0q", "+1q", "0y"], name="period"))
    quarters = revenue_estimate_quarters(estimates, pd.Timestamp("2026-07-31"))
    assert list(quarters.index) == [pd.Timestamp("2026-10-31"), pd.Timestamp("2027-01-31")]
    assert list(quarters) == [110e9, 125e9]


def test_estimates_handle_missing_data():
    assert revenue_estimate_quarters(pd.DataFrame(), pd.Timestamp("2026-07-31")).empty
    assert revenue_estimate_quarters(None, pd.Timestamp("2026-07-31")).empty
    bad = pd.DataFrame({"avg": [np.nan]}, index=["0q"])
    assert revenue_estimate_quarters(bad, pd.Timestamp("2026-07-31")).empty


def test_pe_summary_ranks_the_current_value_against_history():
    pe = pd.Series([40.0, 20.0, 60.0, 30.0, np.nan], index=pd.date_range("2020-01-01", periods=5, freq="YS"))
    summary = pe_summary(pe)
    assert summary["current"] == 30.0
    assert summary["percentile"] == 0.25
    assert summary["low"] == 20.0 and summary["low_date"] == pd.Timestamp("2021-01-01")
    text = pe_summary_text(summary, forward_pe=17.0)
    assert "25th percentile" in text and "Forward P/E today is 17.0x" in text
    assert "No trailing P/E" in pe_summary_text(pe_summary(pd.Series(dtype=float)))


def test_figure_stacks_pe_above_revenue_and_marks_estimates():
    quarters = pd.date_range("2024-01-31", periods=8, freq="QE-JAN")
    revenue = pd.Series(np.linspace(20e9, 60e9, 8), index=quarters)
    pe = pd.Series(np.linspace(60, 30, 400), index=pd.bdate_range("2024-01-01", periods=400))
    estimates = pd.Series([70e9], index=[quarters[-1] + pd.DateOffset(months=3)])
    fig = revenue_pe_figure(revenue, pe, estimates)
    names = [trace.name for trace in fig.data]
    assert names == ["Trailing P/E", "Revenue (reported)", "Revenue (analyst estimate)"]
    # Two separate y-axes in stacked rows, never an overlaid secondary axis.
    assert fig.data[0].yaxis == "y" and fig.data[1].yaxis == "y2"
    assert fig.layout.yaxis2.overlaying is None
    assert fig.data[1].y[-1] == 60.0  # billions


def test_figure_without_pe_or_estimates():
    revenue = pd.Series([1e9, 2e9], index=pd.to_datetime(["2025-03-31", "2025-06-30"]))
    fig = revenue_pe_figure(revenue, pd.Series(dtype=float))
    assert [trace.name for trace in fig.data] == ["Revenue (reported)"]
