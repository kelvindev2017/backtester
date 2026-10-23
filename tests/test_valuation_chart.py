import numpy as np
import pandas as pd

from analysis.valuation_chart import (
    peg_summary_text,
    pe_summary,
    pe_summary_text,
    revenue_estimate_quarters,
    revenue_pe_figure,
    trailing_peg,
)


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


def test_trailing_peg_divides_pe_by_eps_growth_and_skips_shrinking_earnings():
    index = pd.bdate_range("2023-01-02", "2025-12-31")
    eps = pd.Series(np.where(index < pd.Timestamp("2024-01-01"), 2.0, 3.0), index=index)
    eps[index >= pd.Timestamp("2025-01-01")] = 1.5  # earnings halve in 2025
    valuation = pd.DataFrame({"TTM_Reported_EPS": eps, "Trailing_PE": 30.0}, index=index)
    peg = trailing_peg(valuation)
    assert peg.loc["2023"].isna().all()  # no year-ago EPS yet
    assert peg.loc["2024-06-03"] == 30.0 / 50.0  # +50% EPS growth
    assert peg.loc["2025-06-02":].isna().all()  # shrinking EPS: PEG not meaningful


def test_peg_caption_and_panel():
    peg = pd.Series([0.5, 1.5, 1.0], index=pd.to_datetime(["2022-01-03", "2023-01-02", "2024-01-01"]))
    text = peg_summary_text(peg, forward_pe=20.0, forward_growth=0.4)
    assert "Trailing PEG 1.00" in text and "Forward PEG 0.50" in text
    assert "No trailing PEG" in peg_summary_text(pd.Series(dtype=float))
    revenue = pd.Series([1e9, 2e9], index=pd.to_datetime(["2023-12-31", "2024-03-31"]))
    fig = revenue_pe_figure(revenue, pd.Series(dtype=float), peg=peg)
    assert [trace.name for trace in fig.data] == ["Trailing PEG", "Revenue (reported)"]
    assert fig.data[0].yaxis == "y2" and fig.data[1].yaxis == "y3"


def test_peg_axis_is_capped_when_growth_restarts_from_near_zero():
    peg = pd.Series([0.5, 12.0, 0.8], index=pd.to_datetime(["2022-01-03", "2023-06-01", "2024-01-01"]))
    revenue = pd.Series([1e9], index=pd.to_datetime(["2024-03-31"]))
    fig = revenue_pe_figure(revenue, pd.Series(dtype=float), peg=peg)
    assert tuple(fig.layout.yaxis2.range) == (0, 4)
    assert "capped at 4" in fig.layout.annotations[1].text
