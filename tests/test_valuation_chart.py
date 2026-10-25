import numpy as np
import pandas as pd

from analysis.valuation_chart import (
    correlation_text,
    growth_price_correlation,
    price_yoy,
    yoy_change,
    eps_estimate_quarters,
    quarterly_eps,
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


def test_figure_stacks_revenue_above_pe_and_marks_estimates():
    quarters = pd.date_range("2024-01-31", periods=8, freq="QE-JAN")
    revenue = pd.Series(np.linspace(20e9, 60e9, 8), index=quarters)
    pe = pd.Series(np.linspace(60, 30, 400), index=pd.bdate_range("2024-01-01", periods=400))
    estimates = pd.Series([70e9], index=[quarters[-1] + pd.DateOffset(months=3)])
    fig = revenue_pe_figure(revenue, pe, estimates)
    names = [trace.name for trace in fig.data]
    assert names == ["Trailing P/E", "Revenue (reported)", "Revenue (analyst estimate)"]
    # Two separate y-axes in stacked rows; without a close there is no right-hand axis.
    assert fig.data[0].yaxis == "y2" and fig.data[1].yaxis == "y"
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
    assert fig.data[0].yaxis == "y3" and fig.data[1].yaxis == "y"  # revenue, P/E, then PEG at the bottom


def test_peg_axis_is_capped_when_growth_restarts_from_near_zero():
    peg = pd.Series([0.5, 12.0, 0.8], index=pd.to_datetime(["2022-01-03", "2023-06-01", "2024-01-01"]))
    revenue = pd.Series([1e9], index=pd.to_datetime(["2024-03-31"]))
    fig = revenue_pe_figure(revenue, pd.Series(dtype=float), peg=peg)
    assert tuple(fig.layout.yaxis3.range) == (0, 4)
    assert "capped at 4" in fig.layout.annotations[2].text


def test_eps_moves_to_the_quarter_it_reports():
    quarter_ends = pd.to_datetime(["2026-01-25", "2026-04-26", "2026-07-26"])
    earnings = pd.DataFrame({"Reported EPS": [1.62, 1.87, 2.22, np.nan]},
                            index=pd.to_datetime(["2026-02-25", "2026-05-20", "2026-08-26", "2026-11-17"]))
    eps = quarterly_eps(earnings, quarter_ends)
    assert list(eps.index) == list(quarter_ends) and list(eps) == [1.62, 1.87, 2.22]
    assert quarterly_eps(pd.DataFrame(), quarter_ends).empty
    # A quarter the revenue source skipped still gets its EPS bar, a month before the announcement.
    gap = quarterly_eps(earnings, quarter_ends[[0, 2]])
    assert gap.index[1] == pd.Timestamp("2026-04-20") and gap.iloc[1] == 1.87
    estimates = pd.DataFrame({"avg": [2.47, 2.75, 9.3]}, index=["0q", "+1q", "0y"])
    ahead = eps_estimate_quarters(estimates, quarter_ends[-1])
    assert list(ahead.index) == [pd.Timestamp("2026-10-26"), pd.Timestamp("2027-01-26")]


def test_eps_panel_sits_between_revenue_and_pe():
    quarters = pd.to_datetime(["2026-04-26", "2026-07-26"])
    revenue = pd.Series([80e9, 96e9], index=quarters)
    peg = pd.Series([0.4], index=[pd.Timestamp("2026-07-01")])
    eps = pd.Series([1.87, 2.22], index=quarters)
    eps_ahead = pd.Series([2.47], index=[pd.Timestamp("2026-10-26")])
    fig = revenue_pe_figure(revenue, pd.Series(dtype=float), peg=peg, eps=eps, eps_estimates=eps_ahead)
    assert [trace.name for trace in fig.data] == ["Trailing PEG", "Revenue (reported)", "EPS (reported)",
                                                  "EPS (analyst estimate)"]
    assert fig.data[1].yaxis == "y" and fig.data[2].yaxis == "y2" and fig.data[3].yaxis == "y2"
    assert fig.data[0].yaxis == "y4"  # PEG last
    assert fig.layout.annotations[1].text.startswith("EPS by quarter")


def test_yoy_change_matches_by_date_and_skips_losses():
    quarters = pd.to_datetime(["2024-03-31", "2024-06-30", "2025-03-31", "2025-06-30", "2025-09-30"])
    series = pd.Series([2.0, -1.0, 3.0, 0.5, 1.0], index=quarters)
    change = yoy_change(series)
    assert change.loc["2025-03-31"] == 50.0
    assert np.isnan(change.loc["2025-06-30"])  # year-ago loss: no meaningful % change
    assert np.isnan(change.loc["2025-09-30"])  # no quarter a year earlier


def test_price_yoy_uses_the_last_close_a_year_back():
    index = pd.bdate_range("2024-01-01", "2025-06-30")
    close = pd.Series(np.linspace(100, 250, len(index)), index=index)
    change = price_yoy(close)
    assert np.isnan(change.iloc[0])
    when = pd.Timestamp("2025-06-30")
    assert change.loc[when] == (close.loc[when] / close.loc[:when - pd.Timedelta(days=365)].iloc[-1] - 1) * 100


def test_growth_price_correlation_finds_the_lag():
    quarters = pd.date_range("2018-03-31", periods=24, freq="QE")
    rng = np.random.default_rng(1)
    growth = pd.Series(rng.normal(30, 40, len(quarters)), index=quarters)
    # The price moves a quarter after growth: its YoY change at each quarter end is last quarter's growth.
    daily = pd.date_range(quarters[0], quarters[-1] + pd.Timedelta(days=5))
    price_change = growth.shift(1).reindex(daily, method="ffill")
    result = growth_price_correlation(growth, price_change)
    assert result["1 quarter later"][0] > 0.99
    assert abs(result["same quarter"][0]) < 0.6
    assert result["same quarter"][1] == 23  # the first quarter has no price change yet
    assert "Not enough quarters" in correlation_text({"EPS": growth_price_correlation(growth.iloc[:5], price_change)})
    assert "EPS YoY vs price YoY (23 quarters)" in correlation_text({"EPS": result})


def test_growth_panel_is_the_first_row():
    quarters = pd.date_range("2023-03-31", periods=12, freq="QE")
    revenue = pd.Series(np.linspace(10e9, 40e9, 12), index=quarters)
    eps = pd.Series(np.linspace(0.5, 2.0, 12), index=quarters)
    price_change = pd.Series(np.linspace(-20, 500, 400), index=pd.bdate_range("2024-01-01", periods=400))
    fig = revenue_pe_figure(revenue, pd.Series(dtype=float), eps=eps, price_change=price_change)
    names = [trace.name for trace in fig.data]
    assert names[-3:] == ["Price YoY %", "Revenue YoY %", "EPS YoY %"]
    assert fig.data[-1].yaxis == "y" and tuple(fig.layout.yaxis.range)[1] == 300
    assert [annotation.text.split(" ")[0] for annotation in fig.layout.annotations[:4]] == [
        "YoY", "Revenue", "EPS", "Trailing"]
    assert any("capped at 300%" in annotation.text for annotation in fig.layout.annotations)


def test_legend_follows_panel_order():
    quarters = pd.date_range("2023-03-31", periods=12, freq="QE")
    revenue = pd.Series(np.linspace(10e9, 40e9, 12), index=quarters)
    pe = pd.Series(np.linspace(60, 30, 400), index=pd.bdate_range("2024-01-01", periods=400))
    price_change = pd.Series(np.linspace(-20, 80, 400), index=pe.index)
    fig = revenue_pe_figure(revenue, pe, price_change=price_change)
    by_rank = [trace.name for trace in sorted(fig.data, key=lambda trace: trace.legendrank)]
    assert by_rank == ["Price YoY %", "Revenue YoY %", "Revenue (reported)", "Trailing P/E"]


def test_price_panel_sits_right_above_pe_and_peg():
    quarters = pd.date_range("2023-03-31", periods=12, freq="QE")
    revenue = pd.Series(np.linspace(10e9, 40e9, 12), index=quarters)
    days = pd.bdate_range("2024-01-01", periods=400)
    pe = pd.Series(np.linspace(60, 30, 400), index=days)
    peg = pd.Series(np.linspace(1.5, 0.5, 400), index=days)
    close = pd.Series(np.linspace(100, 180, 400), index=days)
    fig = revenue_pe_figure(revenue, pe, peg=peg, close=close)
    titles = [annotation.text for annotation in fig.layout.annotations[:4]]
    assert titles == ["Revenue by quarter, billions", "Daily close (split-adjusted)", "Trailing P/E", "Trailing PEG"]
    close_trace = next(trace for trace in fig.data if trace.name == "Close")
    assert close_trace.yaxis == "y2" and close_trace.y[-1] == 180
    assert fig.layout.yaxis2.type == "linear"
    assert revenue_pe_figure(revenue, pe, peg=peg, close=close, log_price=True).layout.yaxis2.type == "log"


def test_close_on_a_right_hand_axis_in_every_other_panel_and_a_legend_under_each():
    quarters = pd.date_range("2023-03-31", periods=12, freq="QE")
    revenue = pd.Series(np.linspace(10e9, 40e9, 12), index=quarters)
    days = pd.bdate_range("2024-01-01", periods=400)
    pe = pd.Series(np.linspace(60, 30, 400), index=days)
    close = pd.Series(np.linspace(100, 180, 400), index=days)
    fig = revenue_pe_figure(revenue, pe, close=close, log_price=True)
    overlays = [trace for trace in fig.data if trace.name == "Close (right axis)"]
    # Revenue and P/E panels get the overlay; the close panel itself does not.
    overlaid = {fig.layout[trace.yaxis.replace("y", "yaxis")].overlaying for trace in overlays}
    assert overlaid == {"y", "y3"}
    for trace in overlays:
        axis = fig.layout[trace.yaxis.replace("y", "yaxis")]
        assert axis.side == "right" and axis.type == "log"
    # One legend per panel, each right under its panel.
    revenue_trace = next(trace for trace in fig.data if trace.name == "Revenue (reported)")
    pe_trace = next(trace for trace in fig.data if trace.name == "Trailing P/E")
    assert revenue_trace.legend == "legend" and pe_trace.legend == "legend3"
    assert {trace.legend for trace in overlays} == {"legend", "legend3"}
    for legend, axis in [("legend", fig.layout.yaxis), ("legend2", fig.layout.yaxis2), ("legend3", fig.layout.yaxis3)]:
        assert axis.domain[0] - 0.1 < fig.layout[legend].y < axis.domain[0]
