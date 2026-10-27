import numpy as np
import pandas as pd

from analysis.economy import (
    asset_table,
    assets_figure,
    drawdowns,
    indicator_values,
    latest_text,
    macro_figure,
    rebase,
    recession_periods,
    window_start,
)


def test_rebase_starts_on_the_first_common_weekday():
    days = pd.date_range("2024-01-01", periods=60, freq="D")  # includes weekends, like bitcoin
    closes = pd.DataFrame({"S&P 500": np.linspace(100, 160, 60), "Bitcoin": np.linspace(10, 40, 60)}, index=days)
    closes.loc[:"2024-01-09", "Bitcoin"] = np.nan
    rebased, first = rebase(closes)
    assert first == pd.Timestamp("2024-01-10")
    assert (rebased.iloc[0] == 100).all()
    assert (rebased.index.dayofweek < 5).all()
    later, first_later = rebase(closes, start="2024-02-01")
    assert first_later == pd.Timestamp("2024-02-01") and later.iloc[0].tolist() == [100, 100]


def test_rebase_without_overlap():
    days = pd.bdate_range("2024-01-01", periods=10)
    closes = pd.DataFrame({"A": [1.0] * 10, "B": [np.nan] * 10}, index=days)
    rebased, first = rebase(closes)
    assert first is None and rebased.empty


def test_drawdown_and_table():
    days = pd.bdate_range("2020-01-01", periods=600)
    up = pd.Series(np.linspace(100, 200, 600), index=days)
    dip = up.copy()
    dip.iloc[300:350] = dip.iloc[300] * 0.6
    rebased = pd.DataFrame({"S&P 500": up, "Gold": dip})
    assert drawdowns(rebased)["S&P 500"].min() == 0
    assert round(drawdowns(rebased)["Gold"].min()) == -40
    table = asset_table(rebased / rebased.iloc[0] * 100)
    assert table.at["S&P 500", "Correlation with S&P 500"] == 1.0
    assert round(table.at["S&P 500", "Total return %"]) == 100
    assert round(table.at["Gold", "Worst fall %"]) == -40


def test_assets_figure_has_three_panels_with_their_own_legends():
    days = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(1)
    rebased = pd.DataFrame({"S&P 500": 100 * np.cumprod(1 + rng.normal(0, 0.01, 600)),
                            "Gold": 100 * np.cumprod(1 + rng.normal(0, 0.01, 600))}, index=days)
    fig = assets_figure(rebased)
    assert {trace.legend for trace in fig.data} == {"legend", "legend2", "legend3"}
    assert fig.layout.yaxis.type == "log"
    # Only the other asset gets a correlation line.
    assert [trace.name for trace in fig.data if trace.legend == "legend3"] == ["Gold"]


def test_inflation_is_year_over_year():
    months = pd.date_range("2020-01-01", periods=25, freq="MS")
    cpi = pd.Series(100 * 1.002 ** np.arange(25), index=months)
    yoy = indicator_values(cpi, "yoy")
    assert yoy.index[0] == months[12]
    assert round(yoy.iloc[0], 2) == round((1.002 ** 12 - 1) * 100, 2)
    assert indicator_values(cpi, "level").equals(cpi)


def test_recession_periods():
    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    flags = pd.Series([0, 1, 1, 0, 0, 1, 1, 1], index=months)
    assert recession_periods(flags) == [(months[1], months[3]), (months[5], pd.Timestamp("2020-08-31"))]


def test_macro_figure_puts_the_sp500_on_a_right_axis_in_every_panel():
    months = pd.date_range("2000-01-01", periods=300, freq="MS")
    values = {"Unemployment rate": pd.Series(np.linspace(4, 6, 300), index=months),
              "Fed funds rate": pd.Series(np.linspace(5, 1, 300), index=months)}
    sp500 = pd.Series(np.linspace(1000, 6000, 6000), index=pd.bdate_range("2000-01-01", periods=6000))
    recession = pd.Series([0] * 100 + [1] * 18 + [0] * 182, index=months)
    fig = macro_figure(values, sp500, recession, start=window_start(months[-1], 10))
    overlays = [trace for trace in fig.data if trace.name == "S&P 500 (right axis, log)"]
    assert len(overlays) == 2
    assert all(fig.layout[f"yaxis{trace.yaxis[1:]}"].side == "right" for trace in overlays)
    assert all(fig.layout[f"yaxis{trace.yaxis[1:]}"].type == "log" for trace in overlays)
    # The 2008-2009 recession lies before the 10-year window, so no shading and no legend entry for it.
    assert not any(trace.name == "US recession (shaded)" for trace in fig.data)
    full = macro_figure(values, sp500, recession)
    assert any(trace.name == "US recession (shaded)" for trace in full.data)
    assert macro_figure({}, sp500) is None


def test_latest_text():
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    text = latest_text("Unemployment rate", pd.Series(np.linspace(4.0, 5.3, 14), index=months))
    assert text == "Unemployment rate: 5.30% (Feb 2025; 4.10% a year earlier)"
