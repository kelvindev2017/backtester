import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from analysis.model_diagram import (  # noqa: E402
    BOTTOM, POSITIONS, block_signals_figure, format_value, items_table, model_diagram, model_price_figure, parameter_table,
)
from analysis.valuation_model import BLOCKS, model_params, preset_params, preset_register, run_model  # noqa: E402
from tests.test_valuation_model import make_close, make_quarters, register  # noqa: E402


@pytest.fixture
def result():
    quarters = make_quarters()
    close = make_close(quarters, pe=14.0)
    peers = pd.DataFrame({"AAA": 20.0}, index=close.index)
    items = register([
        ["Cyclicality", "Risk", "General", "Beyond horizon", None, None, 2, "Modelled", "", ""],
        ["Export rules", "Risk", "Event", "In horizon", 20, 60, None, "Parked", "", ""],
    ])
    return run_model(quarters, close, items, model_params({"peer_weight": 1.0, "anchor_source": "Peers", "case_spread": "Fixed gaps"}), peers)


def texts(fig):
    return " | ".join(annotation.text for annotation in fig.layout.annotations if annotation.text)


def test_diagram_shows_every_block_with_its_value_and_both_rates(result):
    fig = model_diagram(result, selected="risk_weighting")
    text = texts(fig)
    for block in BLOCKS.values():
        assert f"<b>{block['title']}</b>" in text, block["title"]
    assert "Quarterly rate" in text and "Daily rate" in text and "<b>Hold</b>" in text
    assert "<b>Price per share</b><br>262.32" in text and "P/E 18.0x" in text and "1 modelled · 1 parked" in text
    # One clickable marker per block, carrying the block's key; the hover lists inputs, outputs and parameters.
    markers = fig.data[-1]
    assert [row[0] for row in markers.customdata] == list(BLOCKS)
    hover = markers.hovertext[list(BLOCKS).index("risk_weighting")]
    assert "In: Bear EPS" in hover and "Out: Risk-adjusted EPS" in hover and "Param: Bear weight = 25" in hover
    # The chosen block has the heavy border; every block sits inside the drawing.
    assert sum(shape.line.width == 3 for shape in fig.layout.shapes) == 1
    assert all(0 < x < 10 and 0 < y < BOTTOM for x, y in POSITIONS.values())
    assert fig.layout.dragmode is False and fig.layout.clickmode == "event+select"


def test_diagram_draws_without_data():
    empty = run_model(pd.DataFrame(), pd.Series(dtype=float), None, None)
    assert "n/a" in texts(model_diagram(empty, mobile=True))
    assert model_price_figure(empty).data


def test_block_charts_split_inputs_and_outputs(result):
    fig = block_signals_figure(result, "scenario_engine")
    titles = [annotation.text for annotation in fig.layout.annotations]
    assert titles[0].startswith("Input: Clean base EPS") and any(title.startswith("Output: Bear EPS") for title in titles)
    names = [trace.name for trace in fig.data]
    assert {"Bear EPS", "Base EPS", "Bull EPS", "Revenue per share growth, year on year"} <= set(names)
    # Quarterly signals are held between updates, so they are drawn as steps.
    assert all(trace.line.shape == "hv" for trace in fig.data)
    daily = block_signals_figure(result, "multiple_builder")
    assert any(trace.name == "Acceptable P/E" and trace.line.shape == "linear" for trace in daily.data)
    for block in BLOCKS:
        assert block_signals_figure(result, block) is not None or block == "item_tracker" or BLOCKS[block]["outputs"]


def test_price_chart_has_the_model_the_close_and_the_range(result):
    fig = model_price_figure(result, "TEST")
    names = [trace.name for trace in fig.data]
    assert names[:5] == ["Bull case", "Bear case", "Modeled fair value", "Modeled fair value, last quarter (no forward assumptions)", "Close"]
    assert "Acceptable P/E" in names
    model = next(trace for trace in fig.data if trace.name == "Modeled fair value")
    assert np.nanmax(np.asarray(model.y, dtype=float)) == pytest.approx(result["daily"]["model_price"].max())


def test_parameter_and_item_tables(result):
    weights = parameter_table(result, "risk_weighting")
    assert weights["Parameter"].tolist()[:3] == ["Bear weight", "Base weight", "Bull weight"]
    assert weights["Value"].tolist()[3:] == [25.0, 50.0, 25.0]
    assert parameter_table(result, "product").empty
    table = items_table(result)
    assert table["Goes to"].tolist() == ["Multiple builder", "Scenario engine"]
    assert table["Effect"].tolist() == ["-2.0 P/E points", "-12.0 % of EPS"] and table["In the model"].tolist() == [True, False]
    preset = run_model(pd.DataFrame(), pd.Series([30.0, 31.0], index=pd.bdate_range("2026-11-09", periods=2)),
                       preset_register("ASMVY"), preset_params("ASMVY"))
    assert items_table(preset)["In the model"].sum() == 2  # the cyclicality item is parked: through-cycle EPS covers it


def test_values_are_formatted_by_unit():
    assert format_value(12.345, "%", "percent") == "+12.3%" and format_value(20.0, "%", "share") == "20.0%"
    assert format_value(17.54, "x") == "17.5x" and format_value(-6, "pts") == "-6.0 pts"
    assert format_value(2.5e9) == "2.50B" and format_value(3, "", "count") == "3" and format_value(np.nan) == "n/a"


def test_blocks_with_forward_assumptions_show_now_and_forward(result):
    text = texts(model_diagram(result))
    assert text.count("last Q ") >= 6 and "fwd " in text
    assert "fwd base" in text and "last Q none" in text
    # No policy rate in this run: the rate model has no now/fwd pair.
    from analysis.model_diagram import now_and_forward_text
    from analysis.valuation_model import latest_values
    assert now_and_forward_text("rate_model", latest_values(result)) is None
    assert now_and_forward_text("income_map", latest_values(result)) is None
    assert any("(forward)" in label for label in parameter_table(result, "scenario_engine")["Parameter"])


def test_price_against_growth_margin_and_earnings():
    from analysis.model_diagram import price_and_fundamentals_figure
    quarters = make_quarters().assign(Gross_Profit=lambda frame: frame["Revenue"] * 0.45)
    close = make_close(quarters)
    result = run_model(quarters, close, None, model_params({"anchor_source": "Fundamentals", "peer_weight": 1.0}))
    fig = price_and_fundamentals_figure(result, "TEST")
    axes = {trace.name: trace.yaxis for trace in fig.data}
    assert axes["Close"] == axes["Modeled fair value"] == "y"
    assert axes["Revenue growth"] == axes["Gross margin"] == "y2"
    assert axes["Earnings (highest = 100)"] == "y3"
    earnings = next(trace for trace in fig.data if trace.name == "Earnings (highest = 100)")
    assert np.nanmax(np.asarray(earnings.y, dtype=float)) == pytest.approx(100.0)
    assert next(trace for trace in fig.data if trace.name == "Gross margin").y[-1] == pytest.approx(45.0)


def test_premium_figure_shows_market_against_model_and_the_next_quarter():
    from analysis.model_diagram import premium_figure
    from analysis.premium_model import calibrate
    from tests.test_premium_model import cyclical_stock, run
    results, close = cyclical_stock()
    fig = premium_figure(calibrate(run(results, close)), "TEST")
    names = [trace.name for trace in fig.data]
    assert {"Market trailing P/E", "Justified by fundamentals", "Justified + explained premium", "Close",
            "Forecast a quarter before", "Base next quarter"} <= set(names)


def test_block_chart_hides_through_cycle_lines_when_unused(result):
    fig = block_signals_figure(result, "risk_weighting")
    names = {trace.name for trace in fig.data}
    assert "Risk-adjusted EPS" in names
    assert not names & {"Through-cycle EPS", "Through-cycle EPS on (1 = yes)", "Past years' EPS, summed",
                        "Next year's EPS (risk-adjusted)"}
