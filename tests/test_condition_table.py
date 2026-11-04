import numpy as np
import pandas as pd

from analysis.condition_table import (
    default_row,
    default_table,
    match_options,
    normalize,
    parse_line,
    parse_numbers,
    row_condition,
)
from analysis.screener import condition_label, evaluate_rate_of_change, rate_of_change
from tests.conftest import make_bars


def test_parse_line_and_numbers():
    assert parse_line("Close") == ("Close", None)
    assert parse_line("sma50") == ("SMA", 50)
    assert parse_line(" EMA 20 ") == ("EMA", 20)
    assert parse_line("12.5") == ("Value", 12.5)
    assert parse_line("SMA") is None
    assert parse_numbers("0 to 30", True) == (0.0, 30.0)
    assert parse_numbers("-5 - 5", True) == (-5.0, 5.0)
    assert parse_numbers("15%", False) == (15.0,)
    assert parse_numbers("abc", False) is None


def test_default_table_reads_as_the_default_screen():
    labels = [condition_label(row_condition(row)[0]) for row in default_table().to_dict("records")]
    assert labels == ["Close > SMA 50", "SMA 50 > SMA 200", "Close > SMA 200"]


def test_every_signal_default_row_is_complete():
    options = {"Index": ["S&P 500"], "Sector": ["Energy"], "Industry": ["Banks"], "Analyst rating": ["Buy"]}
    for signal in ["Moving average", "Rate of change", "Price", "P/E", "Perf %", "Recent earnings date"]:
        condition, problem = row_condition(default_row(signal), options)
        assert condition is not None, (signal, problem)
    # A category needs its values typed in.
    assert row_condition(default_row("Sector"), options)[0] is None
    condition, _ = row_condition({**default_row("Sector"), "Value": "energy"}, options)
    assert condition["values"] == ["Energy"]


def test_match_options_by_word():
    sectors = ["Information Technology", "Energy", "Health Care"]
    assert match_options("Tech, energy", sectors) == ["Information Technology", "Energy"]
    assert match_options("nothing", sectors) == []


def test_rate_of_change_condition():
    condition, problem = row_condition({"Signal": "Rate of change", "Line": "SMA 200", "Op": ">", "Value": "0", "Days": 20})
    assert problem == "" and condition["line_kind"] == "SMA" and condition["days"] == 20
    assert condition_label(condition) == "SMA 200 change over 20d > 0%"
    rising = make_bars(np.linspace(50, 150, 400))
    falling = make_bars(np.linspace(150, 50, 400))
    assert evaluate_rate_of_change(rising, condition) and not evaluate_rate_of_change(falling, condition)
    close = rising["Close"]
    expected = (close.rolling(200).mean().iloc[-1] / close.rolling(200).mean().iloc[-21] - 1) * 100
    assert np.isclose(rate_of_change(rising, condition), expected)
    # Too little history for the line and look-back fails the condition.
    assert not evaluate_rate_of_change(rising.iloc[:210], condition)
    between = {**condition, "operator": "Between", "value": 0.0, "value2": 100.0}
    assert evaluate_rate_of_change(rising, between)


def test_normalize_removes_ticked_rows_and_refills_a_changed_signal():
    before = default_table()
    after = before.copy()
    after.loc[0, "Remove"] = True
    after.loc[1, "Signal"] = "P/E"
    table = normalize(before, after)
    assert len(table) == 2
    assert table.loc[0, "Signal"] == "P/E" and table.loc[0, "Op"] == "Between"
    assert table.loc[1, "Value"] == "SMA 200" and not table["Remove"].any()
    edited = before.copy()
    edited.loc[2, "Value"] = "EMA 21"
    assert normalize(before, edited).loc[2, "Value"] == "EMA 21"
