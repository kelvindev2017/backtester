"""Screener conditions as rows of one editable table.

Each row is one condition, with the same few cells whatever its signal group:

    Remove | Signal | Line | Op | Value | Days

What a cell means follows the signal:

- Moving average: Line "SMA 50" (or "Close", "EMA 20"), Op, Value "SMA 200" (or "Close", or a number).
- Rate of change: Line "SMA 200", Op, Value in % (or "0 to 5" with Between), Days = the look-back.
- Number groups (Price, P/E, Mkt cap ...): Op, Value (or "0 to 30" with Between). Perf % takes its period
  ("1W", "1M", "3M", "6M", "YTD", "1Y") in Line.
- Index / Sector / Industry / Analyst rating: Op "is any of" or "is none of", Value a comma-separated list
  ("Tech, Energy"); a word matches any option that contains it.
- Recent / Upcoming earnings date: Days.

``row_condition`` turns a row into the condition dict the screener evaluates, or explains what is missing.
"""

import re

import pandas as pd

from analysis.screener import OPERATORS as MA_OPERATORS, RATE_OF_CHANGE
from analysis.signal_groups import CATEGORY_OPERATORS, MOVING_AVERAGE, NUMBER_OPERATORS, SIGNAL_GROUPS

COLUMNS = ["Remove", "Signal", "Line", "Op", "Value", "Days"]
SIGNALS = [MOVING_AVERAGE, RATE_OF_CHANGE] + [group for group in SIGNAL_GROUPS if group != MOVING_AVERAGE]
ALL_OPERATORS = list(dict.fromkeys(MA_OPERATORS + NUMBER_OPERATORS + CATEGORY_OPERATORS))

DEFAULT_ROWS = [
    {"Signal": MOVING_AVERAGE, "Line": "Close", "Op": ">", "Value": "SMA 50"},
    {"Signal": MOVING_AVERAGE, "Line": "SMA 50", "Op": ">", "Value": "SMA 200"},
    {"Signal": MOVING_AVERAGE, "Line": "Close", "Op": ">", "Value": "SMA 200"},
]

_NUMBER = r"-?\d+(?:\.\d+)?"


def default_row(signal):
    """A new row for ``signal`` with sensible inputs filled in."""

    row = {"Remove": False, "Signal": signal, "Line": "", "Op": "", "Value": "", "Days": None}
    if signal == MOVING_AVERAGE:
        row.update(Line="Close", Op=">", Value="SMA 50")
    elif signal == RATE_OF_CHANGE:
        row.update(Line="SMA 200", Op=">", Value="0", Days=20)
    else:
        spec = SIGNAL_GROUPS[signal]
        kind = spec["kind"]
        if kind == "number":
            default = spec.get("default", (">", 0.0))
            row.update(Op=default[0], Value=f"{default[1]:g}" if default[0] != "Between"
                       else f"{default[1]:g} to {default[2]:g}")
            if spec.get("periods"):
                row["Line"] = "1M"
        elif kind == "category":
            row.update(Op="is any of")
        else:
            row.update(Days=int(spec.get("default", 30)))
    return row


def default_table():
    rows = [{**default_row(row["Signal"]), **row} for row in DEFAULT_ROWS]
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_line(text):
    """'Close' -> ('Close', None); 'SMA 50' / 'sma50' -> ('SMA', 50); a number -> ('Value', 50.0); else None."""

    text = str(text or "").strip()
    if text.lower() == "close":
        return "Close", None
    match = re.fullmatch(r"(sma|ema)\s*(\d+)", text, re.IGNORECASE)
    if match and int(match.group(2)) >= 1:
        return match.group(1).upper(), int(match.group(2))
    if re.fullmatch(_NUMBER, text):
        return "Value", float(text)
    return None


def parse_numbers(text, between):
    """One number, or two for Between ('0 to 30', '0-30', '-5 to 5'). None when it doesn't parse."""

    text = str(text or "").strip().replace("%", "")
    if between:
        match = re.fullmatch(rf"\s*({_NUMBER})\s*(?:to|-|–|,)\s*({_NUMBER})\s*", text)
        return (float(match.group(1)), float(match.group(2))) if match else None
    return (float(text),) if re.fullmatch(_NUMBER, text) else None


def match_options(text, options):
    """Options named by a comma-separated list; a word matches every option that contains it."""

    chosen = []
    for word in [part.strip().lower() for part in str(text or "").split(",") if part.strip()]:
        exact = [option for option in options if option.lower() == word]
        for option in exact or [option for option in options if word in option.lower()]:
            if option not in chosen:
                chosen.append(option)
    return chosen


def _blank(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def row_condition(row, category_options=None):
    """(condition dict, "") for a complete row, or (None, what is wrong)."""

    signal, line, op, value, days = (row.get(column) for column in ["Signal", "Line", "Op", "Value", "Days"])
    if signal not in SIGNALS:
        return None, "pick a signal"
    if signal == MOVING_AVERAGE:
        left, right = parse_line(line), parse_line(value)
        if left is None or left[0] == "Value":
            return None, "Line: Close, SMA 50 or EMA 20"
        if op not in MA_OPERATORS:
            return None, "Op: " + ", ".join(MA_OPERATORS)
        if right is None:
            return None, "Value: Close, SMA 200, EMA 20 or a number"
        return {"group": MOVING_AVERAGE, "left_kind": left[0], "left_period": left[1], "operator": op,
                "right_kind": right[0], "right_period": right[1]}, ""
    if signal == RATE_OF_CHANGE:
        left = parse_line(line)
        if left is None or left[0] == "Value":
            return None, "Line: Close, SMA 200 or EMA 20"
        if op not in NUMBER_OPERATORS:
            return None, "Op: " + ", ".join(NUMBER_OPERATORS)
        numbers = parse_numbers(value, op == "Between")
        if numbers is None:
            return None, "Value: a % change" + (", e.g. 0 to 5" if op == "Between" else ", e.g. 0")
        if _blank(days) or int(days) < 1:
            return None, "Days: the look-back, e.g. 20"
        condition = {"group": RATE_OF_CHANGE, "line_kind": left[0], "line_period": left[1], "days": int(days),
                     "operator": op, "value": numbers[0]}
        if op == "Between":
            condition["value2"] = numbers[1]
        return condition, ""
    spec = SIGNAL_GROUPS[signal]
    kind = spec["kind"]
    if kind == "number":
        if op not in NUMBER_OPERATORS:
            return None, "Op: " + ", ".join(NUMBER_OPERATORS)
        numbers = parse_numbers(value, op == "Between")
        if numbers is None:
            return None, "Value: a number" + (", e.g. 0 to 30" if op == "Between" else "")
        condition = {"group": signal, "operator": op, "value": numbers[0]}
        if op == "Between":
            condition["value2"] = numbers[1]
        if spec.get("periods"):
            period = str(line or "").strip().upper()
            if period not in spec["periods"]:
                return None, "Line: " + ", ".join(spec["periods"])
            condition["period"] = period
        return condition, ""
    if kind == "category":
        if op not in CATEGORY_OPERATORS:
            return None, "Op: is any of / is none of"
        options = (category_options or {}).get(signal, [])
        values = match_options(value, options)
        if not values:
            return None, ("Value: e.g. " + ", ".join(options[:2])) if options else "load the stocks first"
        return {"group": signal, "operator": op, "values": values}, ""
    if _blank(days) or int(days) < 1:
        return None, "Days: e.g. 30"
    return {"group": signal, "days": int(days)}, ""


def normalize(before, after):
    """The table after an edit: ticked Remove rows dropped, rows whose Signal changed refilled with that
    signal's defaults, new rows (added from the grid) filled in."""

    rows = []
    previous = before.to_dict("records")
    for position, row in enumerate(after.to_dict("records")):
        if row.get("Remove") is True:
            continue
        old = previous[position] if position < len(previous) else None
        signal = row.get("Signal") if row.get("Signal") in SIGNALS else MOVING_AVERAGE
        if old is None or old.get("Signal") != signal:
            row = default_row(signal)
        rows.append({**row, "Remove": False})
    return pd.DataFrame(rows, columns=COLUMNS)
