"""Single fiscal quarters from SEC Company Facts duration facts of any length."""

import pandas as pd


def quarters_from_periods(entries):
    """Single fiscal quarters from SEC duration facts of any length: a 3-month fact as it is, else the difference of
    two year-to-date facts with the same start that end about a quarter apart (Q4 = full year - nine months). Unlike
    the fiscal-period grouping it does not need every quarter of the year, so a tag a company used for only part of
    a year still gives those quarters. By quarter end; the latest filing wins."""

    periods = {}
    for entry in entries:
        if entry.get("form") not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
            continue
        start, end = pd.to_datetime(entry.get("start"), errors="coerce"), pd.to_datetime(entry.get("end"), errors="coerce")
        value = pd.to_numeric(entry.get("val"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or pd.isna(value) or not 60 <= (end - start).days <= 380:
            continue
        key = (start, end)
        filed = str(entry.get("filed", ""))
        if key not in periods or filed >= periods[key][1]:
            periods[key] = (float(value), filed)
    quarters = {}
    for (start, end), (value, _) in periods.items():
        if 75 <= (end - start).days <= 100:
            quarters[end] = value
    for (start, end), (value, _) in sorted(periods.items()):
        if any(abs((end - known).days) <= 14 for known in quarters):
            continue
        earlier = [(e, v) for (s, e), (v, _) in periods.items() if s == start and 75 <= (end - e).days <= 100]
        if earlier:
            quarters[end] = value - max(earlier)[1]
    return pd.Series(quarters, dtype=float).sort_index() if quarters else pd.Series(dtype=float)
