"""Revenue, EPS and valuation (P/E, PEG) history for one company, drawn as
stacked panels on a shared date axis.

Each measure keeps its own panel. The daily close is also drawn in every panel on
a right-hand axis, so each measure can be read against the price; where the two
lines cross depends only on how each axis was scaled, so a crossing is not a signal.
Each panel has its own legend right under it.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.fundamentals import ordinal

VALUATION_HISTORY_YEARS = 8
ACTUAL_COLOR = "royalblue"
ESTIMATE_COLOR = "rgba(65, 105, 225, 0.35)"
PE_COLOR = "goldenrod"
PEG_COLOR = "seagreen"
FORWARD_COLOR = "mediumpurple"
PEG_AXIS_CAP = 4
EPS_COLOR = "darkorange"
EPS_ESTIMATE_COLOR = "rgba(255, 140, 0, 0.35)"
PRICE_COLOR = "dimgray"
CLOSE_OVERLAY_COLOR = "rgba(105, 105, 105, 0.55)"
UP_COLOR, DOWN_COLOR = "seagreen", "indianred"
VOLUME_UP_COLOR, VOLUME_DOWN_COLOR = "rgba(46, 139, 87, 0.6)", "rgba(205, 92, 92, 0.6)"
GROWTH_AXIS_CAP = 300
LAGS = {"price 1 quarter earlier": 1, "same quarter": 0, "1 quarter later": -1, "1 year later": -4}


def revenue_estimate_quarters(estimates, last_quarter_end):
    """Analyst average revenue for the current and next fiscal quarter as a Series by quarter end.

    ``estimates`` is Yahoo's ``revenue_estimate`` table (rows ``0q``, ``+1q``, ...,
    column ``avg``). Yahoo gives no dates, so the quarters are placed one and two
    quarters after the last reported quarter end.
    """

    if estimates is None or getattr(estimates, "empty", True) or "avg" not in estimates.columns:
        return pd.Series(dtype=float)
    if last_quarter_end is None or pd.isna(last_quarter_end):
        return pd.Series(dtype=float)
    values = {}
    for step, period in [(1, "0q"), (2, "+1q")]:
        if period in estimates.index:
            value = pd.to_numeric(estimates.at[period, "avg"], errors="coerce")
            if pd.notna(value) and value > 0:
                values[pd.Timestamp(last_quarter_end) + pd.DateOffset(months=3 * step)] = float(value)
    return pd.Series(values, dtype=float)


def pe_summary(trailing_pe):
    """Current trailing P/E against its own history: current, median, percentile, low and its date."""

    series = trailing_pe.dropna() if trailing_pe is not None else pd.Series(dtype=float)
    if series.empty:
        return {}
    current = float(series.iloc[-1])
    return {
        "current": current,
        "median": float(series.median()),
        "percentile": float((series < current).mean()),
        "low": float(series.min()),
        "low_date": series.idxmin(),
        "start": series.index.min(),
    }


def pe_summary_text(summary, forward_pe=None):
    """One caption line for :func:`pe_summary`, plus today's forward P/E when Yahoo has one."""

    if not summary:
        return "No trailing P/E history (no reported EPS, or EPS was negative)."
    text = (
        f"Trailing P/E {summary['current']:.1f}x sits at the {ordinal(round(summary['percentile'] * 100))} percentile "
        f"since {summary['start']:%Y} (median {summary['median']:.1f}x, low {summary['low']:.1f}x in "
        f"{summary['low_date']:%b %Y})."
    )
    if forward_pe is not None and np.isfinite(forward_pe) and forward_pe > 0:
        text += (
            f" Forward P/E today is {forward_pe:.1f}x (Yahoo, on next fiscal year's estimated EPS). "
            "The chart's forward P/E uses the next four quarters instead, so the two can differ."
        )
    return text


PEG_GAP_COLORS = {
    "Loss, no P/E": "rgba(205, 92, 92, 0.16)",
    "Year-ago loss": "rgba(255, 165, 0, 0.18)",
    "EPS below year ago": "rgba(128, 128, 128, 0.16)",
}


PEG_GAP_TEXT = {
    "Loss, no P/E": "EPS is a loss",
    "Year-ago loss": "the year-ago EPS was a loss, and growth from a loss has no %",
    "EPS below year ago": "EPS is below a year ago",
}


def _ttm_eps_growth(valuation):
    """TTM EPS, the TTM EPS of a year earlier and the growth (%) between them, by day."""

    eps = valuation["TTM_Reported_EPS"]
    clean = eps.dropna()
    if clean.empty:
        empty = pd.Series(np.nan, index=valuation.index)
        return eps, empty, empty
    year_ago = clean.reindex(valuation.index - pd.Timedelta(days=365), method="ffill")
    year_ago.index = valuation.index
    growth = (eps / year_ago.where(year_ago > 0) - 1) * 100
    return eps, year_ago, growth


def trailing_peg(valuation):
    """Daily trailing PEG: trailing P/E divided by TTM EPS growth (%) versus a year earlier.

    ``valuation`` holds ``Trailing_PE`` and ``TTM_Reported_EPS`` by day. PEG is left
    empty while EPS growth is zero or negative, where the ratio has no meaning.
    """

    if valuation is None or valuation.empty or not {"Trailing_PE", "TTM_Reported_EPS"}.issubset(valuation.columns):
        return pd.Series(dtype=float)
    _, _, growth = _ttm_eps_growth(valuation)
    return (valuation["Trailing_PE"] / growth.where(growth > 0)).replace([np.inf, -np.inf], np.nan)


def _growth_path(first, total, steps):
    """The quarterly growth rate at which ``first * rate**k`` for ``k`` in ``steps`` adds up to ``total``,
    or None when no rate between 0.2 and 5 fits."""

    if not (np.isfinite(first) and np.isfinite(total)) or first <= 0 or total <= 0:
        return None
    low, high = 0.2, 5.0
    sum_at = lambda rate: sum(first * rate ** k for k in steps)
    if not sum_at(low) <= total <= sum_at(high):
        return None
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if sum_at(middle) < total else (low, middle)
    return (low + high) / 2


def fiscal_quarters_reported(reported, this_year, this_quarter, next_quarter, fiscal_year_end=None,
                             last_quarter_end=None):
    """How many quarters of the current fiscal year (Yahoo's 0y) are already reported: 0 to 3.

    From the fiscal year end when given; otherwise from which count makes the 0y estimate add up
    (it equals the reported quarters plus 0q, plus +1q when the year ends with +1q). None when unknown.
    """

    if fiscal_year_end is not None and last_quarter_end is not None and pd.notna(fiscal_year_end):
        months = (pd.Timestamp(fiscal_year_end) - pd.Timestamp(last_quarter_end)).days / 30.44
        left = int(round(months / 3)) % 4 or 4  # quarters of the fiscal year after the last reported one
        return 4 - left if left < 4 else 0
    recent = list(reported.iloc[::-1])
    if not np.isfinite(this_year) or this_year == 0:
        return None
    errors = {}
    if len(recent) >= 3:
        errors[3] = abs(sum(recent[:3]) + this_quarter - this_year)
    if len(recent) >= 2:
        errors[2] = abs(sum(recent[:2]) + this_quarter + next_quarter - this_year)
    best = min(errors, key=errors.get) if errors else None
    return best if best is not None and errors[best] <= 0.03 * abs(this_year) else None


def later_quarter_estimates(this_quarter, next_quarter, this_year, next_year, reported, fiscal_year_end=None,
                            last_quarter_end=None):
    """Estimates for the two quarters after +1q, which Yahoo doesn't give, from its fiscal-year averages.

    Yahoo's 0y is the current fiscal year and +1y the next one, so which quarters each covers depends
    on how much of the year is reported. The quarters a year estimate leaves open follow a steady
    quarterly growth path from +1q. An approximation; empty when the numbers don't fit one.
    """

    done = fiscal_quarters_reported(reported, this_year, this_quarter, next_quarter, fiscal_year_end,
                                    last_quarter_end)
    if done == 3:  # 0q ends the year: +1q, +2q, +3q and one more make up +1y
        rate = _growth_path(next_quarter, next_year - next_quarter, [1, 2, 3])
    elif done == 2 or done is None:  # +1q ends the year: +2q to +5q make up +1y
        rate = _growth_path(next_quarter, next_year, [1, 2, 3, 4])
    elif done == 1:  # +2q ends the year: what 0y leaves is +2q; +3q starts +1y
        known = float(reported.iloc[-1]) + this_quarter + next_quarter
        rate = _growth_path(next_quarter, this_year - known, [1])
    else:  # the year has just begun: what 0y leaves is +2q and +3q
        rate = _growth_path(next_quarter, this_year - this_quarter - next_quarter, [1, 2])
    return [] if rate is None else [next_quarter * rate, next_quarter * rate ** 2]


def forward_eps(earnings, estimates=None, quarter_days=91, fiscal_year_end=None):
    """EPS of the next four quarters to be reported, as of each announcement date.

    ``earnings`` is indexed by announcement date with ``Reported EPS``. Quarters already
    reported use the EPS later reported (hindsight: not what analysts expected then); quarters
    not yet reported use Yahoo's analyst averages (``estimates`` rows 0q, +1q, 0y, +1y; see
    :func:`later_quarter_estimates`). ``fiscal_year_end`` (optional) is the end of the current
    fiscal year. Returns a frame with ``Forward_EPS`` and ``Estimated`` (True when any of the four
    quarters is an estimate), NaN where the four quarters are not all known.
    """

    if earnings is None or earnings.empty or "Reported EPS" not in earnings.columns:
        return pd.DataFrame(columns=["Forward_EPS", "Estimated"])
    reported = pd.to_numeric(earnings["Reported EPS"], errors="coerce").dropna().sort_index()
    if reported.empty:
        return pd.DataFrame(columns=["Forward_EPS", "Estimated"])
    ahead = []
    if estimates is not None and not getattr(estimates, "empty", True) and "avg" in estimates.columns:
        average = lambda period: (float(pd.to_numeric(estimates.at[period, "avg"], errors="coerce"))
                                  if period in estimates.index else np.nan)
        for period in ["0q", "+1q"]:
            if not np.isfinite(average(period)):
                break
            ahead.append(average(period))
        if len(ahead) == 2:
            # A quarter ends roughly five weeks before it is reported.
            ahead += later_quarter_estimates(ahead[0], ahead[1], average("0y"), average("+1y"), reported,
                                             fiscal_year_end, reported.index[-1] - pd.Timedelta(days=35))
    last = reported.index[-1]
    future = pd.Series(ahead, index=[last + pd.Timedelta(days=quarter_days * (k + 1)) for k in range(len(ahead))],
                       dtype=float)
    quarters = pd.concat([reported, future])
    estimated = pd.Series([False] * len(reported) + [True] * len(future), index=quarters.index)
    # On each announcement date the next four quarters are the four rows after it.
    next_four = quarters[::-1].rolling(4, min_periods=4).sum()[::-1].shift(-1)
    uses_estimate = estimated[::-1].astype(float).rolling(4, min_periods=4).max()[::-1].shift(-1) > 0
    frame = pd.DataFrame({"Forward_EPS": next_four, "Estimated": uses_estimate}).loc[reported.index]
    return frame


def forward_valuation(valuation, forward):
    """Daily forward P/E and forward PEG from :func:`forward_eps`.

    Forward P/E = close / next four quarters' EPS. Forward PEG = forward P/E / EPS growth (%)
    from the trailing four quarters to the next four. Each is NaN where its ratio has no meaning
    (forward EPS a loss, or growth zero or negative). ``Estimated`` marks days that use analyst estimates.
    """

    columns = ["Forward_EPS", "Forward_PE", "Forward_PEG", "Estimated"]
    if valuation is None or valuation.empty or forward is None or forward.empty:
        return pd.DataFrame(columns=columns)
    combined = valuation.index.union(forward.index).sort_values()
    daily = forward.reindex(combined).ffill().reindex(valuation.index)
    eps = pd.to_numeric(daily["Forward_EPS"], errors="coerce")
    pe = (valuation["Close"] / eps.where(eps > 0)).replace([np.inf, -np.inf], np.nan)
    ttm = valuation["TTM_Reported_EPS"]
    growth = (eps / ttm.where(ttm > 0) - 1) * 100
    peg = (pe / growth.where(growth > 0)).replace([np.inf, -np.inf], np.nan)
    estimated = daily["Estimated"].astype("boolean").fillna(False).astype(bool)
    return pd.DataFrame({"Forward_EPS": eps, "Forward_PE": pe, "Forward_PEG": peg, "Estimated": estimated},
                        index=valuation.index)


def break_long_gaps(series, days=7):
    """``series`` with a NaN inserted inside every gap longer than ``days``, so a line breaks there.

    Weekends and market holidays stay joined; a reindex to business days would break the line on every holiday.
    """

    series = series.dropna()
    if len(series) < 2:
        return series
    gaps = series.index[1:][np.diff(series.index.values) > np.timedelta64(days, "D")]
    breaks = pd.Series(np.nan, index=gaps - pd.Timedelta(days=1))
    return pd.concat([series, breaks]).sort_index()


def peg_gap_reasons(valuation):
    """Why trailing PEG is blank, by day: one of the ``PEG_GAP_COLORS`` keys, or NaN where PEG exists.

    Days without a year-ago TTM EPS (too little history) stay NaN: that is missing data, not a reading.
    """

    if valuation is None or valuation.empty or "TTM_Reported_EPS" not in valuation.columns:
        return pd.Series(dtype=object)
    eps, year_ago, growth = _ttm_eps_growth(valuation)
    reasons = pd.Series(np.nan, index=valuation.index, dtype=object)
    reasons[(growth <= 0) & eps.notna()] = "EPS below year ago"
    reasons[(year_ago <= 0) & eps.notna()] = "Year-ago loss"
    reasons[eps <= 0] = "Loss, no P/E"
    return reasons


def gap_spans(reasons):
    """Runs of the same reason as (start, end, reason); ``end`` is the first day after the run."""

    spans = []
    if reasons is None or reasons.empty:
        return spans
    run = (reasons.fillna("") != reasons.fillna("").shift()).cumsum()
    for _, block in reasons.groupby(run):
        reason = block.iloc[0]
        if isinstance(reason, str):
            after = reasons.index[reasons.index > block.index[-1]]
            spans.append((block.index[0], after[0] if len(after) else block.index[-1], reason))
    return spans


def peg_summary_text(peg, forward_pe=None, forward_growth=None, gaps=None):
    """One caption line: today's trailing PEG against its own history, plus a forward PEG when estimates exist.

    ``gaps`` (from :func:`peg_gap_reasons`) explains a blank PEG today instead of quoting a stale one.
    """

    series = peg.dropna() if peg is not None else pd.Series(dtype=float)
    parts = []
    today_gap = gaps.dropna() if gaps is not None else pd.Series(dtype=object)
    if not today_gap.empty and today_gap.index[-1] == gaps.index[-1]:
        since = gap_spans(gaps)[-1][0]
        parts.append(f"No trailing PEG today: {PEG_GAP_TEXT[today_gap.iloc[-1]]} (since {since:%b %Y}).")
        if not series.empty:
            parts.append(f"Last PEG was {series.iloc[-1]:.2f} on {series.index[-1]:%d %b %Y}.")
    elif not series.empty:
        current = float(series.iloc[-1])
        parts.append(
            f"Trailing PEG {current:.2f} (P/E / last year's EPS growth %), {ordinal(round((series < current).mean() * 100))} "
            f"percentile since {series.index.min():%Y} (median {series.median():.2f}, low {series.min():.2f} in "
            f"{series.idxmin():%b %Y})."
        )
    else:
        parts.append("No trailing PEG (EPS growth was zero or negative).")
    if all(value is not None and np.isfinite(value) and value > 0 for value in [forward_pe, forward_growth]):
        parts.append(
            f"Forward PEG {forward_pe / (forward_growth * 100):.2f} (forward P/E {forward_pe:.1f}x / next fiscal year's "
            f"expected EPS growth {forward_growth * 100:.0f}%)."
        )
    parts.append("Below 1 reads as cheap for the growth, but only if that growth lasts.")
    return " ".join(parts)


def quarterly_eps(earnings, quarter_ends, max_lag_days=95, fallback_lag_days=30):
    """Reported EPS per quarter, moved from its announcement date to the fiscal quarter end it reports.

    ``earnings`` is indexed by announcement date with a ``Reported EPS`` column.
    Each announcement is matched to the latest of ``quarter_ends`` (the revenue
    quarters) within ``max_lag_days`` before it, so EPS and revenue bars line up.
    An announcement with no matching revenue quarter (the revenue source skipped
    it) is placed ``fallback_lag_days`` before its date. ``max_lag_days`` stays
    near one quarter so a skipped quarter's EPS doesn't land on the one before it.
    """

    if earnings is None or earnings.empty or "Reported EPS" not in earnings.columns:
        return pd.Series(dtype=float)
    ends = pd.DatetimeIndex(sorted(quarter_ends))
    values = {}
    for announced, eps in pd.to_numeric(earnings["Reported EPS"], errors="coerce").dropna().items():
        candidates = ends[(ends < announced) & (ends >= announced - pd.Timedelta(days=max_lag_days))]
        quarter = candidates[-1] if len(candidates) else announced.normalize() - pd.Timedelta(days=fallback_lag_days)
        values[quarter] = float(eps)
    return pd.Series(values, dtype=float).sort_index()


def eps_estimate_quarters(estimates, last_quarter_end):
    """Analyst average EPS for the current and next fiscal quarter, placed like :func:`revenue_estimate_quarters`."""

    if estimates is None or getattr(estimates, "empty", True) or "avg" not in estimates.columns:
        return pd.Series(dtype=float)
    if last_quarter_end is None or pd.isna(last_quarter_end):
        return pd.Series(dtype=float)
    values = {}
    for step, period in [(1, "0q"), (2, "+1q")]:
        if period in estimates.index:
            value = pd.to_numeric(estimates.at[period, "avg"], errors="coerce")
            if pd.notna(value):
                values[pd.Timestamp(last_quarter_end) + pd.DateOffset(months=3 * step)] = float(value)
    return pd.Series(values, dtype=float)


def yoy_change(series, min_days=330, max_days=400):
    """Percent change of each quarterly value versus the value about a year earlier.

    Matched by date, not position, because some sources skip quarters. Empty
    (NaN) when there is no year-ago value or it is zero or negative, where a
    percent change has no meaning (EPS turning from a loss to a profit).
    """

    clean = series.dropna().sort_index() if series is not None else pd.Series(dtype=float)
    values = {}
    for when, value in clean.items():
        base = clean.loc[(clean.index >= when - pd.Timedelta(days=max_days))
                         & (clean.index <= when - pd.Timedelta(days=min_days))]
        values[when] = (value / base.iloc[-1] - 1) * 100 if not base.empty and base.iloc[-1] > 0 else np.nan
    return pd.Series(list(values.values()), index=pd.DatetimeIndex(list(values)), dtype=float)


def price_yoy(close):
    """Daily percent change of the close versus the close 365 days earlier (or the last trading day before)."""

    clean = close.dropna().sort_index() if close is not None else pd.Series(dtype=float)
    if clean.empty:
        return pd.Series(dtype=float)
    year_ago = clean.reindex(clean.index - pd.Timedelta(days=365), method="ffill")
    year_ago.index = clean.index
    return (clean / year_ago.where(year_ago > 0) - 1) * 100


def growth_price_correlation(growth, price_change, lags=None):
    """Correlation between quarterly YoY growth and the price's YoY change, at several lags.

    ``price_change`` (daily) is sampled at each quarter end in ``growth``. A lag of
    -4 compares each quarter's growth with the price change a year later (does
    growth predict the price?), +1 with the price change a quarter earlier (does
    the price run ahead?). Rank (Spearman) correlation, so a few 400% growth
    quarters don't dominate. Returns ``{label: (r, n)}``; r is NaN with fewer
    than 8 quarters.
    """

    lags = LAGS if lags is None else lags
    clean = growth.dropna() if growth is not None else pd.Series(dtype=float)
    prices = price_change.dropna() if price_change is not None else pd.Series(dtype=float)
    result = {}
    if clean.empty or prices.empty:
        return {label: (np.nan, 0) for label in lags}
    sampled = prices.reindex(clean.index, method="ffill", tolerance=pd.Timedelta(days=7))
    for label, lag in lags.items():
        pair = pd.DataFrame({"growth": clean.values, "price": sampled.shift(lag).values}).dropna()
        ranked = pair.rank()
        result[label] = (float(ranked["growth"].corr(ranked["price"])) if len(pair) >= 8 else np.nan, len(pair))
    return result


def correlation_text(correlations):
    """One caption line for :func:`growth_price_correlation` results by series name."""

    parts = []
    for name, by_lag in correlations.items():
        cells = [f"{label} r = {r:+.2f}" for label, (r, _) in by_lag.items() if np.isfinite(r)]
        if cells:
            n = max(count for _, count in by_lag.values())
            parts.append(f"{name} YoY vs price YoY ({n} quarters): " + ", ".join(cells) + ".")
    if not parts:
        return "Not enough quarters to compare growth with the price."
    return (" ".join(parts) + " r is a rank correlation from -1 to +1. YoY windows overlap, so neighbouring "
            "quarters are not independent and r looks stronger than it is.")


def add_close_overlay(fig, close, row, log_price=False, name="Close (right axis)", color=CLOSE_OVERLAY_COLOR):
    """Draw the daily close in subplot ``row`` (one-column figure) on its own right-hand y-axis."""

    close = close.dropna() if close is not None else pd.Series(dtype=float)
    if close.empty:
        return
    subplot = fig.get_subplot(row, 1)
    x_axis, y_axis = subplot.xaxis.plotly_name, subplot.yaxis.plotly_name  # e.g. "xaxis3", "yaxis3"
    taken = [int(key[5:] or 1) for key in fig.layout.to_plotly_json() if key.startswith("yaxis")]
    number = max(taken) + 1
    fig.layout[f"yaxis{number}"] = dict(
        overlaying=y_axis.replace("axis", ""), anchor=x_axis.replace("axis", ""), side="right", showgrid=False,
        type="log" if log_price else "linear", rangemode="tozero", tickformat=",.0f", automargin=True,
        tickfont=dict(color=PRICE_COLOR, size=10), zeroline=False)
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name=name, xaxis=x_axis.replace("axis", ""),
                             yaxis=f"y{number}", line=dict(color=color, width=1.2),
                             hovertemplate="%{x|%Y-%m-%d} close: %{y:,.2f}<extra></extra>"))


def add_volume_overlay(fig, ohlcv, row):
    """Daily volume as bars on a right-hand axis of subplot ``row``, kept to the bottom quarter of the panel.

    The axis is scaled to the 99th percentile, so a few extreme days don't flatten every other bar.
    """

    volume = pd.to_numeric(ohlcv["Volume"], errors="coerce").fillna(0)
    if not (volume > 0).any():
        return
    subplot = fig.get_subplot(row, 1)
    x_axis, y_axis = subplot.xaxis.plotly_name.replace("axis", ""), subplot.yaxis.plotly_name.replace("axis", "")
    taken = [int(key[5:] or 1) for key in fig.layout.to_plotly_json() if key.startswith("yaxis")]
    number = max(taken) + 1
    fig.layout[f"yaxis{number}"] = dict(overlaying=y_axis, anchor=x_axis, side="right", showgrid=False,
                                        range=[0, float(volume.quantile(0.99)) * 4], tickformat=".2s", automargin=True,
                                        tickfont=dict(color=PRICE_COLOR, size=10), zeroline=False)
    colors = np.where(ohlcv["Close"] >= ohlcv["Open"], VOLUME_UP_COLOR, VOLUME_DOWN_COLOR)
    fig.add_trace(go.Bar(x=volume.index, y=volume, name="Volume (right axis)", xaxis=x_axis, yaxis=f"y{number}",
                         marker=dict(color=colors, line_width=0),
                         hovertemplate="%{x|%Y-%m-%d} volume: %{y:,.0f}<extra></extra>"))


def panel_legend(row):
    """Legend id for subplot ``row``: Plotly names them legend, legend2, legend3 ..."""

    return "legend" if row == 1 else f"legend{row}"


def legends_under_panels(fig, rows, height, gap_px=6, dates_every_row=False, dated_rows=()):
    """Give each subplot row its own horizontal legend, placed right under that row.

    Traces are assigned by their y-axis; overlay axes count for the row they overlay. With
    ``dates_every_row`` every row shows its date tick labels, and each legend goes below them;
    ``dated_rows`` names single rows that do.
    """

    layout = fig.layout
    row_of_axis = {}
    for row in range(1, rows + 1):
        name = fig.get_subplot(row, 1).yaxis.plotly_name.replace("axis", "")
        row_of_axis[name] = row
    for key in layout.to_plotly_json():
        if key.startswith("yaxis") and layout[key].overlaying:
            row_of_axis[key.replace("axis", "")] = row_of_axis.get(layout[key].overlaying)
    used = set()
    for trace in fig.data:
        row = row_of_axis.get(trace.yaxis or "y")
        if row is None:
            continue
        trace.legend = panel_legend(row)
        used.add(row)
    for row in used:
        bottom = fig.get_subplot(row, 1).yaxis.domain[0]
        # The bottom row carries the date tick labels, so its legend goes below them.
        offset = (gap_px + (28 if row == rows or dates_every_row or row in dated_rows else 0)) / height
        layout[panel_legend(row)] = dict(orientation="h", yanchor="top", y=bottom - offset, xanchor="left", x=0,
                                         bgcolor="rgba(0,0,0,0)")


def add_forward_line(fig, values, estimated, name, row, hover):
    """A forward measure as a dashed line, turning dotted where it rests on analyst estimates."""

    values = values.dropna()
    if values.empty:
        return
    estimated = estimated.reindex(values.index).fillna(False).astype(bool)
    known, guessed = values[~estimated], values[estimated]
    if not guessed.empty and not known.empty:
        guessed = pd.concat([known.iloc[-1:], guessed])  # join the two styles without a gap
    # One legend entry for both styles keeps the legend to two lines on a phone; the caption explains the dots.
    for part, label, dash in [(known, name, "dash"), (guessed, f"{name}, estimates", "dot")]:
        if part.empty:
            continue
        line = break_long_gaps(part)
        fig.add_trace(go.Scatter(x=line.index, y=line, mode="lines", name=label, connectgaps=False,
                                 showlegend=part is known or known.empty,
                                 line=dict(color=FORWARD_COLOR, width=2, dash=dash),
                                 hovertemplate=f"%{{x|%Y-%m-%d}}: {hover}<extra>{label}</extra>"), row=row, col=1)


def revenue_pe_figure(revenue, trailing_pe, revenue_estimates=None, peg=None, eps=None, eps_estimates=None,
                      price_change=None, growth=None, close=None, log_price=False, height=None, ohlcv=None,
                      peg_gaps=None, forward=None, link_dates=False):
    """Daily price, growth vs price, revenue, EPS, trailing P/E and PEG panels, top to bottom, on one date axis.

    ``ohlcv`` (optional, daily Open/High/Low/Close/Volume) turns the price panel into candlesticks with
    volume on a right-hand axis; with only ``close`` it is a line.

    ``peg_gaps`` (optional, from :func:`peg_gap_reasons`) shades the days PEG is blank, by reason.
    ``forward`` (optional, from :func:`forward_valuation`) adds forward P/E and forward PEG lines.
    With ``link_dates`` a zoom on one panel zooms them all; otherwise each panel zooms on its own and shows its own dates.

    ``revenue`` is indexed by fiscal quarter end, in reporting currency units.
    ``revenue_estimates`` (optional) holds analyst averages for coming quarters
    and is drawn as faded bars, labelled as estimates, so it never passes for
    reported revenue.
    """

    peg = peg.dropna() if peg is not None else pd.Series(dtype=float)
    forward = forward if forward is not None else pd.DataFrame(columns=["Forward_PE", "Forward_PEG", "Estimated"])
    forward_pe, forward_peg = forward["Forward_PE"].dropna(), forward["Forward_PEG"].dropna()
    eps = eps.dropna() if eps is not None else pd.Series(dtype=float)
    price_change = price_change.dropna() if price_change is not None else pd.Series(dtype=float)
    # Pass ``growth`` when revenue or EPS is trimmed to the window shown, so the first year still has growth.
    growth = growth or {"Revenue": yoy_change(revenue), "EPS": yoy_change(eps)}
    growth = {name: series.loc[series.index >= price_change.index.min()] if not price_change.empty else series
              for name, series in growth.items()}

    # Panels top to bottom: price, growth vs price, revenue, EPS, trailing P/E, PEG. Optional ones drop out.
    # Titles stay short enough for a phone; the page caption defines each one.
    candles = ohlcv.dropna(subset=["Open", "High", "Low", "Close"]) if ohlcv is not None else pd.DataFrame()
    close = close.dropna() if close is not None else pd.Series(dtype=float)
    if close.empty and not candles.empty:
        close = candles["Close"]
    panels = {}
    if not candles.empty:
        panels["price"] = ("Daily OHLC (split-adjusted) and volume", 0.34)
    elif not close.empty:
        panels["price"] = ("Daily close (split-adjusted)", 0.26)
    if not price_change.empty:
        values = pd.concat([price_change] + list(growth.values())).dropna()
        panels["growth"] = ("YoY change, %" + (f" (capped at {GROWTH_AXIS_CAP}%)" if values.max() > GROWTH_AXIS_CAP else ""),
                            0.3)
    panels["revenue"] = ("Revenue by quarter, billions", 0.3)
    if not eps.empty:
        panels["eps"] = ("EPS by quarter (adjusted)", 0.22)
    panels["pe"] = ("P/E, trailing and forward" if not forward_pe.empty else "Trailing P/E", 0.3)
    peg_values = pd.concat([peg, forward_peg])
    if not peg_values.empty:
        panels["peg"] = (("PEG, trailing and forward" if not forward_peg.empty else "Trailing PEG")
                         + (f" (capped at {PEG_AXIS_CAP})" if peg_values.max() > PEG_AXIS_CAP else ""), 0.22)
    row = {key: number for number, key in enumerate(panels, start=1)}
    rows = len(panels)
    heights = [weight for _, weight in panels.values()]
    height = height or {2: 560, 3: 740, 4: 920, 5: 1120}.get(rows, 1320)
    # Room between panels for the legend under each panel and the next panel's title (and its dates when unlinked).
    gap = 80 if link_dates else 108
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=min(0.15, gap / height),
                        row_heights=[h / sum(heights) for h in heights],
                        subplot_titles=[title for title, _ in panels.values()])
    if not candles.empty:
        fig.add_trace(go.Candlestick(x=candles.index, open=candles["Open"], high=candles["High"], low=candles["Low"],
                                     close=candles["Close"], name="OHLC", increasing_line_color=UP_COLOR,
                                     decreasing_line_color=DOWN_COLOR, line=dict(width=1)), row=row["price"], col=1)
        if "Volume" in candles:
            add_volume_overlay(fig, candles, row["price"])
    elif not close.empty:
        fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close", line=dict(color=PRICE_COLOR, width=1.5),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>"), row=row["price"], col=1)
    if "price" in row:
        fig.update_yaxes(tickformat=",.0f", type="log" if log_price else "linear", row=row["price"], col=1)

    pe = trailing_pe.dropna() if trailing_pe is not None else pd.Series(dtype=float)
    if not pe.empty:
        # Days with an EPS loss have no P/E; a straight line across them would invent one.
        pe_line = break_long_gaps(pe)
        fig.add_trace(go.Scatter(x=pe_line.index, y=pe_line, mode="lines", name="Trailing P/E", connectgaps=False,
                                 line=dict(color=PE_COLOR, width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}x<extra></extra>"), row=row["pe"], col=1)
        median = float(pe.median())
        fig.add_hline(y=median, line=dict(color="gray", width=1, dash="dash"), row=row["pe"], col=1,
                      annotation_text=f"median {median:.1f}x", annotation_position="top left")
    add_forward_line(fig, forward_pe, forward["Estimated"], "Forward P/E", row["pe"], "%{y:.1f}x")
    if pe.empty:
        axis = "" if row["pe"] == 1 else row["pe"]
        fig.add_annotation(x=0.5, y=0.5, xref=f"x{axis} domain", yref=f"y{axis} domain", showarrow=False,
                           text="No trailing P/E (no reported EPS, or EPS was negative)", font=dict(color="gray"))

    if not peg.empty:
        # Gaps (growth <= 0) stay gaps: connecting them would invent PEG values.
        full = break_long_gaps(peg)
        fig.add_trace(go.Scatter(x=full.index, y=full, mode="lines", name="Trailing PEG", connectgaps=False,
                                 line=dict(color=PEG_COLOR, width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=row["peg"], col=1)
        fig.add_hline(y=1, line=dict(color="gray", width=1, dash="dash"), row=row["peg"], col=1,
                      annotation_text="PEG 1", annotation_position="top left")
    if "peg" in row:
        add_forward_line(fig, forward_peg, forward["Estimated"], "Forward PEG", row["peg"], "%{y:.2f}")
        # Right after earnings turn up, growth is tiny and PEG spikes; capping keeps the usual range readable.
        fig.update_yaxes(range=[0, min(PEG_AXIS_CAP, float(peg_values.max()) * 1.1)], row=row["peg"], col=1)

    # Shade why PEG (and, for losses, P/E) is blank, with one legend entry per reason.
    spans = gap_spans(peg_gaps.loc[peg_gaps.index >= peg.index.min()] if peg_gaps is not None and not peg.empty
                      else peg_gaps)
    for key, reasons in [("pe", ["Loss, no P/E"]), ("peg", list(PEG_GAP_COLORS))]:
        if key not in row:
            continue
        shown = [reason for reason in reasons if any(span[2] == reason for span in spans)]
        for start, end, reason in spans:
            if reason in shown:
                fig.add_vrect(x0=start, x1=end, fillcolor=PEG_GAP_COLORS[reason], line_width=0, layer="below",
                              row=row[key], col=1)
        for reason in shown:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=reason, hoverinfo="skip",
                                     marker=dict(symbol="square", size=12, color=PEG_GAP_COLORS[reason])),
                          row=row[key], col=1)

    actual = revenue.dropna() / 1e9 if revenue is not None else pd.Series(dtype=float)
    fig.add_trace(go.Bar(x=actual.index, y=actual, name="Revenue (reported)", marker_color=ACTUAL_COLOR,
                         hovertemplate="Quarter to %{x|%Y-%m-%d}: %{y:,.2f}B<extra></extra>"), row=row["revenue"], col=1)
    estimated = revenue_estimates.dropna() / 1e9 if revenue_estimates is not None else pd.Series(dtype=float)
    if not estimated.empty:
        fig.add_trace(go.Bar(x=estimated.index, y=estimated, name="Revenue (analyst estimate)",
                             marker=dict(color=ESTIMATE_COLOR, line=dict(color=ACTUAL_COLOR, width=1),
                                         pattern=dict(shape="/", fgcolor=ACTUAL_COLOR, size=6)),
                             hovertemplate="Quarter to %{x|%Y-%m-%d} (estimate): %{y:,.2f}B<extra></extra>"),
                      row=row["revenue"], col=1)

    if not eps.empty:
        fig.add_trace(go.Bar(x=eps.index, y=eps, name="EPS (reported)", marker_color=EPS_COLOR,
                             hovertemplate="Quarter to %{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=row["eps"], col=1)
        eps_ahead = eps_estimates.dropna() if eps_estimates is not None else pd.Series(dtype=float)
        if not eps_ahead.empty:
            fig.add_trace(go.Bar(x=eps_ahead.index, y=eps_ahead, name="EPS (analyst estimate)",
                                 marker=dict(color=EPS_ESTIMATE_COLOR, line=dict(color=EPS_COLOR, width=1),
                                             pattern=dict(shape="/", fgcolor=EPS_COLOR, size=6)),
                                 hovertemplate="Quarter to %{x|%Y-%m-%d} (estimate): %{y:.2f}<extra></extra>"),
                          row=row["eps"], col=1)
        fig.update_yaxes(tickformat=".2f", row=row["eps"], col=1)

    if "growth" in row:
        fig.add_trace(go.Scatter(x=price_change.index, y=price_change, mode="lines", name="Price YoY %",
                                 line=dict(color=PRICE_COLOR, width=1.2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:+.0f}%<extra></extra>"), row=row["growth"], col=1)
        for name, color in [("Revenue", ACTUAL_COLOR), ("EPS", EPS_COLOR)]:
            series = growth[name].dropna()
            if not series.empty:
                fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines+markers", name=f"{name} YoY %",
                                         line=dict(color=color, width=2), marker=dict(size=8),
                                         hovertemplate=f"{name}, quarter to %{{x|%Y-%m-%d}}: %{{y:+.0f}}%<extra></extra>"),
                              row=row["growth"], col=1)
        fig.add_hline(y=0, line=dict(color="gray", width=1), row=row["growth"], col=1)
        low = max(-100.0, float(values.min()) * 1.1)
        fig.update_yaxes(ticksuffix="%", range=[min(low, -10.0), min(GROWTH_AXIS_CAP, float(values.max()) * 1.1)],
                         row=row["growth"], col=1)

    has_eps_estimates = not eps.empty and eps_estimates is not None and not eps_estimates.dropna().empty
    if (not estimated.empty or has_eps_estimates) and not actual.empty:
        # Added last: add_vline only reaches the panels that already hold traces.
        fig.add_vline(x=actual.index[-1] + pd.Timedelta(days=45), line=dict(color="gray", width=1, dash="dot"))

    # Legend entries in the order the traces were added, within each panel's own legend.
    for position, trace in enumerate(fig.data):
        trace.legendrank = int(trace.yaxis[1:] or 1) * 100 + position

    # The close on a right-hand axis in every panel but the close panel itself.
    for key, number in row.items():
        if key != "price":
            add_close_overlay(fig, close, number, log_price)

    fig.update_yaxes(ticksuffix="x", rangemode="tozero", row=row["pe"], col=1)
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero", row=row["revenue"], col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=60), bargap=0.15)
    if not link_dates:
        # Panels start on one date range, but a zoom or pan on one leaves the others where they are.
        dates = pd.to_datetime(pd.Series([x for trace in fig.data if trace.x is not None for x in trace.x]),
                               errors="coerce").dropna()
        span = dict(range=[dates.min() - pd.Timedelta(days=30), dates.max() + pd.Timedelta(days=30)]) if len(dates) else {}
        fig.update_xaxes(matches=None, showticklabels=True, **span)
    legends_under_panels(fig, rows, height, dates_every_row=not link_dates)
    return fig
