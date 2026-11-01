import numpy as np
import pytest
import pandas as pd

from analysis.earnings_dates import (
    describe_release,
    merge_releases,
    next_release_text,
    past_and_next,
    reaction_day,
    reaction_summary,
    release_table,
    releases_with_reaction,
    sec_releases,
    timing,
    yahoo_releases,
)
from analysis.valuation_chart import revenue_pe_figure


def yahoo_frame():
    index = pd.DatetimeIndex(["2026-11-17 16:00", "2026-08-26 16:00", "2026-05-20 08:00", "2026-02-25 00:00"]
                             ).tz_localize("America/New_York")
    return pd.DataFrame({"EPS Estimate": [2.47, 2.09, 1.77, 1.54], "Reported EPS": [np.nan, 2.22, 1.87, 1.62],
                         "Surprise(%)": [np.nan, 6.16, 5.54, 5.32]}, index=pd.Index(index, name="Earnings Date"))


def test_yahoo_releases_are_eastern_and_sorted():
    releases = yahoo_releases(yahoo_frame())
    assert list(releases.index) == [pd.Timestamp("2026-02-25"), pd.Timestamp("2026-05-20 08:00"),
                                    pd.Timestamp("2026-08-26 16:00"), pd.Timestamp("2026-11-17 16:00")]
    assert (releases["Source"] == "Yahoo").all()
    assert yahoo_releases(None).empty


def test_timing_and_reaction_day():
    sessions = pd.bdate_range("2026-08-24", "2026-09-04")
    assert timing(pd.Timestamp("2026-08-26 16:00")) == "after"
    assert timing(pd.Timestamp("2026-08-26 07:00")) == "before"
    assert timing(pd.Timestamp("2026-08-26 11:00")) == "during"
    assert timing(pd.Timestamp("2026-08-26")) == "unknown"
    assert reaction_day(pd.Timestamp("2026-08-26 16:00"), sessions) == pd.Timestamp("2026-08-27")
    assert reaction_day(pd.Timestamp("2026-08-26 07:00"), sessions) == pd.Timestamp("2026-08-26")
    # A Friday release after the close reacts on Monday; one after the data ends hasn't reacted yet.
    assert reaction_day(pd.Timestamp("2026-08-28 16:05"), sessions) == pd.Timestamp("2026-08-31")
    assert reaction_day(pd.Timestamp("2026-09-04 16:05"), sessions) is None


def test_sec_releases_keep_item_202_8ks():
    submissions = {"filings": {"recent": {
        "form": ["8-K", "8-K", "10-Q", "8-K/A", "8-K"],
        "items": ["2.02,9.01", "5.02", "", "2.02", "2.02,9.01"],
        "acceptanceDateTime": ["2026-08-26T20:21:19.000Z", "2026-08-01T12:00:00.000Z", "2026-08-27T10:00:00.000Z",
                               "2026-08-26T22:00:00.000Z", "2026-05-20T11:00:00.000Z"],
    }}}
    releases = sec_releases(submissions)
    # 20:21 UTC in August is 16:21 in New York; the 8-K/A the same day is the same release.
    assert list(releases.index) == [pd.Timestamp("2026-05-20 07:00"), pd.Timestamp("2026-08-26 16:21:19")]
    assert (releases["Source"] == "SEC 8-K").all()
    assert sec_releases({}).empty


def test_merge_adds_missing_sec_releases_and_times():
    yahoo = yahoo_releases(yahoo_frame())
    sec = pd.DataFrame({"EPS Estimate": np.nan, "Reported EPS": np.nan, "Surprise(%)": np.nan, "Source": "SEC 8-K"},
                       index=pd.DatetimeIndex(["2025-11-19 16:20", "2026-02-25 16:20", "2026-08-26 16:21"]))
    merged = merge_releases(yahoo, sec)
    assert len(merged) == 5
    # Yahoo's midnight release takes the 8-K's time; Yahoo's own time is kept; the missing quarter is added.
    assert pd.Timestamp("2026-02-25 16:20") in merged.index
    assert pd.Timestamp("2026-08-26 16:00") in merged.index
    assert merged.loc[pd.Timestamp("2025-11-19 16:20"), "Source"] == "SEC 8-K"
    assert merged.loc[pd.Timestamp("2026-02-25 16:20"), "Reported EPS"] == 1.62
    assert merge_releases(yahoo, None).equals(yahoo)


def test_past_and_next():
    releases = yahoo_releases(yahoo_frame())
    past, upcoming = past_and_next(releases, pd.Timestamp("2026-11-01 08:00"),
                                   {"earningsTimestampStart": 1794949200, "isEarningsDateEstimate": False})
    assert len(past) == 3
    assert upcoming == (pd.Timestamp("2026-11-17 16:00"), False)
    text = next_release_text(upcoming, pd.Timestamp("2026-11-01"))
    assert text == "Next earnings: Tue 17 Nov 2026, after close (confirmed), in 16 days."
    # No calendar row: the quote's date; none at all: a note.
    _, upcoming = past_and_next(releases.iloc[:3], pd.Timestamp("2026-11-01"), {"earningsTimestamp": 1794949200})
    assert upcoming == (pd.Timestamp("2026-11-17 16:00"), None)
    assert past_and_next(releases.iloc[:3], pd.Timestamp("2026-11-01"), {})[1] is None
    assert next_release_text(None, "2026-11-01") == "Next earnings: no date published yet."


def test_reactions_table_and_text():
    close = pd.Series(100.0, index=pd.bdate_range("2026-02-02", "2026-09-30"))
    close.loc["2026-08-27":] = 110.0  # after the 26 Aug release
    close.loc["2026-09-02":] = 99.0
    releases = yahoo_releases(yahoo_frame()).iloc[:3]
    quarter_ends = pd.DatetimeIndex(["2026-01-25", "2026-04-26", "2026-07-26"])
    reactions = releases_with_reaction(releases, close, quarter_ends)
    august = reactions.iloc[-1]
    assert august["Reaction day"] == pd.Timestamp("2026-08-27")
    assert august["1-day %"] == pytest.approx(10.0) and august["5-day %"] == pytest.approx(-1.0)
    assert august["Quarter end"] == pd.Timestamp("2026-07-26")
    text = describe_release(august)
    assert "Wed 26 Aug 2026, after close" in text and "EPS 2.22 vs est. 2.09 (+6.2%)" in text
    assert "next day +10.0%, 5 days -1.0%" in text
    table = release_table(reactions, 2)
    assert list(table["Date"]) == ["2026-08-26", "2026-05-20"]
    assert "beat the estimate 3 of 3" in reaction_summary(reactions)


def test_chart_marks_releases_on_every_panel():
    dates = pd.bdate_range("2025-01-01", "2026-09-30")
    close = pd.Series(np.linspace(100, 150, len(dates)), index=dates)
    revenue = pd.Series([1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9, 1.5e9],
                        index=pd.DatetimeIndex(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31",
                                                "2026-06-30"]))
    pe = pd.Series(30.0, index=dates)
    releases = pd.DataFrame({"Release": pd.DatetimeIndex(["2025-08-20 16:00", "2026-05-20 16:00"]),
                             "Text": ["first", "second"]})
    fig = revenue_pe_figure(revenue, pe, close=close, releases=releases,
                            next_release=(pd.Timestamp("2026-11-17 16:00"), "Next earnings"))
    marks = [trace for trace in fig.data if trace.meta and trace.meta.get("earnings")]
    rows = 3  # price, revenue, P/E
    assert len(marks) == rows * 2
    assert sum(trace.showlegend is not False for trace in marks) == 2
    assert len([shape for shape in fig.layout.shapes if shape.line.dash in ("dot", "dash") and shape.x0 == shape.x1
                and shape.yref.endswith("domain")]) >= rows * 3


def test_no_releases_at_all():
    # Yahoo rate limited and no SEC 8-K: nothing to show, nothing crashes.
    releases = merge_releases(yahoo_releases(None), sec_releases(None))
    past, upcoming = past_and_next(releases, pd.Timestamp("2026-11-01"), {"earningsTimestampStart": 1794949200})
    assert past.empty and upcoming[0] == pd.Timestamp("2026-11-17 16:00")
    reactions = releases_with_reaction(past, pd.Series(1.0, index=pd.bdate_range("2026-01-01", periods=5)))
    assert reactions.empty and release_table(reactions).empty and reaction_summary(reactions) == ""


def test_valuation_changes_and_signal_chart():
    from analysis.earnings_dates import with_valuation_changes
    from analysis.valuation_chart import earnings_signal_figure

    days = pd.bdate_range("2026-01-01", "2026-09-30")
    close = pd.Series(100.0, index=days)
    pe = pd.Series(20.0, index=days)
    pe.loc["2026-05-20":] = 25.0
    pe.loc["2026-08-27":] = 20.0
    reactions = releases_with_reaction(yahoo_releases(yahoo_frame()).iloc[:3], close)
    out = with_valuation_changes(reactions, pe, None)
    # The first release has nothing before it; then +25% and -20%.
    assert np.isnan(out["P/E change %"].iloc[0])
    assert out["P/E change %"].iloc[1:].round(6).tolist() == [25.0, -20.0]
    assert out["Forward P/E change %"].isna().all()
    assert "P/E chg %" in release_table(out).columns

    from analysis.earnings_dates import change_since_release, surprise_held

    since = change_since_release(pe, out)
    # Measured from the close before each reaction day: 0 into a release, then the jump on it.
    assert np.isnan(since.loc["2026-02-24"]) and since.loc["2026-02-25"] == 0
    assert since.loc["2026-05-19"] == 0 and since.loc["2026-05-20"] == pytest.approx(25.0)
    assert since.loc["2026-08-27"] == pytest.approx(-20.0)
    held = surprise_held(out, days)
    assert np.isnan(held.loc["2026-02-24"]) and held.loc["2026-05-19"] == 5.32 and held.loc["2026-08-27"] == 6.16

    fig = earnings_signal_figure(close, out, surprise=held, pe_change=since * 2)
    names = [trace.name for trace in fig.data]
    assert names[:2] == ["Close", "Release (green beat, red miss)"]
    assert "EPS surprise % (last release)" in names and "P/E % since release" in names
    assert "Forward P/E % since release" not in names
    bold = [shape for shape in fig.layout.shapes if shape.line.width == 2]
    assert len(bold) == 3 * 2  # three releases, two panels


def test_two_releases_on_one_reaction_day():
    from analysis.earnings_dates import change_since_release, surprise_held

    days = pd.bdate_range("2026-01-01", "2026-03-31")
    releases = pd.DataFrame({"Release": pd.DatetimeIndex(["2026-02-06 16:00", "2026-02-07 10:00"]),
                             "Reaction day": pd.DatetimeIndex(["2026-02-09", "2026-02-09"]),
                             "Surprise(%)": [1.0, 2.0]})
    since = change_since_release(pd.Series(10.0, index=days), releases)
    assert since.loc["2026-03-02"] == 0
    assert surprise_held(releases, days).loc["2026-03-02"] == 2.0
