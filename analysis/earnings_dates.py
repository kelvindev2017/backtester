"""Quarterly earnings release dates: past releases with EPS surprise and price reaction, and the next one.

Yahoo's earnings dates are the main source (date and time of the release, EPS estimate, reported EPS,
surprise %). SEC 8-K filings under item 2.02 ("Results of Operations and Financial Condition") fill in
releases Yahoo is missing, and Yahoo's quote fields give the next date when its calendar has none.

All times are US/Eastern, without a time zone, so they compare directly with daily price bars.
"""

import numpy as np
import pandas as pd

EASTERN = "America/New_York"
COLUMNS = ["EPS Estimate", "Reported EPS", "Surprise(%)", "Source"]
OPEN, CLOSE = pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=16)
def empty_releases():
    """No releases, with the date index and columns the functions below expect."""

    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([]))


TIMING_LABELS = {"before": "before open", "during": "during market", "after": "after close", "unknown": "time n/a"}


def to_eastern(index):
    """A DatetimeIndex in US/Eastern wall time, without a time zone (naive input is taken as UTC)."""

    index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(EASTERN).tz_localize(None)


def yahoo_releases(earnings):
    """Yahoo's ``get_earnings_dates`` frame as releases indexed by US/Eastern date and time, oldest first."""

    if earnings is None or len(earnings) == 0:
        return empty_releases()
    releases = earnings.copy()
    releases.index = to_eastern(releases.index)
    releases = releases[~releases.index.isna()]
    for column in COLUMNS[:3]:
        releases[column] = pd.to_numeric(releases[column], errors="coerce") if column in releases else np.nan
    releases["Source"] = "Yahoo"
    releases = releases[COLUMNS].sort_index()
    return releases[~releases.index.duplicated(keep="last")]


def sec_releases(submissions):
    """Earnings releases from an SEC submissions JSON: 8-K filings with item 2.02, at their acceptance time.

    Companies file the press release as an 8-K within minutes of publishing it, so the acceptance time
    is a close stand-in for the release time. Foreign filers (6-K) have no item codes and give nothing.
    """

    recent = (submissions or {}).get("filings", {}).get("recent", {})
    forms, items = recent.get("form", []), recent.get("items", [])
    accepted = recent.get("acceptanceDateTime", [])
    times = [accepted[i] for i, form in enumerate(forms)
             if form in ("8-K", "8-K/A") and i < len(items) and "2.02" in str(items[i]).split(",") and i < len(accepted)]
    if not times:
        return empty_releases()
    index = to_eastern(pd.to_datetime(times, utc=True, errors="coerce"))
    releases = pd.DataFrame({"EPS Estimate": np.nan, "Reported EPS": np.nan, "Surprise(%)": np.nan,
                             "Source": "SEC 8-K"}, index=index[~index.isna()]).sort_index()
    # An 8-K/A or a second filing the same day is the same release.
    return releases[~releases.index.normalize().duplicated(keep="first")]


def merge_releases(yahoo, sec, days=7):
    """Yahoo's releases plus SEC releases with no Yahoo release within ``days``.

    A Yahoo release with no time of day (midnight) takes the SEC filing's time when one is on the same day.
    """

    yahoo = yahoo if yahoo is not None else empty_releases()
    sec = sec if sec is not None else empty_releases()
    if sec.empty:
        return yahoo.copy()
    if yahoo.empty:
        return sec.copy()
    merged = yahoo.copy()
    index = list(merged.index)
    for position, when in enumerate(index):
        if when == when.normalize():
            same_day = sec.index[sec.index.normalize() == when]
            if len(same_day):
                index[position] = same_day[0]
    merged.index = pd.DatetimeIndex(index)
    gaps = np.array([np.abs(merged.index - when).min() > pd.Timedelta(days=days) for when in sec.index])
    return pd.concat([merged, sec[gaps]]).sort_index()


def timing(when):
    """'before', 'during' or 'after' the regular session, or 'unknown' for a date with no time."""

    time = when - when.normalize()
    if time == pd.Timedelta(0):
        return "unknown"
    if time < OPEN:
        return "before"
    return "during" if time < CLOSE else "after"


def reaction_day(when, sessions):
    """The first trading session that could react to a release at ``when``, or None when it hasn't happened.

    After the close it is the next session; before the open, during the session, or with no time, that day's.
    """

    day = when.normalize()
    later = sessions[sessions > day] if timing(when) == "after" else sessions[sessions >= day]
    return later[0] if len(later) else None


def releases_with_reaction(releases, close, quarter_ends=None, max_quarter_days=120):
    """Past releases with their timing, quarter, reaction day and the price change after 1 and 5 sessions.

    The change is measured from the close before the reaction day, so it includes the opening gap. With
    ``quarter_ends`` each release is matched to the latest fiscal quarter that ended before it.
    """

    close = close.dropna().sort_index() if close is not None else pd.Series(dtype=float)
    close = close[~close.index.duplicated(keep="last")]
    sessions = pd.DatetimeIndex(close.index).normalize()
    quarter_ends = pd.DatetimeIndex(sorted(quarter_ends)) if quarter_ends is not None else pd.DatetimeIndex([])
    rows = []
    for when, release in releases.iterrows():
        day = reaction_day(when, sessions)
        change = {1: np.nan, 5: np.nan}
        if day is not None:
            position = sessions.get_loc(day)
            if position > 0:
                base = close.iloc[position - 1]
                for sessions_after in change:
                    if position + sessions_after - 1 < len(close):
                        change[sessions_after] = (close.iloc[position + sessions_after - 1] / base - 1) * 100
        before = quarter_ends[(quarter_ends < when.normalize())
                              & (quarter_ends >= when.normalize() - pd.Timedelta(days=max_quarter_days))]
        rows.append({"Release": when, "Timing": timing(when), "Quarter end": before[-1] if len(before) else pd.NaT,
                     "EPS Estimate": release["EPS Estimate"], "Reported EPS": release["Reported EPS"],
                     "Surprise(%)": release["Surprise(%)"], "Reaction day": day if day is not None else pd.NaT,
                     "1-day %": change[1], "5-day %": change[5], "Source": release["Source"]})
    columns = ["Release", "Timing", "Quarter end", "EPS Estimate", "Reported EPS", "Surprise(%)", "Reaction day",
               "1-day %", "5-day %", "Source"]
    return pd.DataFrame(rows, columns=columns)


def past_and_next(releases, now, snapshot=None):
    """Split releases into those already out by ``now`` and the next scheduled one.

    Returns ``(past, next_release)``; ``next_release`` is ``(timestamp, is_estimate)`` or None. Yahoo's
    calendar row comes first; its quote fields (``earningsTimestampStart``, ``isEarningsDateEstimate``)
    fill in when the calendar has no future date. A release counts as out once its time has passed.
    """

    now = pd.Timestamp(now)
    past = releases[releases.index <= now]
    # A future row can still carry a reported EPS when Yahoo's time is off; that one is out too.
    future = releases[(releases.index > now) & releases["Reported EPS"].isna()]
    past = pd.concat([past, releases[(releases.index > now) & releases["Reported EPS"].notna()]]).sort_index()
    snapshot = snapshot or {}
    estimate = snapshot.get("isEarningsDateEstimate")
    quoted = snapshot.get("earningsTimestampStart") or snapshot.get("earningsTimestamp")
    quoted = to_eastern([pd.to_datetime(quoted, unit="s", utc=True)])[0] if quoted else None
    if len(future):
        when = future.index[0]
        same_day = quoted is not None and quoted.normalize() == when.normalize()
        return past, (when, bool(estimate) if same_day and estimate is not None else None)
    if quoted is not None and quoted > now:
        return past, (quoted, bool(estimate) if estimate is not None else None)
    return past, None


def describe_release(row):
    """Several short lines on one release, for the chart's tap box."""

    when = row["Release"]
    lines = [f"Earnings {when:%a %d %b %Y}, {TIMING_LABELS[row['Timing']]}"]
    if pd.notna(row["Quarter end"]):
        lines.append(f"Quarter to {row['Quarter end']:%d %b %Y}")
    if pd.notna(row["Reported EPS"]):
        eps = f"EPS {row['Reported EPS']:.2f}"
        if pd.notna(row["EPS Estimate"]):
            eps += f" vs est. {row['EPS Estimate']:.2f}"
        if pd.notna(row["Surprise(%)"]):
            eps += f" ({row['Surprise(%)']:+.1f}%)"
        lines.append(eps)
    moves = [f"{label} {row[column]:+.1f}%" for label, column in [("next day", "1-day %"), ("5 days", "5-day %")]
             if pd.notna(row[column])]
    if moves:
        lines.append("Price " + ", ".join(moves))
    if row["Source"] != "Yahoo":
        lines.append(f"Date from {row['Source']}")
    return "\n".join(lines)


def next_release_text(next_release, today):
    """'Next earnings: Tue 17 Nov 2026, after close (confirmed), in 16 days.' or a note that none is known."""

    if next_release is None:
        return "Next earnings: no date published yet."
    when, estimate = next_release
    days = (when.normalize() - pd.Timestamp(today).normalize()).days
    status = {True: " (estimated)", False: " (confirmed)"}.get(estimate, "")
    timing_text = "" if timing(when) == "unknown" else f", {TIMING_LABELS[timing(when)]}"
    away = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
    return f"Next earnings: {when:%a %d %b %Y}{timing_text}{status}, {away}."


def release_table(reactions, count=8):
    """The latest ``count`` releases, newest first, as display columns."""

    latest = reactions.sort_values("Release").tail(count).iloc[::-1]
    return pd.DataFrame({
        "Date": [f"{when:%Y-%m-%d}" for when in latest["Release"]],
        "When": [TIMING_LABELS[value] for value in latest["Timing"]],
        "Quarter to": [f"{when:%Y-%m-%d}" if pd.notna(when) else "" for when in latest["Quarter end"]],
        "EPS est.": latest["EPS Estimate"].to_numpy(),
        "EPS": latest["Reported EPS"].to_numpy(),
        "Surprise %": latest["Surprise(%)"].to_numpy(),
        "1-day %": latest["1-day %"].to_numpy(),
        "5-day %": latest["5-day %"].to_numpy(),
        "Source": latest["Source"].to_numpy(),
    })


def reaction_summary(reactions, count=8):
    """One sentence on the latest ``count`` releases: beats, the typical move and how often it went up."""

    latest = reactions.sort_values("Release").tail(count)
    moves = latest["1-day %"].dropna()
    surprises = latest["Surprise(%)"].dropna()
    parts = []
    if len(surprises):
        parts.append(f"EPS beat the estimate {int((surprises > 0).sum())} of {len(surprises)} times")
    if len(moves):
        parts.append(f"the next-day move averaged {moves.abs().mean():.1f}% either way "
                     f"(up {int((moves > 0).sum())} of {len(moves)})")
    if not parts:
        return ""
    text = " and ".join(parts)
    return f"Last {len(latest)} releases: {text}."
