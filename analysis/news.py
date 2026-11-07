"""Stock news from Yahoo Finance, grouped by industry.

Yahoo's news search (``yf.Ticker(symbol).get_news``) returns the latest
stories about a ticker. Fetching it for a set of companies whose sector and
industry are known, and filing every story under the industries of the
companies it was found for (or names), gives a news feed per industry.
"""

import html
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# About six of the largest S&P 500 companies in each GICS sector (late 2026).
SECTOR_LEADERS = {
    "Information Technology": ["NVDA", "MSFT", "AAPL", "AVGO", "ORCL", "AMD"],
    "Communication Services": ["GOOGL", "META", "NFLX", "TMUS", "DIS", "VZ"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "NKE"],
    "Consumer Staples": ["WMT", "COST", "PG", "KO", "PEP", "PM"],
    "Health Care": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO"],
    "Financials": ["BRK-B", "JPM", "V", "MA", "BAC", "GS"],
    "Industrials": ["GE", "CAT", "RTX", "UBER", "HON", "BA"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "KMI"],
    "Materials": ["LIN", "SHW", "APD", "FCX", "NEM", "ECL"],
    "Real Estate": ["PLD", "AMT", "EQIX", "WELL", "SPG", "O"],
    "Utilities": ["NEE", "SO", "DUK", "CEG", "VST", "AEP"],
}
MARKET_GROUP = "Whole market"
MARKET_SYMBOLS = ["^GSPC", "^IXIC"]
NEWS_PERIODS = {"24 hours": 1, "3 days": 3, "1 week": 7, "1 month": 30}
ARTICLE_COLUMNS = ["id", "title", "summary", "publisher", "url", "published", "mentions", "fetched_for"]


def _url(content, *keys):
    for key in keys:
        value = content.get(key)
        if isinstance(value, dict) and value.get("url"):
            return value["url"]
    return ""


def parse_article(item, fetched_for):
    """One story from Yahoo's news list as a flat dict, or None when it has no title or link.

    yfinance 0.2.5x and later wrap each story in ``{"id", "content": {...}}``; older versions
    returned a flat dict with ``providerPublishTime``. Both are read.
    """

    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content is not None:
        if content.get("contentType") not in (None, "STORY", "VIDEO"):
            return None
        title = content.get("title") or ""
        url = _url(content, "canonicalUrl", "clickThroughUrl")
        published = pd.to_datetime(content.get("pubDate") or content.get("displayTime"), utc=True, errors="coerce")
        publisher = (content.get("provider") or {}).get("displayName") or ""
        summary = content.get("summary") or content.get("description") or ""
        tickers = [entry.get("symbol") for entry in ((content.get("finance") or {}).get("stockTickers") or [])
                   if isinstance(entry, dict) and entry.get("symbol")]
        story_id = item.get("id") or content.get("id") or url
    else:
        title = item.get("title") or ""
        url = item.get("link") or ""
        stamp = item.get("providerPublishTime")
        published = pd.to_datetime(stamp, unit="s", utc=True, errors="coerce") if stamp else pd.NaT
        publisher = item.get("publisher") or ""
        summary = item.get("summary") or ""
        tickers = list(item.get("relatedTickers") or [])
        story_id = item.get("uuid") or url
    if not title or not url:
        return None
    # Summaries sometimes carry HTML tags; keep plain text.
    summary = re.sub(r"<[^>]+>", " ", summary)
    summary = re.sub(r"\s+", " ", html.unescape(summary)).strip()
    return {
        "id": str(story_id), "title": title.strip(), "summary": summary, "publisher": publisher, "url": url,
        "published": published, "mentions": [str(ticker).upper() for ticker in tickers], "fetched_for": [fetched_for],
    }


def fetch_ticker_news(ticker, count=20):
    """Latest stories Yahoo has for one ticker (raw list; [] on any error). Needs network."""

    import yfinance as yf

    try:
        return yf.Ticker(ticker).get_news(count=count, tab="news") or []
    except Exception:
        return []


def collect_news(tickers, fetch=fetch_ticker_news, workers=8, progress=None):
    """Stories for many tickers as one DataFrame, one row per story (same story found for several tickers is merged).

    Yahoo's news search answers parallel requests fine (unlike ``.info``), so tickers are fetched ``workers`` at a time.
    """

    stories = {}
    tickers = list(dict.fromkeys(tickers))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, (ticker, items) in enumerate(zip(tickers, pool.map(fetch, tickers)), start=1):
            for item in items or []:
                article = parse_article(item, ticker)
                if article is None:
                    continue
                known = stories.get(article["id"])
                if known is None:
                    stories[article["id"]] = article
                elif ticker not in known["fetched_for"]:
                    known["fetched_for"].append(ticker)
            if progress is not None:
                progress(done / max(1, len(tickers)))
    frame = pd.DataFrame(list(stories.values()), columns=ARTICLE_COLUMNS)
    if frame.empty:
        return frame
    frame["published"] = pd.to_datetime(frame["published"], utc=True)
    return frame.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)


MAX_MENTIONS = 3


def story_tickers(row, groups, max_mentions=MAX_MENTIONS):
    """Tickers a story is about, limited to the companies in ``groups``: the ones it was found for, then the ones it names.

    Market round-ups name a dozen companies in passing; when a story names more than ``max_mentions`` of them,
    only the companies it was found for count, so a round-up does not land in every industry.
    """

    found = [ticker for ticker in row["fetched_for"] if ticker in groups]
    named = [ticker for ticker in row["mentions"] if ticker in groups]
    if len(named) > max_mentions:
        named = []
    return list(dict.fromkeys(found + named))


def filter_news(frame, days=None, search="", now=None):
    """Stories published in the last ``days`` days whose title, summary or publisher contains ``search``."""

    if frame.empty:
        return frame
    keep = pd.Series(True, index=frame.index)
    if days:
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        keep &= frame["published"].notna() & (frame["published"] >= now - pd.Timedelta(days=days))
    terms = [term for term in str(search or "").lower().split() if term]
    if terms:
        text = (frame["title"] + " " + frame["summary"] + " " + frame["publisher"] + " "
                + frame["fetched_for"].map(" ".join) + " " + frame["mentions"].map(" ".join)).str.lower()
        for term in terms:
            keep &= text.str.contains(term, regex=False)
    return frame[keep]


def group_news(frame, groups, market_symbols=MARKET_SYMBOLS):
    """{group: DataFrame of its stories, newest first}, busiest group first.

    ``groups`` maps ticker -> group name (an industry or sector). A story goes into the group of every company
    it is about, once per group, with a ``tickers`` column listing that group's companies in it. Stories found
    only for the market symbols go into ``MARKET_GROUP``.
    """

    if frame.empty:
        return {}
    buckets = {}
    for index, row in frame.iterrows():
        tickers = story_tickers(row, groups)
        by_group = {}
        for ticker in tickers:
            by_group.setdefault(groups[ticker], []).append(ticker)
        if not by_group and any(symbol in market_symbols for symbol in row["fetched_for"]):
            by_group[MARKET_GROUP] = []
        for group, members in by_group.items():
            buckets.setdefault(group, []).append((index, members))
    result = {}
    for group, entries in buckets.items():
        part = frame.loc[[index for index, _ in entries]].copy()
        part["tickers"] = [members for _, members in entries]
        result[group] = part.sort_values("published", ascending=False, na_position="last")

    def order(item):
        group, part = item
        latest = part["published"].max()
        return (group != MARKET_GROUP, -len(part), -(latest.value if pd.notna(latest) else 0), group)

    return dict(sorted(result.items(), key=order))


def time_ago(published, now=None):
    """'5 min ago', '3 h ago', '2 d ago' or a date for older stories."""

    if published is None or pd.isna(published):
        return ""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    seconds = max(0.0, (now - published).total_seconds())
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)} d ago"
    return f"{published:%d %b %Y}"


def _text(value):
    # Streamlit markdown turns "$...$" into LaTeX, so dollar signs are written as an HTML entity.
    return html.escape(str(value or "")).replace("$", "&#36;")


def story_card_html(row, now=None, summary_chars=220):
    """One story as a compact HTML card: linked title, publisher · age · tickers, a short summary."""

    tickers = row.get("tickers") if isinstance(row.get("tickers"), list) else []
    chips = "".join(f"<span class='news-chip'>{_text(ticker)}</span>" for ticker in tickers)
    meta = " · ".join(part for part in [_text(row.get("publisher")), _text(time_ago(row.get("published"), now))] if part)
    summary = str(row.get("summary") or "")
    if len(summary) > summary_chars:
        summary = summary[:summary_chars].rsplit(" ", 1)[0] + "…"
    summary_html = f"<div class='news-summary'>{_text(summary)}</div>" if summary else ""
    return (
        "<div class='news-card'>"
        f"<a class='news-title' href='{html.escape(str(row.get('url') or ''), quote=True)}' target='_blank' "
        f"rel='noopener noreferrer'>{_text(row.get('title'))}</a>"
        f"<div class='news-meta'>{meta}{(' ' + chips) if chips else ''}</div>"
        f"{summary_html}</div>"
    )


NEWS_CSS = """
<style>
.news-card {padding: 0.55rem 0 0.6rem; border-bottom: 1px solid rgba(128,128,128,0.25)}
.news-card:last-child {border-bottom: none}
.news-title {font-weight: 600; font-size: 1rem; line-height: 1.3; text-decoration: none; color: inherit}
.news-title:hover {text-decoration: underline}
.news-meta {font-size: 0.8rem; opacity: 0.7; margin-top: 0.2rem; line-height: 1.6}
.news-chip {display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0 0.4rem; margin-right: 0.25rem;
            border-radius: 0.6rem; background: rgba(128,128,128,0.18)}
.news-summary {font-size: 0.86rem; opacity: 0.85; margin-top: 0.25rem; line-height: 1.4;
               display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden}
</style>
"""
