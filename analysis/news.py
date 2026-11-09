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

SEMI_AI_SECTOR = "Semiconductors & AI"
# Chip makers and chip equipment (GICS sub-industries, and Yahoo's industry names for non-S&P stocks) ...
SEMI_INDUSTRIES = {"Semiconductors", "Semiconductor Materials & Equipment", "Semiconductor Equipment & Materials"}
# ... plus companies whose business is mainly AI software or AI data center hardware.
AI_COMPANIES = {"PLTR", "SMCI", "ARM", "CRWV", "VRT", "ANET", "ALAB", "CRDO", "AI", "SOUN", "NBIS"}

# About six of the largest S&P 500 companies in each GICS sector (late 2026), with chips and AI as a sector of its own.
SECTOR_LEADERS = {
    SEMI_AI_SECTOR: ["NVDA", "AVGO", "AMD", "TSM", "MU", "ASML", "ARM", "PLTR"],
    "Information Technology": ["MSFT", "AAPL", "ORCL", "CRM", "CSCO", "IBM"],
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


def with_semi_ai_sector(sectors, industries):
    """Sectors with chip makers, chip equipment and AI companies moved into ``SEMI_AI_SECTOR``."""

    return {ticker: SEMI_AI_SECTOR if industries.get(ticker) in SEMI_INDUSTRIES or ticker in AI_COMPANIES else sector
            for ticker, sector in sectors.items()}


# Tone of a story from its words: a finance word list (in the spirit of Loughran-McDonald), not a language model.
# Words that only hint at a direction ("best", "dividend", "deal", "higher") are left out, because they show up as
# often in buying guides and mixed results as in good news.
POSITIVE_WORDS = {
    "beat", "beats", "tops", "topped", "exceeds", "exceeded", "surge", "surges", "surged", "soar", "soars", "soared",
    "jump", "jumps", "jumped", "rally", "rallies", "rallied", "gain", "gains", "gained", "rise", "rises", "rose",
    "climb", "climbs", "climbed", "rebound", "rebounds", "rebounded", "upgrade", "upgrades", "upgraded",
    "outperform", "outperforms", "outperformance", "bullish", "record", "strong", "stronger", "boost", "boosts",
    "boosted", "boosting", "raises", "raised", "lifts", "lifted", "wins", "won", "approval", "approved", "approves",
    "nod", "profitable", "grows", "grew", "expands", "expanding", "expansion", "breakthrough", "optimism",
    "upbeat", "robust", "accelerates", "accelerating", "surpass", "surpasses", "surpassed", "soaring", "jumping",
    "rallying", "skyrockets", "rockets", "booming", "invest", "invests", "partnerships", "secures", "riding",
    "winners", "rebounds", "bounces", "lands",
}
NEGATIVE_WORDS = {
    "miss", "misses", "missed", "fall", "falls", "fell", "drop", "drops", "dropped", "plunge", "plunges", "plunged",
    "tumble", "tumbles", "tumbled", "sink", "sinks", "sank", "slump", "slumps", "slumped", "slide", "slides", "slid",
    "decline", "declines", "declined", "downgrade", "downgrades", "downgraded", "lawsuit", "sued", "sues", "probe",
    "investigation", "recall", "recalls", "loss", "losses", "losing", "lost", "weak", "weaker", "weakness", "warns",
    "layoffs", "layoff", "bearish", "selloff", "sell-off", "rout", "crater", "craters", "cratered", "crumble",
    "crumbles", "wipes", "wiped", "slashes", "slashed", "underperforming", "disappoints", "disappointing", "lags",
    "struggles", "struggling", "risks", "headwinds", "downturn", "shortfall", "collapse", "collapses", "slowdown",
    "broken", "falling", "dropping", "plunging", "tumbling", "sinking", "sliding", "slumping", "declining", "plummet",
    "plummets", "plummeted", "turmoil", "suffer", "suffers", "punished", "fleeing", "squeezing", "bankruptcy", "fraud",
    "losers", "doubt", "doubts", "trouble", "alarm", "dips", "dipped",
}
NEGATIONS = {"not", "no", "never", "without", "isn't", "wasn't", "doesn't", "didn't", "won't", "fails", "failed"}
# Phrases whose meaning is not in their single words.
POSITIVE_PHRASES = ["price target raised", "raises price target", "raises guidance", "raises outlook", "all-time high",
                    "record high", "buy rating", "beat estimates", "beats estimates", "top estimates", "tops estimates",
                    "view up", "outlook up", "outlook raised", "best day", "back above", "net buyers",
                    "(lead|leads|move|moves|trade|trades|close|closes|push|pushes) (\\w+ ){0,4}higher"]
NEGATIVE_PHRASES = ["cut in half", "brick wall", "hit a wall", "price target cut", "cuts price target",
                    "cuts guidance", "lowers guidance", "cuts outlook", "lowers outlook", "misses estimates",
                    "sell rating", "lost trillions", "down \\d+%", "off \\d+%", "risks? rise", "worst day",
                    "falls below", "back below", "net sellers", "could go down",
                    "(lead|leads|move|moves|trade|trades|close|closes|drag|drags) (\\w+ ){0,4}lower"]
# Words that look like tone words but are not, in these phrases.
TONE_FREE_PHRASES = ["weight loss", "loss-making", "misses about", "stop loss", "stop-loss"]
# Buying guides, comparisons and opinion pieces: about investing in a stock, not news about it.
NOT_NEWS = re.compile(
    r"^\s*(top )?\d+ |stocks? to |to buy|buy (now|today|hand)|better (dividend )?stock|"
    r"\bvs\.? |\bversus\b|should you|what you need to know|here'?s (how|what|why)|is it (time|too late)|"
    r"stock picks|passive income|strategy|could (cash|gain|go)|bargain|watch list|to watch|reports soon|"
    r"in the cards|what to expect|misses about|i'?d buy|i'?m buying|i'?ll buy|i would buy|hold (it )?forever|"
    r"^(do|does|are|is|will|can|could|should|would|has|have|what|why|how)\b.*\?\s*$",
    re.IGNORECASE,
)
TONES = ["Positive", "Negative", "Neutral"]


def _side_counts(text):
    """(positive, negative) word and phrase counts in ``text``; a word right after "not", "no" ... counts reversed."""

    text = str(text or "").lower()
    for phrase in TONE_FREE_PHRASES:
        text = text.replace(phrase, " ")
    positive = negative = 0
    # Count a phrase once, then blank it out so its words are not counted again.
    for phrases, side in ((POSITIVE_PHRASES, 1), (NEGATIVE_PHRASES, -1)):
        for phrase in phrases:
            found = len(re.findall(phrase, text))
            if found:
                text = re.sub(phrase, " ", text)
                positive += found * (side > 0)
                negative += found * (side < 0)
    words = re.findall(r"[a-z][a-z'\-]*", text)
    for position, word in enumerate(words):
        sign = 1 if word in POSITIVE_WORDS else -1 if word in NEGATIVE_WORDS else 0
        if sign and position and words[position - 1] in NEGATIONS:
            sign = -sign
        positive += sign > 0
        negative += sign < 0
    return positive, negative


def tone_score(title, summary=""):
    """Positive minus negative, read from the headline.

    The summary only decides when the headline has no tone words at all, and then only with a clear margin (2+),
    because summaries often describe the market around a story rather than the story. A headline with words on
    both sides ("earnings miss, revenue beats"), a buying guide or an opinion piece scores 0 (neutral).
    """

    title = str(title or "")
    lowered = title.lower()
    has_phrase = any(re.search(phrase, lowered) for phrase in POSITIVE_PHRASES + NEGATIVE_PHRASES)
    # A buying guide is neutral unless its headline also reports a clear event ("cut in half", "down 20%").
    if NOT_NEWS.search(title) and not has_phrase:
        return 0
    # "X rebounds after selloff": what happened before "after" / "following" / "despite" is only background.
    lead = re.split(r"\b(?:after|following|despite)\b", title, maxsplit=1, flags=re.IGNORECASE)[0]
    positive, negative = _side_counts(lead)
    if not (positive or negative):
        positive, negative = _side_counts(title)
    if positive and negative:
        return 0
    if positive or negative:
        return positive - negative
    positive, negative = _side_counts(summary)
    score = positive - negative
    return score if abs(score) >= 2 and not (positive and negative and min(positive, negative) * 2 > max(positive, negative)) else 0


def story_tone(title, summary=""):
    score = tone_score(title, summary)
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def add_tone(frame):
    """``frame`` with a ``tone`` column (Positive / Negative / Neutral)."""

    frame = frame.copy()
    frame["tone"] = [story_tone(title, summary) for title, summary in zip(frame["title"], frame["summary"])] \
        if not frame.empty else pd.Series(dtype=str)
    return frame


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
    tone = str(row.get("tone") or "").lower()
    tone_class = f" news-{tone}" if tone in ("positive", "negative") else ""
    return (
        "<div class='news-card'>"
        f"<a class='news-title{tone_class}' href='{html.escape(str(row.get('url') or ''), quote=True)}' target='_blank' "
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
.news-title.news-positive {color: #16a34a}
.news-title.news-negative {color: #dc2626}
.news-tone-head {font-size: 0.85rem; font-weight: 600; margin: 0.6rem 0 0.1rem}
.news-tone-head.news-positive {color: #16a34a}
.news-tone-head.news-negative {color: #dc2626}
.news-meta {font-size: 0.8rem; opacity: 0.7; margin-top: 0.2rem; line-height: 1.6}
.news-chip {display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0 0.4rem; margin-right: 0.25rem;
            border-radius: 0.6rem; background: rgba(128,128,128,0.18)}
.news-summary {font-size: 0.86rem; opacity: 0.85; margin-top: 0.25rem; line-height: 1.4;
               display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden}
</style>
"""
