from pathlib import Path

import pandas as pd

from analysis.news import (
    MARKET_GROUP,
    SEMI_AI_SECTOR,
    add_tone,
    story_tone,
    with_semi_ai_sector,
    collect_news,
    filter_news,
    group_news,
    parse_article,
    story_card_html,
    time_ago,
)

NOW = pd.Timestamp("2026-11-06 12:00", tz="UTC")


def story(story_id, title, hours_ago, tickers=(), summary="", publisher="Wire"):
    return {"id": story_id, "content": {
        "contentType": "STORY", "title": title, "summary": summary,
        "pubDate": f"{NOW - pd.Timedelta(hours=hours_ago):%Y-%m-%dT%H:%M:%SZ}",
        "provider": {"displayName": publisher}, "canonicalUrl": {"url": f"https://example.com/{story_id}"},
        "finance": {"stockTickers": [{"symbol": ticker} for ticker in tickers]}}}


FEEDS = {
    "NVDA": [story("a", "Nvidia and AMD rally", 2, ["NVDA", "AMD", "JPM"]), story("b", "Nvidia earnings", 30, ["NVDA"])],
    "AMD": [story("a", "Nvidia and AMD rally", 2, ["NVDA", "AMD"])],
    "JPM": [story("c", "Banks gain", 5, ["JPM"], summary="Rates <b>up</b> &amp; loans")],
    "^GSPC": [story("d", "Stocks slip", 1, ["^GSPC"]), story("a", "Nvidia and AMD rally", 2)],
}
GROUPS = {"NVDA": "Semiconductors", "AMD": "Semiconductors", "JPM": "Banks"}


def test_parse_article_reads_new_and_old_formats():
    new = parse_article(FEEDS["JPM"][0], "JPM")
    assert new["title"] == "Banks gain" and new["summary"] == "Rates up & loans" and new["publisher"] == "Wire"
    assert new["published"] == NOW - pd.Timedelta(hours=5) and new["mentions"] == ["JPM"]
    old = parse_article({"uuid": "x", "title": "Old", "link": "https://e.com/x", "publisher": "P",
                         "providerPublishTime": int(NOW.timestamp()), "relatedTickers": ["aapl"]}, "AAPL")
    assert old["published"] == NOW and old["mentions"] == ["AAPL"]
    assert parse_article({"id": "y", "content": {"title": "", "canonicalUrl": {"url": "u"}}}, "AAPL") is None


def test_collect_and_group_news():
    news = collect_news(list(FEEDS), fetch=lambda ticker: FEEDS[ticker], workers=2)
    assert list(news["id"]) == ["d", "a", "c", "b"]
    assert news.set_index("id").at["a", "fetched_for"] == ["NVDA", "AMD", "^GSPC"]

    grouped = group_news(news, GROUPS)
    # Market news first, then the busiest industry (ties: latest story, then name); a story about two industries is in both.
    assert list(grouped) == [MARKET_GROUP, "Banks", "Semiconductors"]
    assert list(grouped["Semiconductors"]["id"]) == ["a", "b"]
    assert grouped["Semiconductors"].set_index("id").at["a", "tickers"] == ["NVDA", "AMD"]
    assert list(grouped["Banks"]["id"]) == ["a", "c"]
    assert list(grouped[MARKET_GROUP]["id"]) == ["d"]

    recent = filter_news(news, days=1, now=NOW)
    assert "b" not in set(recent["id"])
    assert list(filter_news(news, search="banks", now=NOW)["id"]) == ["c"]
    assert set(filter_news(news, search="amd", now=NOW)["id"]) == {"a"}


def test_card_and_time_ago():
    assert time_ago(NOW - pd.Timedelta(minutes=5), NOW) == "5 min ago"
    assert time_ago(NOW - pd.Timedelta(hours=3), NOW) == "3 h ago"
    assert time_ago(NOW - pd.Timedelta(days=2), NOW) == "2 d ago"
    card = story_card_html({"title": "Costs $5 <now>", "url": "https://e.com/?a=1&b='2'", "publisher": "Wire",
                            "published": NOW - pd.Timedelta(hours=1), "summary": "x " * 200, "tickers": ["NVDA"]}, NOW)
    assert "&#36;5 &lt;now&gt;" in card and "&#x27;2&#x27;" in card and "NVDA" in card and "…" in card


def test_story_tone():
    assert story_tone("Nvidia beats estimates and raises guidance") == "Positive"
    assert story_tone("Stocks crater as investors flee tech") == "Negative"
    assert story_tone("Q3 2026 earnings call transcript") == "Neutral"
    # A word after "not" counts the other way; the headline outweighs the summary.
    assert story_tone("Shares did not fall") == "Positive"
    assert story_tone("Shares jump", "Despite a loss") == "Positive"
    frame = add_tone(pd.DataFrame({"title": ["Shares plunge"], "summary": [""]}))
    assert list(frame["tone"]) == ["Negative"]
    card = story_card_html({"title": "Up", "url": "u", "tone": "Positive"})
    assert "news-title news-positive" in card


def test_semis_and_ai_get_their_own_sector():
    sectors = {"NVDA": "Information Technology", "AMAT": "Information Technology", "PLTR": "Information Technology",
               "MSFT": "Information Technology", "JPM": "Financials"}
    industries = {"NVDA": "Semiconductors", "AMAT": "Semiconductor Materials & Equipment", "MSFT": "Systems Software"}
    moved = with_semi_ai_sector(sectors, industries)
    assert [ticker for ticker, sector in moved.items() if sector == SEMI_AI_SECTOR] == ["NVDA", "AMAT", "PLTR"]
    assert moved["MSFT"] == "Information Technology" and moved["JPM"] == "Financials"


def test_tone_matches_hand_labels():
    # 157 Yahoo stories from Nov 2026, labelled by hand. "tuning" stories were used to build the word list,
    # "unseen" ones were labelled before any check. Coloured (green/red) labels should rarely be wrong.
    labels = pd.read_csv(Path(__file__).parent / "data" / "news_tone_labels.csv").fillna("")
    labels["got"] = [story_tone(title, summary) for title, summary in zip(labels["title"], labels["summary"])]
    assert (labels["got"] == labels["tone"]).mean() >= 0.8
    coloured = labels[labels["got"] != "Neutral"]
    assert (coloured["got"] == coloured["tone"]).mean() >= 0.85
    # Buying guides and "after the selloff" background no longer colour a story.
    assert story_tone("5 Best Dow Stocks to Buy for 2027", "The Dow tumbled 3.4% and the S&P 500 fell 4.4%.") == "Neutral"
    assert story_tone("Micron, AMD and Nvidia Lead AI Stocks Rebound After Historic Selloff") == "Positive"
    assert story_tone("Intel, Arm, and Marvell Have Each Been Cut in Half Since June. Only 1 Is a Stock I'd Buy Now.") == "Negative"
