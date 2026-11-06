import pandas as pd

from analysis.news import (
    MARKET_GROUP,
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
