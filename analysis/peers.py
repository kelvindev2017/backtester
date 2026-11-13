"""Pick a stock's peers (competitors) automatically, for the Valuation Model's P/E anchor.

Two sources, best first:

1. The S&P 500 members in the stock's GICS sub-industry (e.g. "Transaction & Payment Processing Services"),
   the narrowest standard grouping; the app ranks them by dollars traded a day.
2. The largest companies in the stock's Yahoo industry, ranked by how close their size is to the stock's, so a
   mid-cap is compared with mid-caps rather than only with the giants of its industry.

The user can always edit the list; these functions only make the first guess.
"""

import numpy as np
import pandas as pd


def rank_by_size(weights, target=None, count=5, exclude=()):
    """Tickers of ``weights`` (ticker -> market weight or market cap) closest in size to ``target`` first.

    Closeness is measured on a log scale (twice as big counts the same as half as big). Without a target, the
    largest come first. ``exclude`` (the stock itself, peers already chosen) is left out.
    """

    weights = pd.to_numeric(pd.Series(weights, dtype=float), errors="coerce")
    skip = {str(ticker).upper() for ticker in exclude}
    weights = weights.loc[(weights > 0) & ~weights.index.map(lambda ticker: str(ticker).upper() in skip)]
    if weights.empty:
        return []
    if target is None or not np.isfinite(target) or target <= 0:
        return list(weights.sort_values(ascending=False).index[:count])
    distance = (np.log(weights) - np.log(target)).abs()
    # Ties (equal distance) go to the larger company.
    order = pd.DataFrame({"distance": distance, "size": -weights}).sort_values(["distance", "size"])
    return list(order.index[:count])


def target_weight(weights, symbol, market_cap=None, reference=None):
    """The stock's own market weight in its industry table.

    Its own row when it is listed; otherwise its market cap scaled by a listed company's cap and weight
    (``reference`` = (ticker, market cap)). None when neither is known."""

    weights = pd.Series(weights, dtype=float)
    if symbol in weights.index and weights[symbol] > 0:
        return float(weights[symbol])
    if market_cap and reference:
        ticker, cap = reference
        if ticker in weights.index and cap and weights[ticker] > 0:
            return float(market_cap) * float(weights[ticker]) / float(cap)
    return None


def merge_peers(groups, symbol, count=5):
    """The first ``count`` distinct tickers across ``groups`` (lists, best source first), without the stock."""

    peers = []
    for group in groups:
        for ticker in group:
            ticker = str(ticker).strip().upper()
            if ticker and ticker != symbol.upper() and ticker not in peers:
                peers.append(ticker)
    return peers[:count]
