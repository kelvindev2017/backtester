"""Support and resistance zones identified from past price and volume.

Two independent sources are combined:

* Swing pivots: confirmed swing highs/lows, clustered when they sit within a
  fraction of an ATR of each other. More touches and more volume traded at
  the pivots make a stronger zone.
* Volume profile: price bins where unusually heavy volume changed hands
  (high-volume nodes). Price tends to pause or reverse at these areas.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PriceZone:
    low: float
    high: float
    price: float
    touches: int
    strength: float
    source: str
    last_seen: pd.Timestamp = None
    notes: list = field(default_factory=list)

    def describe(self):
        label = f"{self.low:,.2f}-{self.high:,.2f}"
        if self.source == "pivot":
            seen = f", last {self.last_seen:%Y-%m-%d}" if self.last_seen is not None else ""
            return f"{label} ({self.touches} swing touches{seen})"
        if self.source == "breakout":
            return f"{label} (broken breakout level)"
        if self.source in ("period_high", "period_low"):
            kind = "high" if self.source == "period_high" else "low"
            seen = f", {self.last_seen:%Y-%m-%d}" if self.last_seen is not None else ""
            return f"{label} (lookback {kind}{seen})"
        return f"{label} (high-volume node)"


def find_pivots(df, left=5, right=5):
    """Return confirmed swing highs and lows.

    A pivot at bar ``i`` needs ``right`` later bars to be confirmed, so the
    ``Confirmed_At`` column records the first bar on which the pivot was known.
    """

    if df is None or len(df) < left + right + 1:
        return pd.DataFrame(columns=["Price", "Type", "Volume", "Confirmed_At"])

    window = left + right + 1
    rolling_high = df["High"].rolling(window, center=True).max()
    rolling_low = df["Low"].rolling(window, center=True).min()
    is_high = df["High"].eq(rolling_high)
    is_low = df["Low"].eq(rolling_low)

    confirmed_at = pd.Series(df.index, index=df.index).shift(-right)
    records = []
    for timestamp in df.index[is_high.to_numpy()]:
        records.append((timestamp, float(df.at[timestamp, "High"]), "high", float(df.at[timestamp, "Volume"]), confirmed_at[timestamp]))
    for timestamp in df.index[is_low.to_numpy()]:
        records.append((timestamp, float(df.at[timestamp, "Low"]), "low", float(df.at[timestamp, "Volume"]), confirmed_at[timestamp]))

    pivots = pd.DataFrame(records, columns=["Date", "Price", "Type", "Volume", "Confirmed_At"])
    pivots = pivots.dropna(subset=["Confirmed_At"]).set_index("Date").sort_index()
    return pivots


def cluster_pivots(pivots, tolerance):
    """Group pivot prices that lie within ``tolerance`` of the running cluster mean."""

    if pivots is None or pivots.empty or not np.isfinite(tolerance) or tolerance <= 0:
        return []

    ordered = pivots.sort_values("Price")
    clusters, current = [], []
    for timestamp, row in ordered.iterrows():
        if current and row["Price"] - np.mean([item[1]["Price"] for item in current]) > tolerance:
            clusters.append(current)
            current = []
        current.append((timestamp, row))
    if current:
        clusters.append(current)

    total_volume = float(pivots["Volume"].sum()) or 1.0
    zones = []
    for cluster in clusters:
        prices = np.array([row["Price"] for _, row in cluster])
        volumes = np.array([row["Volume"] for _, row in cluster])
        weights = volumes if volumes.sum() > 0 else None
        touches = len(cluster)
        last_seen = max(timestamp for timestamp, _ in cluster)
        volume_share = float(volumes.sum()) / total_volume
        zones.append(
            PriceZone(
                low=float(prices.min()),
                high=float(prices.max()),
                price=float(np.average(prices, weights=weights)),
                touches=touches,
                strength=touches + 5.0 * volume_share,
                source="pivot",
                last_seen=pd.Timestamp(last_seen),
            )
        )
    return zones


def volume_profile_nodes(df, bins=40, top_n=4):
    """Return the heaviest volume-at-price bins as zones.

    Each bar's volume is spread evenly across the bins its high-low range covers.
    """

    if df is None or df.empty:
        return []

    price_low = float(df["Low"].min())
    price_high = float(df["High"].max())
    if not np.isfinite(price_low) or price_high <= price_low:
        return []

    edges = np.linspace(price_low, price_high, bins + 1)
    profile = np.zeros(bins)
    lows = df["Low"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    volumes = df["Volume"].fillna(0).to_numpy(dtype=float)
    for low, high, volume in zip(lows, highs, volumes):
        first = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, bins - 1))
        last = int(np.clip(np.searchsorted(edges, high, side="right") - 1, 0, bins - 1))
        profile[first:last + 1] += volume / (last - first + 1)

    if profile.sum() <= 0:
        return []

    # Keep local maxima that are clearly above the average bin.
    candidates = []
    for index in range(bins):
        left = profile[index - 1] if index > 0 else -np.inf
        right = profile[index + 1] if index < bins - 1 else -np.inf
        if profile[index] >= left and profile[index] >= right and profile[index] > 1.25 * profile.mean():
            candidates.append(index)

    candidates = sorted(candidates, key=lambda index: profile[index], reverse=True)[:top_n]
    share = profile / profile.sum()
    return [
        PriceZone(
            low=float(edges[index]),
            high=float(edges[index + 1]),
            price=float((edges[index] + edges[index + 1]) / 2),
            touches=0,
            strength=float(share[index] * bins),
            source="volume",
        )
        for index in candidates
    ]


def support_resistance(df, atr_value, lookback=250, pivot_bars=5, tolerance_atr=0.5, min_touches=2):
    """Split recent zones into supports (below the last close) and resistances (above).

    Returns ``(supports, resistances)``; supports are ordered nearest-first
    (descending price) and resistances nearest-first (ascending price).
    """

    recent = df.iloc[-int(lookback):] if lookback else df
    if recent.empty or not np.isfinite(atr_value) or atr_value <= 0:
        return [], []

    last_close = float(recent["Close"].iloc[-1])
    pivots = find_pivots(recent, pivot_bars, pivot_bars)
    pivot_zones = [
        zone for zone in cluster_pivots(pivots, tolerance_atr * atr_value)
        if zone.touches >= min_touches
    ]
    zones = pivot_zones + volume_profile_nodes(recent)

    # The lookback extremes matter even with a single touch (e.g. the 52-week high).
    tolerance = tolerance_atr * atr_value
    for source, timestamp, extreme, band in [
        ("period_high", recent["High"].idxmax(), float(recent["High"].max()), (-0.25, 0.0)),
        ("period_low", recent["Low"].idxmin(), float(recent["Low"].min()), (0.0, 0.25)),
    ]:
        if zone_near(zones, extreme, tolerance) is None:
            zones.append(
                PriceZone(
                    low=extreme + band[0] * atr_value,
                    high=extreme + band[1] * atr_value,
                    price=extreme,
                    touches=1,
                    strength=1.0,
                    source=source,
                    last_seen=pd.Timestamp(timestamp),
                )
            )

    supports = sorted(
        [zone for zone in zones if zone.price < last_close],
        key=lambda zone: zone.price,
        reverse=True,
    )
    resistances = sorted(
        [zone for zone in zones if zone.price >= last_close],
        key=lambda zone: zone.price,
    )
    return supports, resistances


def zone_near(zones, price, tolerance):
    """Return the strongest zone whose band is within ``tolerance`` of ``price``."""

    matches = [
        zone for zone in zones
        if zone.low - tolerance <= price <= zone.high + tolerance
    ]
    if not matches:
        return None
    return max(matches, key=lambda zone: zone.strength)
