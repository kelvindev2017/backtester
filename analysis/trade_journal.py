"""Trade journal: import an IBKR trade export, rebuild round trips, break the results down, and replay them.

Import (``load_ibkr_trades``) reads an IBKR Flex Query trade export (.xlsx or .csv) with at least Symbol,
TradeDate, Buy/Sell, Quantity and Price; CurrencyPrimary, Commission, OrderType, TradeID, OrderID and
DateTime are used when present. Only one currency is kept (US dollars by default) and currency conversions
such as USD.HKD are dropped.

Round trips (``round_trips``): the fills of each stock are grouped from flat to flat, long or short. A fill
that takes a position through zero is split: the part that closes ends one round trip and the rest opens the
next. P&L is the cash flow of the round trip, so it is exact without lot matching; commissions are IBKR's
(rebates count as negative costs).

What-if (``simulate``): every round trip is replayed with your own entries and exits on intraday bars, with
the rules changed:

* a stop placed with the first buy (``stop_pct`` below it, above it for a short), filled at the stop, or at
  the open when a bar gaps past it, less ``slippage_bps``;
* an optional profit target placed the same way;
* no adding to a losing position: a buy below the first buy (a sell above it, for a short) is skipped, and
  your later sells close at most what is still held;
* a fixed dollar amount per trade: every fill is scaled so the first buy is worth ``fixed_size``;
* a time limit: whatever is left is sold at the close of the ``max_days``-th trading day after the entry
  (0 = sold at the close of the entry day).

Your own fills keep your commission per share; new fills (stops, targets, time exits) pay IBKR Pro Tiered
costs. Without trade times in the export, each fill is placed in the first bar of its day (after the previous
fill) whose range contains its price. On the entry bar the stop only counts when the bar closes beyond it,
because the bar's low may have come before the buy; on every later bar the stop is checked before your own
fills in that bar, and before the target when both are inside one bar. Both choices are the cautious ones.
"""

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from analysis.opening_bar import order_costs


COLUMN_ALIASES = {
    "symbol": "Symbol", "underlyingsymbol": "Symbol",
    "tradedate": "TradeDate", "date": "TradeDate",
    "datetime": "DateTime", "date/time": "DateTime", "tradetime": "TradeTime",
    "buy/sell": "Buy/Sell", "side": "Buy/Sell",
    "quantity": "Quantity", "qty": "Quantity",
    "price": "Price", "tradeprice": "Price", "t. price": "Price",
    "currencyprimary": "Currency", "currency": "Currency",
    "commission": "Commission", "ibcommission": "Commission", "comm/fee": "Commission", "comm/fee ": "Commission",
    "ordertype": "OrderType", "tradeid": "TradeID", "orderid": "OrderID", "ibexecid": "TradeID",
    "assetclass": "AssetClass", "assetcategory": "AssetClass",
}
REQUIRED_COLUMNS = ["Symbol", "TradeDate", "Quantity", "Price"]
CASH_LIKE = ["SGOV", "BIL", "SHV", "USFR", "TFLO", "BOXX", "SHY", "TBIL"]
HOLD_BUCKETS = [(-1, 0, "Same day"), (0, 2, "1-2 days"), (2, 7, "3-7 days"), (7, 30, "8-30 days"),
                (30, 100_000, "Over 30 days")]
BREAKDOWNS = {
    "Holding time": "Held",
    "Exit order type": "Last exit type",
    "Symbol": "Symbol",
    "Month (of the exit)": "Exit month",
    "Long or short": "Side",
    "Added to a losing position": "Added to loser",
    "Number of buys": "Entry orders",
    "Position size": "Size bucket",
    "Weekday of the entry": "Entry weekday",
}
SIZE_BUCKETS = [0, 5_000, 10_000, 20_000, 40_000, np.inf]
TRIP_COLUMNS = ["Trip", "Symbol", "Side", "Entry date", "Exit date", "Days held", "Held", "Shares", "Avg entry",
                "Avg exit", "Cost $", "Gross P&L", "Fees", "Net P&L", "Return %", "Entry orders", "Exit orders",
                "Last exit type", "Exit types", "Added to loser", "Status"]


# ============================================================
# Import
# ============================================================

def _normalise_columns(frame):
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in COLUMN_ALIASES and COLUMN_ALIASES[key] not in renamed.values():
            renamed[column] = COLUMN_ALIASES[key]
    return frame.rename(columns=renamed)


def _parse_dates(values):
    """IBKR writes dates as 20260819, '2026-08-19' or Excel dates."""

    text = pd.Series(values).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text.where(compact), format="%Y%m%d", errors="coerce")
    other = pd.to_datetime(text.where(~compact), errors="coerce")
    return parsed.fillna(other)


def _parse_datetimes(values):
    """IBKR DateTime: '20260819;093512', '2026-08-19, 09:35:12' or '2026-08-19 09:35:12'."""

    text = pd.Series(values).astype(str).str.strip()
    compact = text.str.fullmatch(r"\d{8};\d{6}")
    parsed = pd.to_datetime(text.where(compact), format="%Y%m%d;%H%M%S", errors="coerce")
    other = pd.to_datetime(text.where(~compact).str.replace(",", "", regex=False), errors="coerce")
    return parsed.fillna(other)


def read_trade_file(source, name=""):
    """Read an IBKR export (path, file-like object or DataFrame) into a raw DataFrame."""

    if isinstance(source, pd.DataFrame):
        return source.copy()
    name = (name or str(getattr(source, "name", source))).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(source)
    return pd.read_csv(source)


def load_ibkr_trades(source, name="", currency="USD"):
    """Clean IBKR executions for one currency. Returns (trades, notes).

    ``trades`` has one row per execution, oldest first, with Quantity signed (+ buy, - sell), Cost (commission as
    a positive cost) and Time (NaT when the export has no time of day). ``notes`` counts what was dropped.
    """

    raw = _normalise_columns(read_trade_file(source, name))
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"The file has no {', '.join(missing)} column. Export the trades from an IBKR Flex Query "
                         "with Symbol, TradeDate, Buy/Sell, Quantity and Price.")

    frame = raw.copy()
    notes = {"rows": len(frame)}
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if "Currency" in frame.columns:
        other = frame["Currency"].astype(str).str.upper() != currency.upper()
        notes["other_currency"] = int(other.sum())
        notes["other_currencies"] = sorted(frame.loc[other, "Currency"].astype(str).unique())
        frame = frame.loc[~other]
    forex = frame["Symbol"].str.fullmatch(r"[A-Z]{3}\.[A-Z]{3}")
    if "AssetClass" in frame.columns:
        forex |= frame["AssetClass"].astype(str).str.upper().isin(["CASH", "FX", "FXCFD"])
    notes["forex"] = int(forex.sum())
    frame = frame.loc[~forex]

    frame["Date"] = _parse_dates(frame["TradeDate"])
    if "DateTime" in frame.columns:
        frame["Time"] = _parse_datetimes(frame["DateTime"])
    elif "TradeTime" in frame.columns:
        frame["Time"] = _parse_datetimes(frame["Date"].dt.strftime("%Y%m%d") + ";"
                                         + frame["TradeTime"].astype(str).str.replace(":", "").str.zfill(6))
    else:
        frame["Time"] = pd.NaT
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    if "Buy/Sell" in frame.columns:
        side = frame["Buy/Sell"].astype(str).str.upper().str.strip()
        sign = np.where(side.str.startswith("S"), -1.0, np.where(side.str.startswith("B"), 1.0, np.sign(frame["Quantity"])))
        frame["Quantity"] = frame["Quantity"].abs() * sign
    bad = frame["Date"].isna() | frame["Quantity"].isna() | frame["Price"].isna() | (frame["Quantity"] == 0)
    notes["unreadable"] = int(bad.sum())
    frame = frame.loc[~bad]

    commission = pd.to_numeric(frame["Commission"], errors="coerce").fillna(0.0) if "Commission" in frame else 0.0
    frame["Cost"] = -commission
    frame["OrderType"] = frame["OrderType"].astype(str).str.upper() if "OrderType" in frame else "N/A"
    frame["TradeID"] = frame["TradeID"] if "TradeID" in frame else np.arange(len(frame))
    frame["OrderID"] = frame["OrderID"] if "OrderID" in frame else frame["TradeID"]
    order = ["Date", "Time", "TradeID"] if frame["Time"].notna().all() else ["Date", "TradeID"]
    frame = frame.sort_values(order, kind="stable").reset_index(drop=True)
    notes["kept"] = len(frame)
    notes["has_times"] = bool(frame["Time"].notna().all()) and len(frame) > 0
    columns = ["Symbol", "Date", "Time", "Quantity", "Price", "Cost", "OrderType", "TradeID", "OrderID"]
    return frame[columns], notes


# ============================================================
# Round trips
# ============================================================

def split_fills(trades):
    """Assign every fill to a round trip; a fill that crosses zero is split in two. Adds Trip and Opening."""

    rows = []
    trades = trades.assign(Sequence=np.arange(len(trades)))
    for symbol, group in trades.groupby("Symbol", sort=False):
        position, trip_no = 0.0, 0
        for fill in group.itertuples(index=False):
            quantity = fill.Quantity
            pieces = []
            if position != 0 and np.sign(quantity) != np.sign(position) and abs(quantity) > abs(position) + 1e-9:
                pieces = [-position, quantity + position]       # close the old position, open the new one
            else:
                pieces = [quantity]
            for piece in pieces:
                if position == 0:
                    trip_no += 1
                row = fill._asdict()
                row["Quantity"] = piece
                row["Cost"] = fill.Cost * abs(piece) / abs(quantity)
                row["Opening"] = position == 0 or np.sign(piece) == np.sign(position)
                position = round(position + piece, 8)
                row["Position"] = position
                row["Trip"] = f"{symbol} #{trip_no}"
                rows.append(row)
    fills = pd.DataFrame(rows)
    if fills.empty:
        return fills
    return fills.sort_values("Sequence", kind="stable").drop(columns="Sequence").reset_index(drop=True)


def _worse_than_first(price, first_price, sign, tolerance=0.002):
    """A buy more than 0.2% below the first buy (a short sale above it): adding to a losing position."""

    return price < first_price * (1 - tolerance) if sign > 0 else price > first_price * (1 + tolerance)


def _hold_bucket(days):
    for low, high, label in HOLD_BUCKETS:
        if low < days <= high:
            return label
    return HOLD_BUCKETS[-1][2]


def round_trips(trades):
    """Return (fills, trips): fills with Trip/Opening/Position, and one row per round trip (see TRIP_COLUMNS)."""

    fills = split_fills(trades)
    if fills.empty:
        return fills, pd.DataFrame(columns=TRIP_COLUMNS)
    records = []
    for trip, group in fills.groupby("Trip", sort=False):
        opening = group[group["Opening"]]
        closing = group[~group["Opening"]]
        first = group.iloc[0]
        side = "Long" if first["Quantity"] > 0 else "Short"
        sign = 1.0 if side == "Long" else -1.0
        status = "Closed" if abs(group["Position"].iloc[-1]) < 1e-9 else "Open"
        entry_cost = float((opening["Quantity"].abs() * opening["Price"]).sum())
        shares_in = float(opening["Quantity"].abs().sum())
        shares_out = float(closing["Quantity"].abs().sum())
        gross = float(-(group["Quantity"] * group["Price"]).sum()) if status == "Closed" else np.nan
        fees = float(group["Cost"].sum())
        first_price = first["Price"]
        worse = _worse_than_first(opening["Price"], first_price, sign)
        entry_date, exit_date = group["Date"].iloc[0], group["Date"].iloc[-1]
        days = int(np.busday_count(entry_date.date(), exit_date.date()))
        exit_types = sorted(set(closing["OrderType"]))
        records.append({
            "Trip": trip, "Symbol": first["Symbol"], "Side": side, "Entry date": entry_date,
            "Exit date": exit_date if status == "Closed" else pd.NaT, "Days held": days if status == "Closed" else np.nan,
            "Held": _hold_bucket(days) if status == "Closed" else "Still open",
            "Shares": float(group["Position"].abs().max()), "Avg entry": entry_cost / shares_in,
            "Avg exit": float((closing["Quantity"].abs() * closing["Price"]).sum()) / shares_out if shares_out else np.nan,
            "Cost $": entry_cost, "Gross P&L": gross, "Fees": fees,
            "Net P&L": gross - fees if status == "Closed" else np.nan,
            "Return %": (gross - fees) / entry_cost * 100 if status == "Closed" and entry_cost else np.nan,
            "Entry orders": int(opening["OrderID"].nunique()), "Exit orders": int(closing["OrderID"].nunique()),
            "Last exit type": closing["OrderType"].iloc[-1] if len(closing) else "",
            "Exit types": "+".join(exit_types), "Added to loser": bool(worse.any()), "Status": status,
        })
    trips = pd.DataFrame(records)
    trips["Exit month"] = trips["Exit date"].dt.strftime("%Y-%m")
    trips["Entry weekday"] = pd.Categorical(trips["Entry date"].dt.day_name(),
                                            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    trips["Size bucket"] = pd.cut(trips["Cost $"], SIZE_BUCKETS,
                                  labels=["Under $5k", "$5-10k", "$10-20k", "$20-40k", "Over $40k"])
    trips["Held"] = pd.Categorical(trips["Held"], [label for _, _, label in HOLD_BUCKETS] + ["Still open"])
    trips["Entry orders"] = np.where(trips["Entry orders"] > 1, "Added (2+ buys)", "One buy")
    trips["Added to loser"] = np.where(trips["Added to loser"], "Yes", "No")
    return fills, trips


# ============================================================
# Statistics
# ============================================================

def summarize(trips, column="Net P&L"):
    """Headline numbers for closed round trips (``column`` picks actual or what-if P&L)."""

    closed = trips.dropna(subset=[column])
    pnl = closed[column]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = avg_win / -avg_loss if len(wins) and len(losses) and avg_loss < 0 else np.nan
    return {
        "Trades": int(len(pnl)),
        "Net P&L": float(pnl.sum()),
        "Win rate %": float((pnl > 0).mean() * 100) if len(pnl) else np.nan,
        "Avg win": float(avg_win) if pd.notna(avg_win) else np.nan,
        "Avg loss": float(avg_loss) if pd.notna(avg_loss) else np.nan,
        "Payoff": float(payoff) if pd.notna(payoff) else np.nan,
        "Breakeven win rate %": float(100 / (1 + payoff)) if pd.notna(payoff) else np.nan,
        "Profit factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
        "Per trade": float(pnl.mean()) if len(pnl) else np.nan,
        "Largest loss": float(pnl.min()) if len(pnl) else np.nan,
        "Fees": float(closed["Fees" if column == "Net P&L" else "What-if fees"].sum()),
    }


def breakdown(trips, by, column="Net P&L"):
    """Trades, P&L, win rate, average win/loss and profit factor per value of ``by`` (a trips column)."""

    closed = trips.dropna(subset=[column])
    rows = {}
    grouped = closed.groupby(by, observed=True, sort=True)
    for key, group in grouped:
        stats = summarize(group, column)
        rows[key] = {name: stats[name] for name in ["Trades", "Net P&L", "Win rate %", "Avg win", "Avg loss",
                                                    "Profit factor"]}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = by
    if by == "Symbol":
        table = table.sort_values("Net P&L")
    return table


def equity_curve(trips, column="Net P&L", date_column="Exit date"):
    closed = trips.dropna(subset=[column, date_column])
    return closed.groupby(date_column)[column].sum().sort_index().cumsum()


def apply_filters(trips, start=None, end=None, symbols=None, exclude=None, side="All"):
    keep = pd.Series(True, index=trips.index)
    if start is not None:
        keep &= trips["Entry date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= trips["Entry date"] <= pd.Timestamp(end)
    if symbols:
        keep &= trips["Symbol"].isin(symbols)
    if exclude:
        keep &= ~trips["Symbol"].isin(exclude)
    if side in ("Long", "Short"):
        keep &= trips["Side"] == side
    return trips.loc[keep]


# ============================================================
# What-if replay
# ============================================================

@dataclass
class WhatIfRules:
    stop_pct: float | None = None
    target_pct: float | None = None
    no_averaging_down: bool = False
    fixed_size: float | None = None
    max_days: int | None = None
    slippage_bps: float = 5.0
    commission_plan: str = "tiered"

    def active(self):
        return any([self.stop_pct, self.target_pct, self.no_averaging_down, self.fixed_size,
                    self.max_days is not None])

    def describe(self):
        parts = []
        if self.stop_pct:
            parts.append(f"stop {self.stop_pct:g}% from the first buy")
        if self.target_pct:
            parts.append(f"target +{self.target_pct:g}%")
        if self.no_averaging_down:
            parts.append("no adding to a losing position")
        if self.fixed_size:
            parts.append(f"${self.fixed_size:,.0f} per trade")
        if self.max_days is not None:
            parts.append("sold at the close of the entry day" if self.max_days == 0
                         else f"sold after at most {self.max_days} trading days")
        return ", ".join(parts) or "your actual trades"


@dataclass
class TripReplay:
    """Everything about one round trip that does not depend on the rules: its fills and the bars it spans."""

    sign: float
    first_price: float
    first_order_value: float
    quantity: np.ndarray
    price: np.ndarray
    cost_per_share: np.ndarray
    opening: np.ndarray
    located: list
    estimated: bool
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    bar_days: np.ndarray


def _locate_fills(fills, index, days, low, high):
    """Bar position of each fill: from its time when known, else the first bar of its day (after the previous
    fill) whose range contains the price. Returns (positions, estimated); positions is None when a fill's day has
    no bars."""

    positions, estimated = [], False
    previous = -1
    for fill_date, fill_time, fill_price in zip(fills["Date"].values, fills["Time"].values, fills["Price"].values):
        same_day = np.flatnonzero(days == fill_date)
        if len(same_day) == 0:
            return None, True
        if not pd.isna(fill_time):
            position = int(np.searchsorted(index, fill_time, side="right") - 1)
            position = min(max(position, same_day[0]), same_day[-1])
        else:
            estimated = True
            candidates = same_day[same_day >= previous]
            if len(candidates) == 0:
                candidates = same_day[-1:]
            tolerance = fill_price * 0.001
            inside = candidates[(low[candidates] - tolerance <= fill_price) & (fill_price <= high[candidates] + tolerance)]
            position = int(inside[0] if len(inside) else candidates[0])
        position = max(position, previous)
        positions.append(position)
        previous = position
    return positions, estimated


def fill_bar_positions(fills, bars):
    """Bar number (in ``bars``) of each fill, placed as the what-if replay places it; None without bars that day."""

    if bars is None or bars.empty:
        return None
    return _locate_fills(fills.reset_index(drop=True), bars.index.values, bars.index.normalize().values,
                         bars["Low"].to_numpy(float), bars["High"].to_numpy(float))[0]


def prepare_trip(fills, bars):
    """Build a TripReplay from one round trip's fills and its symbol's bars, or None without usable bars."""

    if bars is None or bars.empty:
        return None
    fills = fills.reset_index(drop=True)
    first_day, last_day = fills["Date"].iloc[0], fills["Date"].iloc[-1]
    index = bars.index.values
    lo = int(np.searchsorted(index, np.datetime64(first_day), side="left"))
    hi = int(np.searchsorted(index, np.datetime64(last_day + pd.Timedelta(days=1)), side="left"))
    window = bars.iloc[lo:hi]
    if window.empty:
        return None
    w_index = window.index.values
    days = window.index.normalize().values
    low, high = window["Low"].to_numpy(float), window["High"].to_numpy(float)
    located, estimated = _locate_fills(fills, w_index, days, low, high)
    if located is None:
        return None
    first = fills.iloc[0]
    same_order = fills["Opening"] & (fills["OrderID"] == first["OrderID"]) & (fills["Date"] == first["Date"])
    return TripReplay(
        sign=1.0 if first["Quantity"] > 0 else -1.0,
        first_price=float(first["Price"]),
        first_order_value=float((fills.loc[same_order, "Quantity"].abs() * fills.loc[same_order, "Price"]).sum()),
        quantity=fills["Quantity"].to_numpy(float),
        price=fills["Price"].to_numpy(float),
        cost_per_share=(fills["Cost"] / fills["Quantity"].abs()).to_numpy(float),
        opening=fills["Opening"].to_numpy(bool),
        located=located,
        estimated=estimated,
        open=window["Open"].to_numpy(float), high=high, low=low, close=window["Close"].to_numpy(float),
        bar_days=days,
    )


def replay_trip(trip, rules):
    """Replay a prepared round trip under ``rules``. Returns a dict of What-if fields."""

    sign, first_price = trip.sign, trip.first_price
    scale = rules.fixed_size / trip.first_order_value if rules.fixed_size and trip.first_order_value else 1.0
    stop = first_price * (1 - sign * rules.stop_pct / 100) if rules.stop_pct else None
    target = first_price * (1 + sign * rules.target_pct / 100) if rules.target_pct else None
    slip = rules.slippage_bps / 1e4
    located = trip.located
    entry_bar = located[0]
    time_exit_bar = None
    if rules.max_days is not None:
        trading_days = np.unique(trip.bar_days)
        day_no = int(np.searchsorted(trading_days, trip.bar_days[entry_bar])) + int(rules.max_days)
        if day_no < len(trading_days):  # otherwise your own exit comes first
            time_exit_bar = int(np.flatnonzero(trip.bar_days == trading_days[day_no])[-1])

    position = cash = fees = 0.0
    skipped = 0
    exit_reason, exit_bar, exit_price = "Your exit", None, np.nan

    def close_all(price, bar, reason, adverse):
        nonlocal position, cash, fees, exit_reason, exit_bar, exit_price
        fill_price = price * (1 - sign * slip) if adverse else price
        shares = abs(position)
        cash += sign * shares * fill_price
        fees += order_costs(shares, fill_price, sell=sign > 0, plan=rules.commission_plan)
        position = 0.0
        exit_reason, exit_bar, exit_price = reason, bar, fill_price

    n_fills = len(located)
    next_fill = 0
    for bar in range(entry_bar, located[-1] + 1):
        if bar > entry_bar and position != 0:
            open_ = trip.open[bar]
            if stop is not None and sign * (open_ - stop) <= 0:
                close_all(open_, bar, "Stop (gap)", adverse=True)
            elif stop is not None and (trip.low[bar] <= stop if sign > 0 else trip.high[bar] >= stop):
                close_all(stop, bar, "Stop", adverse=True)
            elif target is not None and sign * (open_ - target) >= 0:
                close_all(open_, bar, "Target", adverse=False)
            elif target is not None and (trip.high[bar] >= target if sign > 0 else trip.low[bar] <= target):
                close_all(target, bar, "Target", adverse=False)
            if exit_reason != "Your exit":
                break
        while next_fill < n_fills and located[next_fill] == bar:
            k = next_fill
            next_fill += 1
            quantity = trip.quantity[k] * scale
            if trip.opening[k]:
                if rules.no_averaging_down and position != 0 and _worse_than_first(trip.price[k], first_price, sign):
                    skipped += 1
                    continue
                position += quantity
                cash -= quantity * trip.price[k]
                fees += trip.cost_per_share[k] * abs(quantity)
            elif position != 0:
                shares = min(abs(quantity), abs(position))
                position -= sign * shares
                cash += sign * shares * trip.price[k]
                fees += trip.cost_per_share[k] * shares
                if abs(position) < 1e-9 * max(1.0, abs(quantity)):
                    position = 0.0
                    exit_bar, exit_price = bar, trip.price[k]
        if bar == entry_bar and position != 0 and stop is not None and sign * (trip.close[bar] - stop) <= 0:
            close_all(stop, bar, "Stop", adverse=True)  # the bar closed beyond the stop, so it crossed after the buy
            break
        if position == 0 and next_fill >= n_fills:
            break
        if time_exit_bar is not None and bar >= time_exit_bar and position != 0:
            close_all(trip.close[bar], bar, "Time limit", adverse=False)
            break
    if position != 0:  # shares left after your last fill (only possible after skipped buys): sold at that price
        close_all(trip.price[-1], located[-1], "Your exit", adverse=False)

    return {"What-if P&L": cash - fees, "What-if fees": fees, "What-if exit": exit_reason,
            "What-if exit date": pd.Timestamp(trip.bar_days[exit_bar]) if exit_bar is not None else pd.NaT,
            "What-if exit price": exit_price, "Skipped buys": skipped, "Times estimated": trip.estimated}


NO_DATA = {"What-if P&L": np.nan, "What-if fees": np.nan, "What-if exit": "No price data",
           "What-if exit date": pd.NaT, "What-if exit price": np.nan, "Skipped buys": 0, "Times estimated": True}


def simulate_trip(fills, bars, rules):
    """Replay one round trip's fills on ``bars`` (a DataFrame of OHLC bars) under ``rules``."""

    prepared = prepare_trip(fills, bars)
    return dict(NO_DATA) if prepared is None else replay_trip(prepared, rules)


def prepare(fills, trips, bars_by_symbol):
    """Prepare every closed round trip once; reuse the result for many rule sets (see ``simulate``).

    ``bars_by_symbol`` maps a symbol to intraday OHLC bars (exchange time, tz-naive); a symbol can instead map to
    a list of frames, finest first, and each trip uses the first frame that covers its entry day. Returns a dict
    {trip: TripReplay or None}; trips that are still open are left out.
    """

    fills_by_trip = {trip: group for trip, group in fills.groupby("Trip", sort=False)}
    prepared = {}
    for trip, symbol, status, entry_date in zip(trips["Trip"], trips["Symbol"], trips["Status"], trips["Entry date"]):
        if status != "Closed":
            continue
        frames = bars_by_symbol.get(symbol)
        frames = frames if isinstance(frames, (list, tuple)) else [frames]
        replay = None
        for frame in frames:
            if frame is not None and not frame.empty and frame.index[0].normalize() <= entry_date:
                replay = prepare_trip(fills_by_trip[trip], frame)
                if replay is not None:
                    break
        prepared[trip] = replay
    return prepared


def simulate(fills, trips, bars_by_symbol, rules, prepared=None):
    """Replay every closed round trip in ``trips``. Returns ``trips`` with What-if columns added.

    Pass ``prepared`` (from ``prepare``) to skip locating the fills again when trying many rule sets.
    """

    prepared = prepare(fills, trips, bars_by_symbol) if prepared is None else prepared
    results = []
    for trip, status in zip(trips["Trip"], trips["Status"]):
        if status != "Closed":
            results.append({**NO_DATA, "What-if exit": "Still open"})
        elif prepared.get(trip) is None:
            results.append(dict(NO_DATA))
        else:
            results.append(replay_trip(prepared[trip], rules))
    out = pd.concat([trips, pd.DataFrame(results, index=trips.index)], axis=1)
    out["What-if return %"] = out["What-if P&L"] / out["Cost $"] * 100
    out["Change"] = out["What-if P&L"] - out["Net P&L"]
    return out


def stop_sweep(fills, trips, bars_by_symbol, rules, stops=(1, 1.5, 2, 3, 4, 5, 7, 10), prepared=None):
    """Net what-if P&L for each stop distance, the other rules unchanged; the first row is no stop."""

    prepared = prepare(fills, trips, bars_by_symbol) if prepared is None else prepared
    rows = []
    for stop in [None, *stops]:
        tested = WhatIfRules(**{**rules.__dict__, "stop_pct": stop})
        result = simulate(fills, trips, bars_by_symbol, tested, prepared=prepared)
        valid = result.dropna(subset=["What-if P&L"])
        rows.append({"Stop %": "None" if stop is None else f"{stop:g}%", "Net P&L": valid["What-if P&L"].sum(),
                     "Win rate %": (valid["What-if P&L"] > 0).mean() * 100 if len(valid) else np.nan,
                     "Stopped %": valid["What-if exit"].str.startswith("Stop").mean() * 100 if len(valid) else np.nan})
    return pd.DataFrame(rows)
