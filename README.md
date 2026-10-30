# backtester

Streamlit app for stock trend analysis, breakout identification and pre-trade reports.

## Layout

Each tab keeps its own settings. On a desktop they fill the sidebar and change with the open tab; on a phone
or tablet (detected from the browser) there is no sidebar and the settings sit in collapsible menus above the
charts. **Display > Layout** overrides the detection. In the mobile layout a swipe over a chart scrolls the page
and a tap shows the values; **Display > Swiping on a chart** switches charts back to zoom and pan. The main US and industry indexes (S&P 500, Nasdaq,
Dow, Russell 2000, ^SOX, the sector ETFs ...) load when the app starts and are cached for 6 hours.

## Tabs

**Backtest**: moving-average rule backtester with EPS, quarterly fundamentals (SEC + Yahoo),
P/E, benchmark, equity and drawdown panels.

**Signal Analysis & Trade Plan**: the analysis date is the end of the date range in the settings.

| Step | What it does | Module |
|---|---|---|
| Signal rule (settings, shared by Backtest, Signal Analysis and Validate) | **Moving-average rule**: the Backtest tab's buy/sell rule on a fast and slow MA (EMA/SMA, default SMA 50 / SMA 200), firing on the first bar the condition becomes true. **Breakout**: fresh close beyond the prior N-bar high/low (default 20), volume-confirmed in Signal Analysis (default 1.5x the 50-bar average); the Backtest sells on a close below the exit SMA (default 200). **Pullback in trend**: trend fast MA (default 50) rising and above the trend slow MA (default 200), a dip of at least N ATR (default 1.5) from the swing high to the pullback EMA (default 20) on light volume, then a close above the prior bar's high (mirror image for downtrends). **Reversal**: in a long-term uptrend (same trend MAs), buy the first close below the prior N-bar low (default 20); the Backtest sells at the first close above the prior bar's high, or after N bars (default 10). **Day breakout**: price trades above the prior N-day high (default 10) and closes in the top third of the day's range (adjustable); a held position is sold at the first close below that, long only (the Backtest exits there). Every period is editable in the settings. The defaults were tuned on 60 large US stocks, 2015-2026, keeping only changes that improved results both before and after 2022. An illustration under the settings shows how the chosen rule enters and exits (`analysis/rule_diagrams.py`). | `analysis/patterns.py` |
| Trend | Daily and weekly trend from MA position/slope, swing structure (HH/HL) and ADX. | `analysis/patterns.py` |
| Support / resistance | Clustered swing pivots (more touches = stronger), volume-profile high-volume nodes, and the lookback high/low. | `analysis/levels.py` |
| Volume | Up/down-day volume ratio, OBV trend and divergence, Chaikin Money Flow. | `analysis/patterns.py` |
| Intraday entry | On the session after the daily signal: 5-min opening-range breakout beyond the daily trigger and VWAP, checked against the latest 15-min bar (VWAP, EMA 9/20). | `analysis/intraday.py` |
| Fundamentals | P/E vs its own history, PEG, TTM EPS growth, forward vs trailing P/E, revenue growth, margin trend, and whether the price move is earnings-driven or a re-rating. Quarterly data is dated by its earnings release, so past analysis never sees unreleased results. | `analysis/fundamentals.py` |
| Historical edge | How the same signal performed on this ticker before (entry at next open), versus the base rate. | `analysis/patterns.py` |
| Sizing | Stop 0.5 ATR beyond the nearest support (or the setup's invalidation level), target at the next zone; shares = equity x risk% / (entry - stop), capped at a max position %. Size tier by confidence: full >= 75, half >= 60, none below. | `analysis/risk.py` |
| Report | Action, confidence (0-100), reasoning, trade plan; download as Markdown/JSON or send to a Slack/Discord webhook. | `analysis/engine.py`, `analysis/notify.py` |

**Long-only by default.** A bearish signal is reported as **EXIT** (close or reduce an existing long, no new
short). Tick "Allow short trades" in the settings to get short trade plans instead.

Confidence = 50 + 50 x weighted average of component scores in [-1, 1]
(signal quality 25%, trend alignment 20%, volume 15%, intraday 15%, fundamentals 15%,
historical edge 10%). Components that cannot be evaluated are dropped and the weights are re-normalised;
the report shows how much of the model was available. **The weights are a starting point; check them in the
Validate tab before relying on the score.**

**Validate**: replays the engine on every past day the selected setup fired, using only data available on
that day, and simulates the trade plan (entry at the next open, stop/target with gap handling, time exit).
Reports results in R (profit / planned risk) with a t-statistic, results per confidence bucket and threshold,
and whether each score component actually ranks outcomes better than a random score would. Use it to test
any signal setting across many tickers before trusting it (`analysis/validation.py`).

**Trade Journal** (US-dollar trades): upload an IBKR Flex Query trade export (.xlsx or .csv with CurrencyPrimary, Symbol,
TradeID, OrderID, TradeDate, Buy/Sell, Quantity, Price, Commission, OrderType; Date/Time optional). Trades in other
currencies and currency conversions are left out. The fills of each stock are grouped into round trips (from the first
buy, or short sale, to the sale that leaves no shares; a fill through zero is split), with P&L after IBKR's fees. The
tab shows the headline numbers (win rate against the break-even win rate, average win against average loss, profit
factor), cumulative P&L, a breakdown by holding time, exit order type, stock, month, long or short, adding to a losing
position, number of buys, position size or weekday, and the trade list; select a trade to see its fills on its price
bars. Filters: entry dates, stocks, side (cash-like funds such as SGOV are left out by default).

**What-if rules** replay your own buys and sells on Yahoo's hourly bars (about two years) or 5-minute bars (about 60
days, hourly before that), with any of: a stop placed with the first buy (filled at the stop, or the open on a gap,
less a slippage), a profit target, no adding to a losing position (buys more than 0.2% below the first buy are skipped
and later sells close only what is held), the same dollar amount on every trade (the first order scaled to it), and a
time limit (sold at the close after N trading days). Your fills keep your commissions; new exits pay IBKR Pro Tiered.
The result is compared with your own on the same trades, with a chart of net P&L for stop distances from 1% to 10%.
Without trade times in the export, each fill is placed in the first bar of its day that traded at its price; stops are
checked before your own fills in a bar, and on the buy's own bar only count when the bar closes beyond the stop - both
the cautious choice. With no rule on, the replay gives back your actual P&L exactly (`analysis/trade_journal.py`).

**Day Breakout Rule**: the rule from the review of the 2026 IBKR trades, run mechanically on a list of tickers.
Long only: buy the first day a stock trades above its prior N-day high (default 10), filled at that level or at the
open on a gap up; at each close hold overnight only if the close is in the top third of the day's range (other
overnight rules: always sell at the close, or hold while the close is above an SMA/EMA of any period, default
SMA 20), with a time limit (default 60 days), and an optional SMA
trend filter. Fixed dollars per trade and flat fees per round trip. Results are split at a date into the period the
rule was fitted on and the check after it, with equity, monthly P&L, per-ticker results and the trade list
(`analysis/day_breakout.py`).

**First-bar day trade** (a signal rule in the Backtest tab): a day trade decided by the session's first intraday bar
(1, 5 or 15 minutes, chosen as the Timeframe). If that bar closes at least N points above its open (default 0, any
green bar) the day is a long day: buy at its close with the stop at its low, sized so a stop-out loses at most 10% of
the allocated capital (adjustable, and by default never more shares than the capital buys; the Account size is the
capital allocated to the stock). Other days the rule stays out, unless "Short on short days" is ticked: then a first
bar that closes at least N points below its open is shorted, the mirror image of the long trade. After ignoring the
next M bars (default 2; the stop still applies), sell everything on the chosen exit: **pullback** (a close below the
previous close sets the pullback; the next close below it sells, filled at that close or, as written, at the pullback
price) or **lower high** (the first bar whose high is below the previous bar's high sells at its close). Anything
still held is sold 5 minutes before the close. Costs are IBKR Pro commissions for US stocks (Tiered by default:
$0.0035 a share, min $0.35 an order, plus exchange, clearing and regulatory fees, about $1.30 a round trip on
$10,000) and a slippage per share (default 1 cent) on every fill. The Backtest shows its usual chart with the
day trades as signals. Yahoo keeps ~60 days of 5- and 15-minute bars and ~30 days of 1-minute bars; Signal Analysis
and Validate do not run this rule (`analysis/opening_bar.py`).

The first three tabs cover the market from the top down (`analysis/screener.py`, `analysis/signal_groups.py`,
`analysis/universe.py`):

**Stock Screener**: scans US stocks (S&P 500, Nasdaq-100, Dow Jones 30 or a custom list). Each condition row starts with a **signal group**, modelled on TradingView's
screener filters: Moving average, Index, Price, Chg %, Mkt cap, P/E, EPS dil growth, Div yield %, Sector,
Industry, Analyst rating, Perf %, Revenue growth, PEG, ROE, Beta, Recent / Upcoming earnings date. The inputs
after the group follow it (e.g. `Close Cross Above SMA 50`, `P/E between 0 and 30`, `Sector is any of Energy`,
`Earnings in the next 14 days`), and the results table adds a column for each group in use. Fundamentals come
from Yahoo's quote summary, fetched only when a condition needs them and kept for 12 hours. The settings can be
collapsed, and the screen can be limited to days when the Market Analysis filter passes. Results can be
downloaded as CSV; select a row to open that stock in Signal Analysis.

**Market Analysis** (US): trend of the main indexes (close vs short/medium/long SMAs, % from the 52-week high,
returns) and breadth of the S&P 500, Nasdaq-100 or Dow 30 (% of stocks above their medium/long SMA, % that set
a new 52-week high or low in the last N days), plus the VIX (latest close, level from calm to fear, 20-day
average, rank over the past year and a one-year chart). A market filter turns these into a PASS/FAIL verdict,
optionally requiring the VIX to be below a chosen level.

**Industry Analysis** (US): overlays the main index of each chosen industry (e.g. ^SOX for semiconductors, sector
SPDR ETFs such as XLK / XLF, ^NBI, ^BKX, ^DJT), optionally normalized to 100 at the start of the period (1M to
5Y), with a table of returns, trend and distance from the 52-week high. Optionally compares up to 5 leading
companies of one industry on price, MAs and quarterly revenue / net income trend.

**Stock Fundamentals** (any Yahoo ticker): five stacked panels on one date axis over 3 to 10 years. 1) Year-over-year
% change of quarterly revenue and EPS next to the daily price YoY %, with the rank correlation between growth and price
at several lags (price a quarter earlier, same quarter, a quarter later, a year later). 2) Quarterly revenue (SEC,
falling back to Yahoo) and 3) quarterly reported EPS, each with the next two quarters of analyst estimates as hatched
bars. 4) Trailing P/E with its median and 5) trailing PEG, with captions placing today's values in their own history
plus Yahoo's forward P/E and a forward PEG. Code in `analysis/valuation_chart.py`.

Below it, **What happened next**: at each check date (every day, week or month) it ranks the trailing P/E against every
trading day in the prior 3, 5 or 10 years (only EPS reported by then), and measures the total return over the next week
to 12 months minus SPY's. A scatter shows every check, with those near today's P/E percentile highlighted; a table
summarises them, counting how many checks do not overlap; and a walkthrough draws the close, trailing EPS, P/E window
and look-ahead return behind any one dot. Code in `analysis/what_happened_next.py`.

**Semicon Supply Chain** (US-listed companies only): a bottom-up graph of the semiconductor supply chain behind
an end market: **Mobile devices**, **AI / data center**, **Automotive**, or **All end markets** merged into one graph
(which shows where the chains meet, e.g. TSM and ASML). Columns run from the end market (and system makers such as
DELL, SMCI, ANET) through chips and components, memory, fabs and packaging, to tools, software and materials; a line
means the left company buys from the right one. Each company shows a node metric (3-month return by default, also
1m/6m/YTD/1y return or % from the 52-week high) as its label and color. Hover a company or a line for details; tap or
click a company (or pick it under the graph) to highlight everything upstream and downstream of it and open its
detail panel: who it buys from and sells to, its price over the last year against ^SOX, and quarterly revenue. The
chains and their links are curated by hand in `analysis/supply_chain.py` (`CHAINS`).

US index members come from Wikipedia (S&P 500 falls back to a public CSV).

The tool never places orders. The report is the gate before you execute.


## What validation found (October 2026)

Every number below comes from the Validate tab's walk-forward replay (signals use only data available on the
day, trades enter at the next open, 20-bar maximum hold, R measured against the planned risk). Each signal is
compared with five random entry days in the same stock and period that copy its stop and target distances,
so market drift and the exit rules are netted out.

Rules were fixed before testing. Results were checked on a design set (AAPL MSFT NVDA AMZN META GOOGL JPM XOM
KO COST UNH CAT), confirmed on a holdout set of 20 other large US stocks, and in two periods (2016-21, 2022-26).

**Long trades the engine would take - edge over matched random entries:**

| Setup | Stocks | 2016-21 | 2022-26 |
|---|---|---|---|
| Breakout | design | -0.03R (t -0.3) | -0.03R (t -0.3) |
| Breakout | holdout | -0.03R (t -0.3) | -0.26R (t -2.5) |
| Pullback | design | -0.03R (t -0.7) | -0.02R (t -0.4) |
| Pullback | holdout | -0.08R (t -2.7) | -0.10R (t -2.2) |

* **No setup beat random entries.** Raw results look positive (pullbacks earned +0.20R per trade in
  2016-21 on the design set), but random days with the same stops and targets did as well or better - the
  profit came from a rising market, not from the signal timing.
* **The confidence score and its components did not rank outcomes** consistently better than a random score.
  The weights were therefore not re-tuned: fitting them to this data would only fit noise.
* **Bearish signals were poor exit signals.** In all eight cells (2 setups x 2 stock sets x 2 periods) the stock
  rose *more* in the 10 bars after a bearish signal than after an average day.
* That pattern suggested a **reversal** rule (buy a 20-bar breakdown inside an uptrend). Tested afterwards on 20
  fresh stocks (QCOM IBM GE AMGN GS MS BLK SBUX MCD LOW ABT TMO DHR NEE LIN UPS MMM AXP SPGI ISRG), it was **not
  confirmed**: +0.11R (t 1.2) in 2016-21 and -0.10R (t -0.8) in 2022-26 versus random entries.

**What this means:** the data, levels, sizing, reports and validation work as designed, but none of the
built-in signal rules has a demonstrated edge on large US stocks. Use the Signal Analysis tab as a structured
checklist and risk calculator, and require any rule you add or tune to show a positive edge versus random
(|t| >= 2) on tickers and years you did not tune it on. Limits of this study: large US stocks only, daily bars
(the intraday trigger cannot be replayed beyond ~60 days of Yahoo data), no commissions or slippage.

## Secrets

Configure in Streamlit Cloud (App settings -> Secrets), or locally in `.streamlit/secrets.toml` (not committed):

```toml
password = "..."
SEC_USER_AGENT = "YourApp/1.0 you@example.com"
ALERT_WEBHOOK_URL = "https://hooks.slack.com/services/..."   # optional
```

## Data limitations (Yahoo Finance)

* 5-minute bars only go back ~60 days, so the intraday trigger can be evaluated live but not backtested over years.
* Point-in-time fields (forward P/E, Yahoo PEG) describe *today*; they are skipped when the analysis date is historical.
* "Reported EPS" from Yahoo is usually adjusted (non-GAAP) EPS, so P/E can differ from GAAP P/E.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
streamlit run App.py
```

`tests/test_app_smoke.py` runs the whole app headlessly with Yahoo and SEC replaced by synthetic data.
