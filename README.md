# backtester

Streamlit app for stock trend analysis, breakout identification and pre-trade reports.

## Layout

Each tab keeps its own settings. On a desktop they fill the sidebar and change with the open tab; on a phone
or tablet (detected from the browser) there is no sidebar and the settings sit in collapsible menus above the
charts. **Display > Layout** overrides the detection. The main US and industry indexes (S&P 500, Nasdaq,
Dow, Russell 2000, ^SOX, the sector ETFs ...) load when the app starts and are cached for 6 hours.

## Tabs

**Backtest**: moving-average rule backtester with EPS, quarterly fundamentals (SEC + Yahoo),
P/E, benchmark, equity and drawdown panels.

**Signal Analysis & Trade Plan**: the analysis date is the end of the date range in the settings.

| Step | What it does | Module |
|---|---|---|
| Signal rule (settings, shared by Backtest, Signal Analysis and Validate) | **Moving-average rule**: the Backtest tab's buy/sell rule on a fast and slow MA (EMA/SMA), firing on the first bar the condition becomes true. **Breakout**: fresh close beyond the prior N-bar high/low (default 20), volume-confirmed (default 1.5x the 50-bar average). **Pullback in trend**: trend fast MA (default 50) rising and above the trend slow MA (default 200), a dip of at least N ATR (default 1.5) from the swing high to the pullback EMA (default 20) on light volume, then a close above the prior bar's high (mirror image for downtrends). **Reversal**: in a long-term uptrend (same trend MAs), buy the first close below the prior N-bar low (default 20). Every period is editable in the settings. | `analysis/patterns.py` |
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
a new 52-week high or low in the last N days). A market filter turns these into a PASS/FAIL verdict.

**Industry Analysis** (US): overlays the main index of each chosen industry (e.g. ^SOX for semiconductors, sector
SPDR ETFs such as XLK / XLF, ^NBI, ^BKX, ^DJT), optionally normalized to 100 at the start of the period (1M to
5Y), with a table of returns, trend and distance from the 52-week high. Optionally compares up to 5 leading
companies of one industry on price, MAs and quarterly revenue / net income trend.

**Semicon Supply Chain** (US-listed companies only): a bottom-up graph of the semiconductor supply chain behind
an end market: **Mobile devices**, **AI / data center**, **Automotive**, or **All end markets** merged into one graph
(which shows where the chains meet, e.g. TSM and ASML). Columns run from the end market (and system makers such as
DELL, SMCI, ANET) through chips and components, memory, fabs and packaging, to tools, software and materials; a line
means the left company buys from the right one. Suppliers are boxed by type within each column (e.g. Equipment,
Design software & IP, Materials; Foundry, Memory, Packaging & test). Each company shows a node metric (3-month return by default, also
1m/6m/YTD/1y return or % from the 52-week high) as its label and color. Hover a company or a line for details; tap or
click a company (or pick it under the graph) to highlight everything upstream and downstream of it and open its
detail panel: who it buys from and sells to, its price over the last year against its group's index and ^SOX, and
quarterly revenue. Under **Edit groups** you set each company's group yourself: type a name to create or join a
group, clear it to leave the company ungrouped, rename a group, or reset to the defaults. Every group gets an
equal-weight index (members rebased to 100 on their first common day, then averaged); the box label shows the
index's node metric, and **Group indexes** charts the last year with each group's returns and trend. Custom groups
last for the browser session. The chains and their links are curated by hand in `analysis/supply_chain.py` (`CHAINS`).

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
