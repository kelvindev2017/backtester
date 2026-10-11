# backtester

Streamlit app for stock trend analysis, breakout identification and pre-trade reports.

## Tabs

**Backtest**: moving-average rule backtester with EPS, quarterly fundamentals (SEC + Yahoo),
P/E, benchmark, equity and drawdown panels.

**Signal Analysis & Trade Plan**: the analysis date is the end of the sidebar date range.

| Step | What it does | Module |
|---|---|---|
| Setup (sidebar) | **Breakout**: fresh close beyond the prior N-bar high/low (default 20), volume-confirmed (default 1.5x the 50-bar average). **Pullback in trend**: 50-bar MA above a rising 200-bar MA, a dip of at least 1.5 ATR to the 20-bar EMA on light volume, then a close above the prior bar's high (mirror image for downtrends). | `analysis/patterns.py` |
| Trend | Daily and weekly trend from MA position/slope, swing structure (HH/HL) and ADX. | `analysis/patterns.py` |
| Support / resistance | Clustered swing pivots (more touches = stronger), volume-profile high-volume nodes, and the lookback high/low. | `analysis/levels.py` |
| Volume | Up/down-day volume ratio, OBV trend and divergence, Chaikin Money Flow. | `analysis/patterns.py` |
| Intraday entry | On the session after the daily signal: 5-min opening-range breakout beyond the daily trigger and VWAP, checked against the latest 15-min bar (VWAP, EMA 9/20). | `analysis/intraday.py` |
| Fundamentals | P/E vs its own history, PEG, TTM EPS growth, forward vs trailing P/E, revenue growth, margin trend, and whether the price move is earnings-driven or a re-rating. Quarterly data is dated by its earnings release, so past analysis never sees unreleased results. | `analysis/fundamentals.py` |
| Historical edge | How the same signal performed on this ticker before (entry at next open), versus the base rate. | `analysis/patterns.py` |
| Sizing | Stop 0.5 ATR beyond the nearest support (or the setup's invalidation level), target at the next zone; shares = equity x risk% / (entry - stop), capped at a max position %. Size tier by confidence: full >= 75, half >= 60, none below. | `analysis/risk.py` |
| Report | Action, confidence (0-100), reasoning, trade plan; download as Markdown/JSON or send to a Slack/Discord webhook. | `analysis/engine.py`, `analysis/notify.py` |

**Long-only by default.** A bearish signal is reported as **EXIT** (close or reduce an existing long, no new
short). Tick "Allow short trades" in the sidebar to get short trade plans instead.

Confidence = 50 + 50 x weighted average of component scores in [-1, 1]
(signal quality 25%, trend alignment 20%, volume 15%, intraday 15%, fundamentals 15%,
historical edge 10%). Components that cannot be evaluated are dropped and the weights are re-normalised;
the report shows how much of the model was available. **The weights are a starting point; check them in the
Validate tab before relying on the score.**

**Validate**: replays the engine on every past day the selected setup fired, using only data available on
that day, and simulates the trade plan (entry at the next open, stop/target with gap handling, time exit).
Reports results in R (profit / planned risk) with a t-statistic, results per confidence bucket and threshold,
and whether each score component actually ranks outcomes better than a random score would. Use it to test
any sidebar setting across many tickers before trusting it (`analysis/validation.py`).

The tool never places orders. The report is the gate before you execute.

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
