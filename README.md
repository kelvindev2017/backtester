# backtester

Streamlit app for stock trend analysis, breakout identification and pre-trade reports.

## Tabs

**Backtest**: moving-average rule backtester with EPS, quarterly fundamentals (SEC + Yahoo),
P/E, benchmark, equity and drawdown panels.

**Signal Analysis & Trade Plan**: the analysis date is the end of the sidebar date range.

| Step | What it does | Module |
|---|---|---|
| Pattern / trend | Fresh close beyond the prior N-bar high/low (default 20), volume-confirmed (default 1.5x the 50-bar average). Daily and weekly trend from MA position/slope, swing structure (HH/HL) and ADX. | `analysis/patterns.py` |
| Support / resistance | Clustered swing pivots (more touches = stronger), volume-profile high-volume nodes, and the lookback high/low. | `analysis/levels.py` |
| Volume | Up/down-day volume ratio, OBV trend and divergence, Chaikin Money Flow. | `analysis/patterns.py` |
| Intraday entry | On the session after the daily signal: 5-min opening-range breakout beyond the daily trigger and VWAP, checked against the latest 15-min bar (VWAP, EMA 9/20). | `analysis/intraday.py` |
| Fundamentals | P/E vs its own history, PEG, TTM EPS growth, forward vs trailing P/E, revenue growth, margin trend, and whether the price move is earnings-driven or a re-rating. | `analysis/fundamentals.py` |
| Historical edge | How the same signal performed on this ticker before (entry at next open), versus the base rate. | `analysis/patterns.py` |
| Sizing | Stop 0.5 ATR beyond the nearest support (or the broken breakout level), target at the next zone; shares = equity x risk% / (entry - stop), capped at a max position %. Size tier by confidence: full >= 75, half >= 60, none below. | `analysis/risk.py` |
| Report | Action, confidence (0-100), reasoning, trade plan; download as Markdown/JSON or send to a Slack/Discord webhook. | `analysis/engine.py`, `analysis/notify.py` |

Confidence = 50 + 50 x weighted average of component scores in [-1, 1]
(signal quality 25%, trend alignment 20%, volume 15%, intraday 15%, fundamentals 15%,
historical edge 10%). Components that cannot be evaluated are dropped and the weights are re-normalised;
the report shows how much of the model was available.

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
