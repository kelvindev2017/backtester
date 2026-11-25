# Logs

Written daily by .github/workflows/log_guidance.yml (scripts/log_guidance.py).

- guidance_numbers.csv: company guidance figures read from 8-K results releases (2012 onward for S&P 500 + ANT + SPCX; loaded once on 2026-11-25), then each day's new releases.
- guidance_read.csv: tickers whose history is loaded (the app reads these from the log).
- guidance_days.csv: days whose SEC filing list the daily run has read.
- estimates/: analysts' estimates, one file a day.
