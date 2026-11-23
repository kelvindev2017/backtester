# Price trend layers on top of the fair value model: first study (2026-11-23)

Question: can a layer on top of the Valuation Model's fair value (unchanged, app defaults) say where a stock's
price goes next, using the gap between the close and the modeled fair value plus the trends of growth, earnings
and gross margin? This is analysis, not a buy/sell instruction.

## Set-up

- 40 large US stocks (COST, WMT, KO, PEP, PG, JNJ, MCD, HD, NKE, SBUX, AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO,
  ADBE, CRM, ORCL, INTC, TXN, AMAT, KLAC, LRCX, TSLA, UNH, LLY, MRK, ABBV, CAT, DE, HON, NFLX, ISRG, TGT, LOW, QCOM,
  CVX, XOM), each loaded and modelled exactly as the app does, Nov 2014 to Oct 2026: 5,692 stock-months.
- Inputs at each month-end, known the day before:
  - gap = ln(close / modeled fair value)
  - premium vs usual (today's implied premium minus the 2-year median)
  - revenue growth per share and its change over a year
  - gross margin change
  - EPS growth
  - the last two earnings surprises
  - 12-1 month momentum vs the S&P 500
- Target: return above the S&P 500 (SPY) over the next 3, 12 and 36 months.
- Walk-forward: each test year is predicted by a model fitted only on months whose outcome was already known
  before that year (test years 2019-2026; 36 months: 2019-2023). The model is a ridge regression on standardised
  inputs, with no intercept, so the default forecast is "same as the market".
- Baselines: the market (forecast zero), momentum only, and the gap only.

## Point-in-time check

The model's history uses only what was known each day:
- Quarterly results are dated by their release.
- Analysts' estimates apply only from the latest release.
- The usual premium is the median of earlier months.
- Beta is rolling.

The stock-based pay haircut is today's figure applied to all history. It cancels out of the gap, because the
usual premium is solved with the same haircut. The check found two data bugs in old quarters, both fixed in the
same PR:
- Some filings state share counts in millions (KO 2018: 4,306). That made revenue growth per share read -100%,
  and KO's fair value history fell to 18-48 in 2020-22.
- KO's revenue before 2018 was missing (it used the tag SalesRevenueGoodsNet).

Two limits remain:
- **Survivorship.** The 40 stocks are today's large caps, chosen with hindsight.
- **Restated data.** Yahoo and the SEC give restated figures, not the figures as first published.

## Results

Out-of-sample, the core metric is rank IC: how well each month's forecast ranks the 40 stocks' later returns.
0 means no skill; 0.05 or more would be useful.

| Horizon | Market | Momentum | Gap only | Gap + trends | Gap + trends + momentum |
|---|---|---|---|---|---|
| 3 months: rank IC | - | +0.01 | -0.02 | -0.02 | -0.03 |
| 3 months: R² vs market | 0 | -0.1% | -0.3% | -0.8% | -0.8% |
| 12 months: rank IC | - | -0.06 | -0.04 | -0.03 | -0.05 |
| 12 months: R² vs market | 0 | -1.8% | -0.7% | -3.2% | -4.3% |
| 36 months: rank IC | - | -0.01 | -0.02 | -0.16 | -0.13 |
| 36 months: R² vs market | 0 | -4.6% | -0.7% | -21% | -23% |

No layer beat "same as the market" out of sample at any horizon. Adding the trends made the 3-year forecast
clearly worse. Their link to later returns flipped sign around 2020, so a fit on earlier years pointed the wrong
way afterwards.

**How the gap closes.** The gap does fade, with a half-life of about 7-8 months. It closes because the fair value
moves to the price, not because the price moves to the fair value. Regressed on today's gap, the change in fair
value over 12 months has a slope of +1.03, while the change in price has a slope of +0.03 (3 months: +0.21 vs +0.01;
36 months: +0.93 vs -0.05).

This is built in: the fair value uses the stock's own usual premium of the last 2 years, so a stock that stays
expensive slowly raises its own "usual". A layer that assumed the price fades back to the fair value was much worse
than the market (R² -119% at 3 months, -368% at 12 months).

**Single inputs, no fitting** (mean monthly rank IC with excess return, share of months positive):

| Input | 3 months | 12 months | 36 months | Before 2020 / from 2020 (36 months) |
|---|---|---|---|---|
| Revenue growth per share | +0.10 (65%) | +0.14 (70%) | +0.25 (80%) | +0.39 / +0.05 |
| Premium vs usual (cheap +) | -0.01 | +0.01 | +0.06 (69%) | +0.15 / -0.05 |
| Premium vs beta rule (cheap +) | -0.01 | +0.01 | +0.06 (64%) | |
| Premium vs 5-year median | -0.05 | -0.06 | -0.07 | |
| Growth acceleration | +0.01 | -0.00 | -0.02 | |
| Gross margin change | -0.01 | -0.01 | +0.03 | |
| EPS growth | -0.00 | -0.03 | +0.04 | +0.14 / -0.07 |
| Earnings surprise | +0.03 | -0.01 | +0.04 | +0.17 / -0.14 |
| Momentum | +0.02 | +0.02 | +0.08 (66%) | +0.12 / +0.04 |

- **Revenue growth is the only steady input.** Its strength mostly comes from survivorship: the 40 are today's
  winners, so high growth in 2015 shows up as high returns afterwards. It fades from 2020 on.
- **The 3-year valuation tilt is small.** Cheapest fifth by premium vs usual minus richest fifth: +6.8 points of
  excess return over 3 years, about 2 a year. It was positive before 2020 and not since.

**Ranges.** The 80% ranges, taken from training errors, held only 63-69% of the time. Honest ranges need to be
about a third wider than past errors suggest, because the market's behaviour shifts between periods.

## What this means for the layers

1. **Read the modeled fair value as "price at the stock's usual premium", not a target.** The data shows it moves
   toward the price, not the other way round.
2. **There is no measurable 3- to 12-month price trend from the gap or the fundamental trends on these 40 stocks.**
   A price-path forecast built from them would present noise as signal.
3. **What can be shown honestly:**
   - a range for 3/12/36 months from the model's own past errors, widened as above
   - the 3-year valuation tilt as a small expected-return adjustment (about 2 points a year), flagged as unstable
4. **A sounder test needs data we lack today:**
   - a survivorship-free universe (index members as of each date, including delisted stocks)
   - several hundred stocks
   - history of analysts' estimate revisions (Yahoo has only today's)

   With 40 stocks, one month's rank IC has a noise of about ±0.16, so only a strong, steady effect could show up.
