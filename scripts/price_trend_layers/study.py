"""Walk-forward test of price-trend layers on the fair value model's history."""
import numpy as np, pandas as pd, sys
P = pd.read_pickle("panel.pkl")
P["year"] = P["month"].dt.year
MONTHS = {"3m": 3, "12m": 12, "36m": 36}
TRENDS = ["growth_accel", "eps_growth", "margin_change", "surprise"]
MODELS = {
    "Market (zero)": [],
    "Momentum only": ["momentum"],
    "Gap only (Layer A)": ["gap"],
    "Gap + trends": ["gap"] + TRENDS,
    "Gap + trends + momentum": ["gap"] + TRENDS + ["momentum"],
    "Trends only": TRENDS,
}
LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

def fit_predict(train, test, cols, y):
    if not cols:
        return np.zeros(len(test)), train[y].to_numpy()
    X = train[cols]; mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).clip(-3, 3).fillna(0).to_numpy(); Zt = ((test[cols] - mu) / sd).clip(-3, 3).fillna(0).to_numpy()
    Z1 = np.column_stack([np.ones(len(Z)), Z]); Zt1 = np.column_stack([np.ones(len(Zt)), Zt])
    pen = np.eye(Z1.shape[1]) * LAMBDA * len(Z) / 100; pen[0, 0] = 0
    yv = train[y].to_numpy(); beta = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ yv)
    beta[0] = 0  # no intercept: excess return vs the market is not forecast from the past average
    return Zt1 @ beta, yv - Z1 @ beta

def structural(train, test, h):
    # Layer A from first principles: the gap fades with a half-life measured on training data only.
    a = train[["gap", f"gap_next_{h}"]].dropna()
    b = np.clip(np.polyfit(a["gap"], a[f"gap_next_{h}"], 1)[0], 0, 1)
    return -(1 - b) * test["gap"].to_numpy() * 100, b

out, calib, halflife = [], [], []
for h, m in MONTHS.items():
    y = f"ret_{h}"
    for year in range(2019, 2027):
        cutoff = pd.Period(f"{year}-01", "M")
        train = P[(P["month"] + m <= cutoff) & P[y].notna()]
        test = P[(P["year"] == year) & P[y].notna()]
        if len(train) < 300 or len(test) < 30:
            continue
        for name, cols in MODELS.items():
            pred, resid = fit_predict(train, test, cols, y)
            lo, hi = np.percentile(resid, [10, 90])
            out.append(pd.DataFrame({"h": h, "model": name, "year": year, "month": test["month"].values,
                                     "ticker": test["ticker"].values, "pred": pred, "actual": test[y].values,
                                     "in_band": (test[y].values >= pred + lo) & (test[y].values <= pred + hi)}))
        pred, b = structural(train, test, h)
        halflife.append((h, year, b, m * np.log(0.5) / np.log(b) if 0 < b < 1 else np.inf))
        out.append(pd.DataFrame({"h": h, "model": "Gap fades (Layer A, no fit)", "year": year, "month": test["month"].values,
                                 "ticker": test["ticker"].values, "pred": pred, "actual": test[y].values, "in_band": np.nan}))
R = pd.concat(out)

def summary(g):
    ic = g.groupby("month").apply(lambda x: x["pred"].rank().corr(x["actual"].rank()) if x["pred"].std() > 0 else np.nan, include_groups=False)
    def spread(x):
        if x["pred"].std() == 0: return np.nan
        q = x["pred"].rank(pct=True); return x.loc[q > 0.8, "actual"].mean() - x.loc[q <= 0.2, "actual"].mean()
    ls = g.groupby("month").apply(spread, include_groups=False)
    r2 = 1 - ((g["actual"] - g["pred"]) ** 2).sum() / (g["actual"] ** 2).sum()
    hit = ((g["pred"] > 0) == (g["actual"] > 0)).mean() if g["pred"].std() > 0 else np.nan
    return pd.Series({"n": len(g), "years": f"{g['year'].min()}-{g['year'].max()}", "R2_oos_%": r2 * 100,
                      "rank_IC": ic.mean(), "IC>0_%": (ic > 0).mean() * 100, "top-bottom_%": ls.mean(),
                      "direction_%": hit * 100, "band80_%": g["in_band"].astype(float).mean() * 100})
pd.set_option("display.width", 250)
S = R.groupby(["h", "model"], sort=False).apply(summary, include_groups=False)
print(S.round(2).to_string())
print("\nRank IC by test year (Gap + trends + momentum / Gap only / Momentum only):")
for h in MONTHS:
    g = R[R.h == h]
    t = g.groupby(["model", "year"]).apply(lambda x: x.groupby("month").apply(lambda z: z["pred"].rank().corr(z["actual"].rank()), include_groups=False).mean(), include_groups=False).unstack()
    print(h); print(t.loc[["Gap only (Layer A)", "Gap + trends", "Gap + trends + momentum", "Momentum only"]].round(2).to_string())
print("\nGap persistence b (gap after h vs now) and implied half-life in months, by training cut-off:")
print(pd.DataFrame(halflife, columns=["h", "year", "b", "half_life_m"]).round(2).to_string(index=False))
R.to_pickle(f"results_{LAMBDA}.pkl"); S.to_csv(f"summary_{LAMBDA}.csv")
