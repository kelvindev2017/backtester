"""Monthly panel: features known the day before each month-end, excess returns after it."""
import pickle, glob, os, numpy as np, pandas as pd
H = {"3m": 63, "12m": 252, "36m": 756}
rows = []
for path in sorted(glob.glob("data/*.pkl")):
    t = os.path.basename(path)[:-4]
    d = pickle.load(open(path, "rb"))
    D, Q, E = d["daily"].copy(), d["quarterly"], d["earnings"]
    close = D["close"].astype(float); mkt = d["market_close"].reindex(D.index).ffill().astype(float)
    f = pd.DataFrame(index=D.index)
    mp = D["model_price"].astype(float).where(lambda s: s > 0)
    f["gap"] = np.log(close / mp)                                   # + = priced above fair value
    f["prem_dev"] = (D["implied_erp"] - D["usual_erp"]).astype(float)  # + = cheaper than usual premium
    yoy = D["held_yoy"].astype(float).clip(-50, 150)
    f["rev_growth"] = yoy
    f["growth_accel"] = yoy - yoy.shift(252)
    gm = D["held_gross_margin"].astype(float)
    f["margin_change"] = gm - gm.shift(252)
    ttm = Q["ttm_eps"].astype(float); ttm = ttm[~ttm.index.duplicated(keep="last")]
    eg = np.log(ttm / ttm.shift(4)).where((ttm > 0) & (ttm.shift(4) > 0)).clip(-1, 1) * 100
    f["eps_growth"] = eg.reindex(eg.index.union(D.index)).ffill().reindex(D.index)
    if "Surprise(%)" in E:
        s = E["Surprise(%)"].dropna().astype(float).clip(-50, 50); s.index = pd.DatetimeIndex(s.index).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index().rolling(2, min_periods=1).mean()
        f["surprise"] = s.reindex(s.index.union(D.index)).ffill().reindex(D.index)
    else:
        f["surprise"] = np.nan
    rel = np.log(close / mkt)
    f["momentum"] = (rel.shift(21) - rel.shift(252)) * 100
    f = f.shift(1)                                                # known the day before
    for name, n in H.items():
        f[f"ret_{name}"] = (rel.shift(-n) - rel) * 100              # excess log return vs S&P 500 (SPY), %
        f[f"gap_next_{name}"] = np.log(close / mp).shift(-n)
    f["close"] = close
    me = f.groupby(f.index.to_period("M")).tail(1)
    me = me[me.index < D.index[-1] - pd.Timedelta(days=3)]
    me = me.assign(ticker=t, month=me.index.to_period("M"))
    rows.append(me)
panel = pd.concat(rows)
panel = panel[np.isfinite(panel["gap"])]
panel.to_pickle("panel.pkl")
print(panel.shape, panel["ticker"].nunique(), panel["month"].min(), panel["month"].max())
print(panel.groupby(panel["month"].dt.year).size().to_string())
print(panel[["gap","prem_dev","rev_growth","growth_accel","margin_change","eps_growth","surprise","momentum"]].describe().round(2).to_string())
