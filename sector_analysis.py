import datetime as dt
from datetime import timezone
import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

# config
SECTOR_ETFS = {
    "SMH": "Semiconductors",
    "XLK": "Tech",
    "XLC": "Comm Services",
    "XLF": "Financials",
    "XLE": "Energy",
    "XBI": "Biotech",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "IWM": "Small Caps",
    "QQQ": "Nasdaq",
    "SPY": "S&P"
}

# leaders per sector (liquid, high-beta-ish)
LEADERS = {
    "SMH": ["NVDA", "AMD", "AVGO", "TSM", "MU", "ASML", "AMAT", "SMCI"],
    "XLK": ["MSFT", "AAPL", "META", "GOOGL", "CRM", "ADBE", "NOW"],
    "XLF": ["JPM", "BAC", "MS", "GS", "C", "SCHW"],
    "XLE": ["XOM", "CVX", "SLB", "COP", "EOG"],
    "XBI": ["VRTX", "REGN", "LLY", "MRNA", "CRSP", "BEAM"],
    "XLV": ["UNH", "JNJ", "PFE", "ABBV", "DHR", "TMO"],
    "XLI": ["CAT", "DE", "HON", "GE", "BA"],
    "IWM": ["PLTR", "SMCI", "CELH", "RBLX", "RIVN", "AFRM"],  # not rlly IWM leaders, but popular enough
    "QQQ": ["NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL"],
    "SPY": ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL"]
}

LOOKBACK_DAYS = 180
INTRADAY = False  # set True for partial-day RVOL est.
TIMEZONE = "America/Toronto"


# helpers
def true_range(high, low, close_prev):
    return np.maximum.reduce([high - low, np.abs(high - close_prev), np.abs(low - close_prev)])


def download_daily(tickers, period_days=LOOKBACK_DAYS):
    end = dt.datetime.now(timezone.utc)
    start = end - dt.timedelta(days=period_days + 10)  # padding for weekends/holidays
    df = yf.download(tickers, start=start.date(), end=end.date(), group_by="ticker", auto_adjust=False, progress=False)
    return df


def compute_metrics_for_ticker(df_t):
    # df_t: columns ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    d = df_t.dropna().copy()
    if d.empty:
        return None
    d["TR"] = true_range(d["High"].values, d["Low"].values, d["Close"].shift(1).fillna(d["Close"]))
    d["ATR20"] = d["TR"].rolling(20).mean()
    d["ATRpct"] = (d["ATR20"] / d["Close"]) * 100.0
    for n in [10, 20, 50, 200]:
        d[f"SMA{n}"] = d["Close"].rolling(n).mean()
    d["RVOL20"] = d["Volume"] / d["Volume"].rolling(20).mean()
    d["Ret5"] = d["Close"].pct_change(5)
    latest = d.iloc[-1]
    out = {
        "close": latest["Close"],
        "volume": latest["Volume"],
        "rvol20": latest["RVOL20"],
        "atrpct": latest["ATRpct"],
        "a20": latest["SMA20"],
        "a50": latest["SMA50"],
        "a200": latest["SMA200"],
        "ret5": latest["Ret5"],
        "above20": float(latest["Close"] > latest["SMA20"]),
        "above50": float(latest["Close"] > latest["SMA50"]),
        "above200": float(latest["Close"] > latest["SMA200"]),
    }
    return out, d


def sector_breadth(leader_metrics):
    # percentage of leaders above their 20DMA
    flags = [m["above20"] for m in leader_metrics if m is not None]
    return np.nan if len(flags) == 0 else float(np.mean(flags))


def score_sector(metrics_etf, breadth, momo_cut=0.0):
    # score 0-100
    # components:
    #  - RVOL contribution (cap at 2.0) -> 40
    #  - trend stack (above 20/50/200) -> 20
    #  - breadth (leaders above 20-day avg) -> 25
    #  - momentum (5-day return) -> 15

    if metrics_etf is None:
        return np.nan

    rvol = min(metrics_etf["rvol20"], 2.0) / 2.0  # 0..1
    trend = (metrics_etf["above20"] + metrics_etf["above50"] + metrics_etf["above200"]) / 3.0  # 0..1
    br = 0.0 if np.isnan(breadth) else breadth  # 0..1
    momo = max(metrics_etf["ret5"], momo_cut)  # allow negative capped at "momo_cut"
    # normalize momo roughly: assume -5%..+5% -> 0..1 scale
    momo_norm = np.clip((momo + 0.05) / 0.10, 0, 1)
    score = 40 * rvol + 20 * trend + 25 * br + 15 * momo_norm
    return round(float(score), 1)


def pick_top_components(metrics_map, n=3):
    # choose swingy leaders: above 20DMA, rvol>1.2, positive 5d return, atrpct>=2
    rows = []
    for t, m in metrics_map.items():
        if m is None:
            continue
        ok = (m["above20"] == 1.0) and (m["rvol20"] > 1.2) and (m["atrpct"] >= 2.0)
        rows.append((t, m["rvol20"], m["ret5"], m["atrpct"], ok))
    # sort by rvol then ret5
    rows.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
    good = [r for r in rows if r[4]]
    return good[:n], rows[:n]


def main():
    print("Downloading daily OHLCV...")
    tickers_all = list(SECTOR_ETFS.keys())
    for leaders in LEADERS.values():
        tickers_all += leaders
    tickers_all = sorted(set(tickers_all))

    df = download_daily(tickers_all, LOOKBACK_DAYS)
    # split back into dict per ticker
    history = {}
    for t in tickers_all:
        try:
            dft = df[t].dropna()
        except Exception:
            dft = pd.DataFrame()
        history[t] = dft

    # compute metrics
    metrics = {}
    full_hist = {}
    for t, dft in history.items():
        if dft.empty:
            metrics[t] = None
            full_hist[t] = None
            continue
        try:
            m, dh = compute_metrics_for_ticker(dft)
            metrics[t] = m
            full_hist[t] = dh
        except Exception:
            metrics[t] = None
            full_hist[t] = None

    # sector table
    rows = []
    sector_components_table = {}

    for etf, name in SECTOR_ETFS.items():
        m_etf = metrics.get(etf)
        leaders = LEADERS.get(etf, [])
        leader_metrics = [metrics.get(t) for t in leaders]
        br = sector_breadth([m for m in leader_metrics if m is not None])
        score = score_sector(m_etf, br)
        # pick top components
        comp_metrics = {t: metrics.get(t) for t in leaders}
        best, top3_raw = pick_top_components(comp_metrics, n=3)
        rows.append({
            "ETF": etf,
            "Sector": name,
            "Score": score,
            "RVOL": None if m_etf is None else round(m_etf["rvol20"], 2),
            "Above 20/50/200": None if m_etf is None else f"{int(m_etf['above20'])}/{int(m_etf['above50'])}/{int(m_etf['above200'])}",
            "Breadth 20": "-" if np.isnan(br) else round(br, 2),
            "5D%": None if m_etf is None or pd.isna(m_etf["ret5"]) else round(m_etf["ret5"] * 100, 2),
            "ATR%": None if m_etf is None else round(m_etf["atrpct"], 2),
            "Top Candidates": ", ".join([f"{t}" for (t, *_rest) in best]) if len(best) > 0 else ""
        })
        sector_components_table[etf] = best

    # rank by score
    table_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    print("\n=== Sector Early‑Warning Dash ===\n")
    print(tabulate(table_df.fillna(""), headers="keys", tablefmt="github", showindex=False))

    # write csv
    #ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    #table_df.to_csv(f"sector_rotation_dashboard_{ts}.csv", index=False)

    # detailed components per sector file
    '''comp_rows = []
    for etf, best in sector_components_table.items():
        for (t, rvol, ret5, atrpct, ok) in best:
            comp_rows.append({
                "ETF": etf, "Ticker": t, "RVOL": round(rvol, 2), "5D%": round(ret5 * 100, 2), "ATR%": round(atrpct, 2),
                "SwingOK": ok
            })
    if comp_rows:
        comp_df = pd.DataFrame(comp_rows).sort_values(["ETF", "SwingOK", "RVOL"], ascending=[True, False, False])
        comp_df.to_csv(f"sector_rotation_candidates_{ts}.csv", index=False)
        print("\nSaved: sector_rotation_dashboard_{ts}.csv and sector_rotation_candidates_{ts}.csv")
    else:
        print("\nNo component candidates met the strict filter; relax thresholds or check again tomorrow.")

    print("\nRules:")
    print(" - Focus on sectors with Score >= 60 and ETF RVOL >= 1.3 on consecutive days")
    print(" - Prefer components above 20DMA with RVOL > 1.2 and ATR% >= 2")
    print(" - If everything is sub-50 score, it's a chop regime. Reduce risk, wait for clearer rotation, and don't be a dumbass")'''


if __name__ == "__main__":
    main()
