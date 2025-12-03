import datetime as dt
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from state import SectorState

SECTOR_ETFS: Dict[str, str] = {
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
    "SPY": "S&P",
}

LEADERS: Dict[str, List[str]] = {
    "SMH": ["NVDA", "AMD", "AVGO", "TSM", "MU", "ASML", "AMAT", "SMCI"],
    "XLK": ["MSFT", "AAPL", "META", "GOOGL", "CRM", "ADBE", "NOW"],
    "XLF": ["JPM", "BAC", "MS", "GS", "C", "SCHW"],
    "XLE": ["XOM", "CVX", "SLB", "COP", "EOG"],
    "XBI": ["VRTX", "REGN", "LLY", "MRNA", "CRSP", "BEAM"],
    "XLV": ["UNH", "JNJ", "PFE", "ABBV", "DHR", "TMO"],
    "XLI": ["CAT", "DE", "HON", "GE", "BA"],
    "IWM": ["PLTR", "SMCI", "CELH", "RBLX", "RIVN", "AFRM"],
    "QQQ": ["NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL"],
    "SPY": ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL"],
}

LOOKBACK_DAYS = 220

# per-run cache so you don't spam Yahoo if graph calls this multiple times
_DOWNLOAD_CACHE: Dict[Tuple[Tuple[str, ...], int], pd.DataFrame] = {}


def true_range(high: np.ndarray, low: np.ndarray, close_prev: pd.Series) -> np.ndarray:
    return np.maximum.reduce([
        high - low,
        np.abs(high - close_prev),
        np.abs(low - close_prev),
    ])


def download_daily(tickers: List[str], period_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("No tickers to download")

    key = (tuple(tickers), period_days)
    if key in _DOWNLOAD_CACHE:
        return _DOWNLOAD_CACHE[key]

    end = dt.datetime.now(timezone.utc)
    start = end - dt.timedelta(days=period_days + 5)
    df = yf.download(
        tickers,
        start=start.date(),
        end=end.date(),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )
    _DOWNLOAD_CACHE[key] = df
    return df


def compute_metrics_for_ticker(df_t: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    d = df_t.dropna().copy()
    if d.empty:
        return None, None

    d["TR"] = true_range(
        d["High"].values,
        d["Low"].values,
        d["Close"].shift(1).fillna(d["Close"])
    )
    d["ATR20"] = d["TR"].rolling(20).mean()
    d["ATRpct"] = (d["ATR20"] / d["Close"]) * 100.0

    for n in [20, 50, 200]:
        d[f"SMA{n}"] = d["Close"].rolling(n).mean()

    d["RVOL20"] = d["Volume"] / d["Volume"].rolling(20).mean()
    d["Ret5"] = d["Close"].pct_change(5)

    latest = d.iloc[-1]

    out: Dict[str, Any] = {
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


def sector_breadth(leader_metrics: List[Dict[str, Any]]) -> float:
    flags = [m["above20"] for m in leader_metrics if m is not None]
    return np.nan if len(flags) == 0 else float(np.mean(flags))


def score_sector(metrics_etf: Optional[Dict[str, Any]], breadth: float, momo_cut: float = 0.0) -> float:
    if metrics_etf is None:
        return float("nan")

    rvol = min(metrics_etf["rvol20"], 2.0) / 2.0
    trend = (metrics_etf["above20"] + metrics_etf["above50"] + metrics_etf["above200"]) / 3.0
    br = 0.0 if np.isnan(breadth) else breadth
    momo = max(metrics_etf["ret5"], momo_cut)
    momo_norm = float(np.clip((momo + 0.05) / 0.10, 0.0, 1.0))

    score = 40 * rvol + 20 * trend + 25 * br + 15 * momo_norm
    return round(float(score), 1)


def pick_top_components(metrics_map: Dict[str, Optional[Dict[str, Any]]], n: int = 3):
    rows: List[Tuple[str, float, float, float, bool]] = []
    for t, m in metrics_map.items():
        if m is None:
            continue
        ok = (m["above20"] == 1.0) and (m["rvol20"] > 1.2) and (m["atrpct"] >= 2.0)
        rows.append((t, m["rvol20"], m["ret5"], m["atrpct"], ok))

    rows.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
    good = [r for r in rows if r[4]]
    return good[:n], rows[:n]


def build_sector_dashboard(selected_etfs: Optional[List[str]] = None) -> pd.DataFrame:
    # 1. pick sectors (only what parser asked for)
    if selected_etfs:
        etfs = [e for e in selected_etfs if e in SECTOR_ETFS]
    else:
        etfs = list(SECTOR_ETFS.keys())

    if not etfs:
        raise ValueError("No valid sector ETFs provided")

    # 2. collect tickers: only ETFs + leaders for those sectors
    tickers_all: List[str] = []
    for etf in etfs:
        tickers_all.append(etf)
        tickers_all.extend(LEADERS.get(etf, []))
    tickers_all = sorted(set(tickers_all))

    # 3. download only needed tickers
    df = download_daily(tickers_all, LOOKBACK_DAYS)

    # 4. split to per-ticker OHLCV
    history: Dict[str, pd.DataFrame] = {}
    for t in tickers_all:
        try:
            dft = df[t].dropna()
        except Exception:
            dft = pd.DataFrame()
        history[t] = dft

    # 5. compute metrics
    metrics: Dict[str, Optional[Dict[str, Any]]] = {}
    for t, dft in history.items():
        if dft.empty:
            metrics[t] = None
            continue
        try:
            m, _ = compute_metrics_for_ticker(dft)
            metrics[t] = m
        except Exception:
            metrics[t] = None

    # 6. build one row per sector ETF
    rows: List[Dict[str, Any]] = []

    for etf in etfs:
        name = SECTOR_ETFS[etf]
        m_etf = metrics.get(etf)

        leaders = LEADERS.get(etf, [])
        leader_metrics = [metrics.get(t) for t in leaders]
        br = sector_breadth([m for m in leader_metrics if m is not None])
        score = score_sector(m_etf, br)

        comp_metrics = {t: metrics.get(t) for t in leaders}
        best, _top_raw = pick_top_components(comp_metrics, n=3)

        rows.append({
            "ETF": etf,
            "Sector": name,
            "Score": score,
            "RVOL": None if m_etf is None else round(m_etf["rvol20"], 2),
            "Above 20/50/200": None if m_etf is None else f"{int(m_etf['above20'])}/{int(m_etf['above50'])}/{int(m_etf['above200'])}",
            "Breadth20": "-" if np.isnan(br) else round(br, 2),
            "5D%": None if m_etf is None or pd.isna(m_etf["ret5"]) else round(m_etf["ret5"] * 100, 2),
            "ATR%": None if m_etf is None else round(m_etf["atrpct"], 2),
            "TopCandidates": ", ".join([t for (t, *_rest) in best]) if len(best) > 0 else "",
        })

    table_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return table_df


async def fetch_data(state: SectorState) -> SectorState:
    print("fetch_data start")
    try:
        table_df = build_sector_dashboard(state.sectors)
        print(f"fetch_data done")
        return state.model_copy(update={
            "raw_rows": table_df.to_dict(orient="records"),
            "error": None,
        })
    except Exception as e:
        return state.model_copy(update={
            "error": str(e),
        })
