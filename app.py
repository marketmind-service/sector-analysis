# app.py
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

import datetime as dt
from datetime import timezone
import numpy as np
import pandas as pd

# import your module
import sector_analysis as sa


app = FastAPI(title="Sector Analysis Dashboard API")


def compute_sector_dashboard(
    lookback_days: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Runs the same logic as your CLI main, but returns data instead of printing.
    Returns:
      table_df: summary table per sector
      candidates_by_sector: dict[etf] -> list of candidate dicts
    """

    tickers_all = list(sa.SECTOR_ETFS.keys())
    for leaders in sa.LEADERS.values():
        tickers_all += leaders
    tickers_all = sorted(set(tickers_all))

    df = sa.download_daily(tickers_all, period_days=lookback_days)

    # split by ticker
    history: Dict[str, pd.DataFrame] = {}
    for t in tickers_all:
        try:
            dft = df[t].dropna()
        except Exception:
            dft = pd.DataFrame()
        history[t] = dft

    metrics: Dict[str, Any] = {}
    full_hist: Dict[str, Any] = {}
    for t, dft in history.items():
        if dft.empty:
            metrics[t] = None
            full_hist[t] = None
            continue
        try:
            m, dh = sa.compute_metrics_for_ticker(dft)
            metrics[t] = m
            full_hist[t] = dh
        except Exception:
            metrics[t] = None
            full_hist[t] = None

    rows: List[Dict[str, Any]] = []
    candidates_by_sector: Dict[str, List[Dict[str, Any]]] = {}

    for etf, name in sa.SECTOR_ETFS.items():
        m_etf = metrics.get(etf)
        leaders = sa.LEADERS.get(etf, [])
        leader_metrics = [metrics.get(t) for t in leaders]
        br = sa.sector_breadth([m for m in leader_metrics if m is not None])
        score = sa.score_sector(m_etf, br)

        comp_metrics = {t: metrics.get(t) for t in leaders}
        best, _top_raw = sa.pick_top_components(comp_metrics, n=3)

        rows.append(
            {
                "etf": etf,
                "sector": name,
                "score": score,
                "rvol20": None if m_etf is None else round(m_etf["rvol20"], 2),
                "above_20_50_200": None
                if m_etf is None
                else f"{int(m_etf['above20'])}/{int(m_etf['above50'])}/{int(m_etf['above200'])}",
                "breadth_20": None if np.isnan(br) else round(br, 2),
                "ret5_pct": None
                if m_etf is None or pd.isna(m_etf["ret5"])
                else round(m_etf["ret5"] * 100.0, 2),
                "atr_pct": None if m_etf is None else round(m_etf["atrpct"], 2),
                "top_candidates": [t for (t, *_rest) in best] if len(best) > 0 else [],
            }
        )

        cand_list: List[Dict[str, Any]] = []
        for (t, rvol, ret5, atrpct, ok) in best:
            cand_list.append(
                {
                    "ticker": t,
                    "rvol20": round(float(rvol), 2),
                    "ret5_pct": round(float(ret5) * 100.0, 2),
                    "atr_pct": round(float(atrpct), 2),
                    "swing_ok": bool(ok),
                }
            )
        candidates_by_sector[etf] = cand_list

    table_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return table_df, candidates_by_sector


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sectors")
def sectors_api(
    lookback_days: int = Query(
        sa.LOOKBACK_DAYS,
        ge=20,
        le=365,
        description="Number of calendar days to look back for daily data",
    )
):
    """
    Returns the sector rotation dashboard as JSON.
    Example:
      /api/sectors
      /api/sectors?lookback_days=120
    """

    try:
        table_df, candidates_by_sector = compute_sector_dashboard(lookback_days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to compute dashboard: {e}")

    now_utc = dt.datetime.now(timezone.utc).isoformat()

    sectors = table_df.to_dict(orient="records")

    return JSONResponse(
        {
            "as_of_utc": now_utc,
            "lookback_days": lookback_days,
            "sectors": sectors,
            "candidates_by_sector": candidates_by_sector,
        }
    )