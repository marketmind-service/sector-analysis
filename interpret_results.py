from typing import Any, Dict, List, Optional, cast
from langchain_core.messages import SystemMessage, HumanMessage

from state import SectorState
from config import query2


def format_raw_rows(raw_rows: List[Dict[str, Any]]) -> str:
    print("format_raw_rows")
    lines: List[str] = []
    for row in raw_rows:
        etf = row.get("ETF")
        sector = row.get("Sector")
        score = row.get("Score")
        rvol = row.get("RVOL")
        breadth = row.get("Breadth20")
        ret5 = row.get("5D%")
        atr = row.get("ATR%")
        tops = row.get("TopCandidates")

        lines.append(
            f"{etf} | {sector} | Score={score} | RVOL={rvol} | Breadth20={breadth} | 5D%={ret5} | ATR%={atr} | Top={tops}"
        )
    return "\n".join(lines)


async def structure_results(state: SectorState) -> SectorState:
    print("structure_results")
    if not state.raw_rows:
        return state.model_copy(update={"error": "No raw_rows available for sector analysis"})

    table_text = format_raw_rows(state.raw_rows)

    system = SystemMessage(
        content=(
            "You are a quantitative sector rotation analyst. "
            "You receive a ranked list of sectors with scores and metrics, and must output a JSON summary.\n\n"
            "Input format (one per line):\n"
            "ETF | Sector | Score=<float or nan> | RVOL=<float or ''> | Breadth20=<float or '-' or ''> | "
            "5D%=<float or ''> | ATR%=<float or ''> | Top=<comma separated tickers or ''>\n\n"
            "Your job:\n"
            "1. Infer if the environment is risk-on, risk-off, or neutral.\n"
            "2. Identify the top 3 strongest sectors (by score and confirmation from RVOL/Breadth/5D%).\n"
            "3. Identify the bottom 3 weakest sectors.\n"
            "4. Mark which sectors look like short-term overextended, and which look like early basing/mean-reversion.\n"
            "5. Suggest a rotation bias: e.g. 'rotate from X into Y', 'stay defensive', 'focus on growth', etc.\n\n"
            "Output STRICT JSON with keys:\n"
            "{\n"
            "  \"risk_mode\": \"risk_on\" | \"risk_off\" | \"neutral\",\n"
            "  \"strong_sectors\": [ {\"etf\": \"XLK\", \"sector\": \"Tech\", \"reason\": \"...\" }, ... ],\n"
            "  \"weak_sectors\": [ {\"etf\": \"XLE\", \"sector\": \"Energy\", \"reason\": \"...\" }, ... ],\n"
            "  \"overextended\": [\"XLK\", \"SMH\", ...],\n"
            "  \"basing_or_reverting\": [\"XLF\", \"IWM\", ...],\n"
            "  \"rotation_view\": \"one short paragraph summary of where money is flowing\",\n"
            "  \"notes\": \"any subtle observations about breadth, leadership, or volatility\"\n"
            "}\n\n"
            "No explanations outside JSON. Do not add comments. Do not wrap in markdown."
        )
    )

    user = HumanMessage(
        content=(
            "User prompt:\n"
            f"{state.prompt or ''}\n\n"
            "Sector table:\n"
            f"{table_text}"
        )
    )

    resp = await query2.ainvoke([system, user])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    import json

    structured: Optional[Dict[str, Any]] = None
    try:
        # try to find JSON substring if model is a bit messy
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            structured = json.loads(raw[start:end + 1])
    except Exception:
        structured = None

    if structured is None:
        return state.model_copy(update={"error": "Failed to parse structured sector analysis JSON"})

    print("structured_results done")

    return state.model_copy(update={
        "structured_view": structured,
        "error": None,
    })


async def interpret_results(state: SectorState) -> SectorState:
    print("interpret_results")

    if not state.raw_rows:
        return state.model_copy(update={"error": "No raw_rows available for commentary"})
    if not state.structured_view:
        return state.model_copy(update={"error": "No structured_view available for commentary"})

    table_text = format_raw_rows(state.raw_rows)

    system = SystemMessage(
        content=(
            "You are a short, blunt, high-level sector strategist for an active swing trader. "
            "You receive: (1) a structured machine-readable view, (2) the raw sector metrics table, "
            "and (3) the user's original question.\n\n"
            "Write a concise analysis that covers:\n"
            "- Overall risk tone of the market (risk-on / risk-off / mixed).\n"
            "- Which sectors are leading and why (volume, breadth, momentum).\n"
            "- Which sectors are weakest and should be avoided or shorted.\n"
            "- Any obvious rotation ideas (e.g. 'trim X, rotate into Y').\n"
            "- Mention 2-5 specific leader tickers that are actionable examples.\n\n"
            "Style:\n"
            "- Focused, trader-friendly language.\n"
            "- Bullet points preferred.\n"
            "- No generic macro filler. Make it about the actual numbers.\n"
        )
    )

    user = HumanMessage(
        content=(
            f"User prompt:\n{state.prompt or ''}\n\n"
            f"Structured view (JSON):\n{state.structured_view}\n\n"
            f"Sector table:\n{table_text}\n"
        )
    )

    resp = await query2.ainvoke([system, user])
    interpreted_results = resp.content if isinstance(resp.content, str) else str(resp.content)

    print("interpret_results done")

    return state.model_copy(update={
        "interpreted_results": interpreted_results,
        "error": None,
    })
