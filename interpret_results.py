from typing import Any, Dict, List, Optional, cast
from langchain_core.messages import SystemMessage, HumanMessage

from state import SectorState
from config import query2


def classify_style(row: Dict[str, Any]) -> str:
    """Classify sector behavior style based on RVOL, Breadth, 5D% and ATR%."""
    rvol = row.get("RVOL") or 0.0
    br = row.get("Breadth20")
    br_val = br if isinstance(br, (int, float)) else 0.0
    ret5 = row.get("5D%") or 0.0
    atr = row.get("ATR%") or 0.0

    # Durable leadership: broad participation, normal-ish volume, controlled vol
    if rvol < 1.2 and br_val > 0.6 and atr < 2.5:
        return "durable"

    # Momentum thrust: big short term move and higher vol
    if ret5 > 3.0 and atr > 2.5:
        return "momentum"

    # Volatile leadership: elevated volume or wild ATR
    if rvol > 1.5 or atr > 2.8:
        return "volatile"

    return "neutral"


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
        style = classify_style(row)

        lines.append(
            f"{etf} | {sector} | Score={score} | Style={style} | "
            f"RVOL={rvol} | Breadth20={breadth} | 5D%={ret5} | ATR%={atr} | Top={tops}"
        )
    return "\n".join(lines)


def normalize_small_universe(structured: Dict[str, Any]) -> Dict[str, Any]:
    """
    For small universes (1–2 sectors), avoid silly overlaps like
    the same sector being both strong and weak. Keep only the top strong
    and drop any weak that collide.
    """
    strong = structured.get("strong_sectors") or []
    weak = structured.get("weak_sectors") or []

    # Only keep the top strong in tiny universes
    if len(strong) > 1:
        strong = strong[:1]

    strong_etfs = {s.get("etf") for s in strong if isinstance(s, dict)}
    weak = [w for w in weak if isinstance(w, dict) and w.get("etf") not in strong_etfs]

    structured["strong_sectors"] = strong
    structured["weak_sectors"] = weak
    return structured


def scrub_basing(structured: Dict[str, Any], raw_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Remove clearly bogus 'basing' tags from sectors with trash breadth.
    Example rule: if Breadth20 < 0.25, do not call it basing/reverting.
    """
    basing = structured.get("basing_or_reverting") or []
    if not isinstance(basing, list):
        return structured

    raw_map: Dict[str, Dict[str, Any]] = {}
    for r in raw_rows:
        etf = r.get("ETF")
        if isinstance(etf, str):
            raw_map[etf] = r

    filtered: List[str] = []
    removed: List[str] = []

    for etf in basing:
        if not isinstance(etf, str):
            continue
        row = raw_map.get(etf, {})
        br = row.get("Breadth20")
        if isinstance(br, (int, float)):
            br_val = br
        else:
            # treat missing or '-' breadth as garbage, not neutral
            br_val = 0.0
        if br_val < 0.25:
            removed.append(etf)
        else:
            filtered.append(etf)

    structured["basing_or_reverting"] = filtered

    # Append note so you can see which ones were scrubbed
    notes = structured.get("notes") or ""
    if removed:
        extra = f"Removed low-breadth basing flags for: {', '.join(removed)}."
        notes = f"{notes} {extra}".strip()
    structured["notes"] = notes

    return structured


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
            "ETF | Sector | Score=<float or nan> | Style=<durable|momentum|volatile|neutral> | "
            "RVOL=<float or ''> | Breadth20=<float or '-' or ''> | "
            "5D%=<float or ''> | ATR%=<float or ''> | Top=<comma separated tickers or ''>\n\n"
            "Your job:\n"
            "1. Infer if the environment is risk-on, risk-off, or neutral.\n"
            "2. Identify the top 3 strongest sectors (by score and confirmation from RVOL/Breadth/5D%/Style).\n"
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

    # Post-processing tweaks

    # 1) For tiny universes (1–2 sectors), keep only the top strong
    #    and avoid overlap between strong and weak lists.
    if state.sectors and len(state.sectors) <= 2:
        structured = normalize_small_universe(structured)

    # 2) Scrub obviously bogus basing/reverting tags (e.g. ultra-low breadth).
    structured = scrub_basing(structured, state.raw_rows)

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
            "- Which sectors are leading and why (volume, breadth, momentum, style tags).\n"
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
