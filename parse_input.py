import re
import json
import textwrap
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from config import query
from state import SectorState

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


async def extract_sectors(prompt: str) -> List[str]:
    messages = [
        SystemMessage(
            content=textwrap.dedent("""
                Parse a sector relative strength request and output which sector ETFs to include.

                Return EXACTLY one JSON object and nothing else:
                {"sectors":[...]} or {"sectors":null}
                No spaces. No extra text
                
                SECTOR_ETFS={
                    "SMH":"Semiconductors",
                    "XLK":"Tech",
                    "XLC":"Comm Services",
                    "XLF":"Financials",
                    "XLE":"Energy",
                    "XBI":"Biotech",
                    "XLV":"Healthcare",
                    "XLI":"Industrials",
                    "IWM":"Small Caps",
                    "QQQ":"Nasdaq",
                    "SPY":"S&P"
                }
                
                Rules:
                1. If user mentions any ETF tickers (case insensitive), include them in order of appearance.
                2. Map phrases to tickers:
                    - tech, technology -> XLK
                    - semis, chips -> SMH
                    - comm services, telecom, media -> XLC
                    - financials, banks -> XLF
                    - energy, oil -> XLE
                    - biotech -> XBI
                    - healthcare, pharma -> XLV
                    - industrials, manufacturing -> XLI
                    - small caps, russell -> IWM
                    - nasdaq, ndx, growth -> QQQ
                    - s&p, spx, broad market -> SPY
                3. Comparisons ("X vs Y", "compare A and B") include ONLY the sectors mentioned
                4. "strongest sector", "weakest sector", "leading", "lagging" with no specific names means include ALL
                5. "defensives" -> ["XLV","XLI"]
                6. Fix obvious typos ("tehc"->tech->XLK).
                7. If nothing can be inferred: {"sectors":null}
            """).strip()
        ),
        HumanMessage(content=f"Prompt: {prompt}")
    ]

    response = query.invoke(messages)
    raw = response.content if isinstance(response.content, str) else str(response.content)

    sectors: List[str] = []

    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(raw[start:end + 1])
            val = obj.get("sectors", None)
            if isinstance(val, list):
                sectors = [
                    str(t).upper()
                    for t in val
                    if isinstance(t, str) and str(t).upper() in SECTOR_ETFS
                ]
    except Exception:
        pass

    # fallback - extract explicit tickers from user prompt
    if not sectors:
        upper_prompt = prompt.upper()
        seen = set()
        for ticker in SECTOR_ETFS.keys():
            pattern = r"\b" + re.escape(ticker) + r"\b"
            if re.search(pattern, upper_prompt):
                if ticker not in seen:
                    seen.add(ticker)
                    sectors.append(ticker)

    return sectors


async def parse_input(state: SectorState) -> SectorState:
    print("parse_input")

    sectors = await extract_sectors(state.prompt)

    print(f"Sectors: {sectors}")

    return state.model_copy(update={
        "sectors": sectors,
    })
