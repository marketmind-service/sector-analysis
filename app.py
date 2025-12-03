import textwrap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from state import AgentState, SectorState
from sector_agent import sector_agent, sector_agent_direct

app = FastAPI(title="Market Sector Analysis API")


@app.post("/api/sector-agent", response_model=AgentState)
async def run_sector(state: AgentState):
    try:
        updated_state = await sector_agent(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sector_agent_error: {e}")

    return updated_state


class DirectSectorRequest(BaseModel):
    sectors: List[str]


@app.post("/api/sector", response_model=SectorState)
async def direct_sector(req: DirectSectorRequest):
    in_state = SectorState(
        sectors=req.sectors,
        source="direct"
    )

    try:
        out_state = await sector_agent_direct(in_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sector_error: {e}")

    if out_state.error:
        raise HTTPException(status_code=400, detail=out_state.error)

    return out_state


async def local_cli():
    print("MarketMind CLI (type 'exit' to quit)")
    while True:
        prompt = input("\nYou: ").strip()
        if not prompt or prompt.lower() in {"exit", "quit"}:
            print("Done.")
            break
        state = AgentState(prompt=prompt)
        try:
            result = await sector_agent(state)
            print(textwrap.dedent(f"""
                ===================================== RESULTS =====================================
                Prompt: {result.prompt}
                Sectors: {result.sector_result.sectors}  
                Rows: {result.sector_result.raw_rows}
                Structured: {result.sector_result.structured_view}
                Commentary: {result.sector_result.interpreted_results}
                ===================================================================================
            """).strip())
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(local_cli())
