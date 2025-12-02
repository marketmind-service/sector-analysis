import textwrap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from state import AgentState
from sector_agent import sector_agent

app = FastAPI(title="Market Sector Analysis API")


@app.post("/api/sector-agent", response_model=AgentState)
async def run_sector(state: AgentState):
    try:
        updated_state = await sector_agent(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sector_agent_error: {e}")

    return updated_state


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
