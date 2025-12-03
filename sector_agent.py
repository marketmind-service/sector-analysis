from typing import cast
from langchain_core.runnables import RunnableConfig
from state import AgentState, SectorState
from sector_graph import create_sector_graph
from sector_adapters import into_sector_state, out_of_sector_state


async def sector_agent(parent: AgentState) -> AgentState:
    in_state = into_sector_state(parent, SectorState())
    raw = await create_sector_graph().ainvoke(
        in_state,
        config=cast(RunnableConfig, cast(object, {"recursion_limit": 100}))
    )

    out_state = out_of_sector_state(parent, SectorState(**raw))
    return out_state

async def sector_agent_direct(state: SectorState) -> SectorState:
    raw = await create_sector_graph().ainvoke(
        state,
        config=cast(RunnableConfig, cast(object, {"recursion_limit": 100}))
    )

    return SectorState(**raw)
