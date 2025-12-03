from state import AgentState, SectorState


def into_sector_state(parent: AgentState, child: SectorState) -> SectorState:
    return child.model_copy(update={
        "source": "agent",
        "prompt": parent.prompt,
    })


def out_of_sector_state(parent: AgentState, child: SectorState) -> AgentState:
    return parent.model_copy(update={
        "sector_result": child,
        "route_taken": [*parent.route_taken, "sector_agent_done"],
    })
