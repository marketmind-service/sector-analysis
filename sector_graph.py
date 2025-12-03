from langgraph.graph import StateGraph, END
from state import SectorState
from parse_input import parse_input
from fetch_data import fetch_data
from interpret_results import structure_results, interpret_results


async def entry(state: SectorState) -> SectorState:
    return state


def create_sector_graph():
    graph = StateGraph(SectorState)
    graph.set_entry_point("entry")

    graph.add_node("entry", entry)
    graph.add_node("parse_input", parse_input)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("structure_results", structure_results)
    graph.add_node("interpret_results", interpret_results)

    graph.add_conditional_edges(
        "entry",
        lambda state: state.source,
        {
            "agent": "parse_input",
            "direct": "fetch_data",
        }
    )

    graph.add_edge("parse_input", "fetch_data")
    graph.add_edge("fetch_data", "structure_results")
    graph.add_edge("structure_results", "interpret_results")
    graph.add_edge("interpret_results", END)

    return graph.compile()
