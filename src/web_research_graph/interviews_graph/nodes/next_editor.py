"""Node for managing editor transitions in interviews."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.state import InterviewState

EXPERT_NAME = "expert"


async def next_editor(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Move to the next editor or end if all editors are done."""
    next_index = state.current_editor_index + 1

    if next_index >= len(state.editors):
        return {"is_complete": True, "current_editor_index": next_index}  # type: ignore

    # Add a separator message to mark the start of a new conversation
    separator = AIMessage(
        content=f"\n--- Starting interview with {state.editors[next_index].name} ---\n",
        name="system",
    )

    # Start fresh conversation with next editor while keeping history
    initial_message = AIMessage(
        content="So you said you were writing an article on this topic?",
        name=EXPERT_NAME,
    )

    return {
        "messages": [separator, initial_message],
        "current_editor_index": next_index,
    }  # type: ignore
