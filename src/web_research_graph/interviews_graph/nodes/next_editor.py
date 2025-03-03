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

    # first expert response
    initial_message = AIMessage(
        content=f"So you said you were writing an article on {state.topic.topic}?",
        name=EXPERT_NAME,
    )
    interviews = state.interviews
    interviews[next_index].append(initial_message)

    return {
        "current_editor_index": next_index,
        "interviews": interviews,
    }  # type: ignore
