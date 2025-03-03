"""Node for searching relevant context for answers."""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.state import InterviewState
from web_research_graph.tools import search
from web_research_graph.utils import swap_roles

EXPERT_NAME = "expert"


async def search_for_context(
    state: InterviewState, config: RunnableConfig
) -> InterviewState:
    """Search for relevant information to answer the question."""
    editor = state.editors[state.current_editor_index]
    if editor is None:
        raise ValueError("Editor not found in state")

    messages = state.interviews[state.current_editor_index]
    # Swap roles to get the correct perspective
    swapped_messages = swap_roles(messages, EXPERT_NAME)

    # Get the last question (now as HumanMessage after swap)
    last_question = next(
        (msg for msg in reversed(swapped_messages) if isinstance(msg, HumanMessage)),
        None,
    )

    if not last_question:
        return state

    # Perform search
    search_results = await search(str(last_question.content), config=config)

    # Store results in references
    if search_results:
        references = {}
        for result in search_results:
            if isinstance(result, dict):
                references[result.get("link", "unknown")] = result.get("snippet", "")
            elif isinstance(result, str):
                references[f"source_{len(references)}"] = result
        return {
            "references": references,
        }  # type: ignore

    return {}  # type: ignore
