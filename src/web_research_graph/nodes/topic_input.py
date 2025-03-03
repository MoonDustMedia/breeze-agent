"""Node for handling invalid topics and waiting for user input."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.state import State


async def request_topic(state: State, config: RunnableConfig) -> State:
    """Request a new topic from the user."""
    message = state.topic.message or "Please provide a specific topic for research."

    return {"messages": AIMessage(content=message)}  # type: ignore
