"""Node for handling invalid topics and waiting for user input."""

from typing import Dict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages

from web_research_graph.state import State


async def request_topic(
    state: State, config: RunnableConfig
) -> Dict[str, list[AIMessage]]:
    """Request a new topic from the user."""
    message = state.topic.message or "Please provide a specific topic for research."

    new_message = AIMessage(content=message)
    new_messages = add_messages(state.messages, [new_message])

    return {"messages": new_messages}
