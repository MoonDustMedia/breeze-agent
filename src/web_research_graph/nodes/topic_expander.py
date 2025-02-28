"""Node for expanding topics with related subjects."""

from typing import Dict

from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import RELATED_TOPICS_PROMPT
from web_research_graph.state import RelatedTopics, State
from web_research_graph.utils import load_chat_model


async def expand_topics(
    state: State, config: RunnableConfig
) -> Dict[str, RelatedTopics]:
    """Expand a topic with related subjects."""
    configuration = Configuration.from_runnable_config(config)

    # Initialize the fast LLM for topic expansion
    model = load_chat_model(configuration.fast_llm_model)

    # Get the topic from the last user message
    last_user_message = next(
        (msg for msg in reversed(state.messages) if msg.type == "human"),
        None,
    )
    if not last_user_message:
        raise ValueError("No user message found in state")

    # Create the chain for topic expansion with structured output
    chain = (
        RELATED_TOPICS_PROMPT | model.with_structured_output(RelatedTopics)
    ).with_config(config)

    # Generate related topics
    related_topics = await chain.ainvoke({"topic": last_user_message.content})

    return {
        "related_topics": related_topics,
    }
