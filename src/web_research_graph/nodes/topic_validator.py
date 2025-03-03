"""Node for validating and extracting the topic from user input."""

from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import TOPIC_VALIDATOR_PROMPT
from web_research_graph.state import State, TopicValidation
from web_research_graph.utils import load_chat_model


async def validate_topic(state: State, config: RunnableConfig) -> State:
    """Validate and extract the topic from user input."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)

    # Validate the topic using structured output
    chain = (
        TOPIC_VALIDATOR_PROMPT | model.with_structured_output(TopicValidation)
    ).with_config(config)
    response = await chain.ainvoke({"input": state.input})

    return {
        "topic": response,
    }  # type: ignore
