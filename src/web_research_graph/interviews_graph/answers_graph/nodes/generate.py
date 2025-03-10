"""Node for generating expert answers."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import INTERVIEW_ANSWER_PROMPT
from web_research_graph.state import InterviewState
from web_research_graph.utils import load_chat_model

EXPERT_NAME = "expert"


async def generate_expert_answer(
    state: InterviewState, config: RunnableConfig
) -> InterviewState:
    """Generate an expert answer using the gathered information."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)

    editor = state.editors[state.current_editor_index]
    if editor is None:
        raise ValueError("Editor not found in state")
    messages = state.interviews[state.current_editor_index]

    # Format references for the prompt
    references_text = ""
    if state.references:
        references_text = "\n\n".join(
            f"Source: {url}\nContent: {content}"
            for url, content in state.references.items()
        )

    # Create the chain
    chain = (INTERVIEW_ANSWER_PROMPT | model).with_config(config)

    # Generate answer
    result = await chain.ainvoke({"messages": messages, "references": references_text})

    content = result.content if hasattr(result, "content") else str(result)

    if not content:
        return state

    return {
        "messages": AIMessage(content=content, name=EXPERT_NAME),
    }  # type: ignore
