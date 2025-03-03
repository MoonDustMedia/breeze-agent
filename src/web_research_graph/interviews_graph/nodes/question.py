"""Node for generating interview questions from editors."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import INTERVIEW_QUESTION_PROMPT
from web_research_graph.state import InterviewState
from web_research_graph.utils import load_chat_model, sanitize_name, swap_roles


async def generate_question(
    state: InterviewState, config: RunnableConfig
) -> InterviewState:
    """Generate a question from the editor's perspective."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)

    editor = state.editors[state.current_editor_index]
    if editor is None:
        raise ValueError(
            "Editor not found in state. Make sure to set the editor before starting the interview."
        )

    editor_name = sanitize_name(editor.name)
    swapped = swap_roles(state, editor_name)

    chain = (INTERVIEW_QUESTION_PROMPT | model).with_config(config)

    result = await chain.ainvoke({"messages": swapped, "persona": editor.persona})

    content = result.content if hasattr(result, "content") else str(result)

    return {"messages": AIMessage(content=content, name=editor_name)}  # type: ignore
