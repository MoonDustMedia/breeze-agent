"""Node for refining the outline based on interview results."""

from typing import Optional

from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import REFINE_OUTLINE_PROMPT
from web_research_graph.state import Outline, State
from web_research_graph.utils import get_message_text, load_chat_model


async def refine_outline(
    state: State, config: Optional[RunnableConfig] = None
) -> State:
    """Refine the outline based on interview results."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.tool_model)

    if not state.outline:
        raise ValueError("No initial outline found in state")

    current_outline = state.outline

    # Format conversations from the state's messages
    conversations = "\n\n".join(
        f"### {m.name}\n\n{get_message_text(m)}" for m in state.messages
    )

    # Create the chain with structured output
    chain = (REFINE_OUTLINE_PROMPT | model.with_structured_output(Outline)).with_config(
        config
    )

    # Generate refined outline with explicit structure validation
    refined_outline = await chain.ainvoke(
        {
            "topic": state.topic.topic,
            "old_outline": current_outline.as_str,
            "conversations": conversations,
        }
    )

    # Ensure we maintain the structure
    if not refined_outline.sections:
        # If no sections, copy the sections from the current outline
        refined_outline.sections = current_outline.sections
    elif len(refined_outline.sections) < len(current_outline.sections):
        # If we lost sections, merge with original
        existing_sections = {s.section_title: s for s in current_outline.sections}
        refined_sections = {s.section_title: s for s in refined_outline.sections}
        # Update existing sections with refined ones, keeping any that weren't refined
        existing_sections.update(refined_sections)
        refined_outline.sections = list(existing_sections.values())

    # Return updated state with new outline
    return {
        "outline": refined_outline,
    }  # type: ignore
