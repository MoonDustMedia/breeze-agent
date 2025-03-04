"""Router functions for managing interview flow."""

from langchain_core.messages import AIMessage

from web_research_graph.state import InterviewState
from web_research_graph.utils import sanitize_name

MAX_TURNS = 3
EXPERT_NAME = "expert"


def route_messages(state: InterviewState) -> str:
    """Determine whether to continue the interview or end it."""
    editor = state.editors[state.current_editor_index]
    current_editor_name = sanitize_name(editor.name)

    messages = state.interviews[state.current_editor_index]
    # Get the last message
    if messages:
        last_message = messages[-1]

        # If the last message was from the expert, wait for editor's response
        if isinstance(last_message, AIMessage) and last_message.name == EXPERT_NAME:
            return "ask_question"

        # If the last message was from the editor
        if (
            isinstance(last_message, AIMessage)
            and last_message.name == current_editor_name
        ):
            # Count expert responses in this conversation
            expert_responses = len(
                [
                    m
                    for m in messages
                    if isinstance(m, AIMessage) and m.name == EXPERT_NAME
                ]
            )

            # Check if we've reached max turns or got a thank you
            if expert_responses >= MAX_TURNS or last_message.content.endswith(
                "Thank you so much for your help!"
            ):
                return "next_editor"

            return "next_editor"

    # If we're just starting, ask a question
    return "ask_question"
