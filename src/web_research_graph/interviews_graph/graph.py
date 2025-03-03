"""Define the interview workflow graph."""

from langgraph.graph import END, StateGraph
from langgraph.pregel import RetryPolicy

from web_research_graph.interviews_graph.answers_graph.graph import answer_graph
from web_research_graph.interviews_graph.nodes.initialize import initialize_interview
from web_research_graph.interviews_graph.nodes.next_editor import next_editor
from web_research_graph.interviews_graph.nodes.question import generate_question
from web_research_graph.interviews_graph.router import route_messages
from web_research_graph.state import InterviewState

builder = StateGraph(InterviewState)

# FIXME:
# - each interview should have its own dialogue history
# - all interview dialogues should be accessible for the next step
# - in the paper, they are not including the interview history for the expert to answer; though it is reasonable to provide it.

# Add nodes
builder.add_node("initialize", initialize_interview)
builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=5))
builder.add_node("answer_question", answer_graph, retry=RetryPolicy(max_attempts=5))
builder.add_node("next_editor", next_editor)

# Add edges
builder.set_entry_point("initialize")
builder.add_edge("initialize", "ask_question")
builder.add_conditional_edges(
    "answer_question",
    route_messages,
    {"ask_question": "ask_question", "next_editor": "next_editor", "end": END},
)
builder.add_edge("ask_question", "answer_question")
builder.add_conditional_edges(
    "next_editor",
    lambda x: "end" if x.is_complete else "ask_question",
    {"ask_question": "ask_question", "end": END},
)

interview_graph = builder.compile()
interview_graph.name = "Interview Conductor"
