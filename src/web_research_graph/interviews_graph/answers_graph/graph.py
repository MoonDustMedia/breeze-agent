"""Define the answer generation workflow graph."""

from langgraph.graph import StateGraph

from web_research_graph.interviews_graph.answers_graph.nodes.generate import (
    generate_expert_answer,
)
from web_research_graph.interviews_graph.answers_graph.nodes.search import (
    search_for_context,
)
from web_research_graph.state import InterviewState

builder = StateGraph(InterviewState)

# Add nodes
builder.add_node("search_context", search_for_context)
builder.add_node("generate_answer", generate_expert_answer)

# Add edges
builder.set_entry_point("search_context")
builder.add_edge("search_context", "generate_answer")
builder.set_finish_point("generate_answer")

answer_graph = builder.compile()
answer_graph.name = "Answer Generator"
