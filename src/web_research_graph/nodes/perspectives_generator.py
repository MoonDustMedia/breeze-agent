"""Node for generating diverse editorial perspectives."""

import random

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import PERSPECTIVES_PROMPT
from web_research_graph.state import Perspectives, State
from web_research_graph.utils import load_chat_model


def format_doc(doc: Document, max_length: int = 1000) -> str:
    """Format a Wikipedia document for use in prompts."""
    all_related = doc.metadata.get("related_titles", [])
    related = ", ".join(random.sample(all_related, min(10, len(all_related))))
    return f"### {doc.metadata['title']}\n- Summary: {doc.page_content[: int(max_length * 2 / 3)]}\n\n- Related: {related[: int(max_length / 3)]}"


def format_docs(docs: list[Document]) -> str:
    """Format multiple Wikipedia documents."""
    return "\n\n".join(format_doc(doc) for doc in docs)


async def generate_perspectives(state: State, config: RunnableConfig) -> State:
    """Generate diverse editorial perspectives based on related topics."""
    configuration = Configuration.from_runnable_config(config)

    # Initialize the Wikipedia retriever
    wikipedia_retriever = WikipediaRetriever(
        load_all_available_meta=True, top_k_results=1
    ).with_config(config)

    # Get related topics from state
    if not state.related_topics:
        raise ValueError("No related topics found in state")

    # Retrieve Wikipedia documents for each topic
    retrieved_docs = await wikipedia_retriever.abatch(
        state.related_topics.topics, return_exceptions=True
    )

    # Filter out any failed retrievals and format the successful ones
    all_docs: list[Document] = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)

    formatted_docs = format_docs(all_docs)

    # Initialize the model and create the chain
    model = load_chat_model(configuration.fast_llm_model)
    chain = (
        PERSPECTIVES_PROMPT | model.with_structured_output(Perspectives)
    ).with_config(config)

    # Generate perspectives
    perspectives = await chain.ainvoke(
        {"examples": formatted_docs, "topic": state.topic.topic}
    )

    return {"perspectives": perspectives}  # type: ignore
