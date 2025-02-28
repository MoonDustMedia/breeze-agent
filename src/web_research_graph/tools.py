"""This module provides tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.language_models import BaseChatModel
from typing_extensions import Annotated
from langchain_core.output_parsers import StrOutputParser
from web_research_graph.utils import load_chat_model

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import QUERY_SUMMARIZATION_PROMPT


async def summarize_query(query: str, model: BaseChatModel, config: RunnableConfig) -> str:
    """Summarize a long query into a shorter, focused version."""

    chain = (QUERY_SUMMARIZATION_PROMPT | model | StrOutputParser()).with_config(config)
    return await chain.ainvoke({"query": query})


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.

    If the query is longer than 350 characters, it will be automatically summarized
    using an LLM to create a more focused search query.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.long_context_model)
    # If query is too long, summarize it using the LLM
    if len(query) > 40:
        query = await summarize_query(query, model, config)
    
    wrapped = DuckDuckGoSearchAPIWrapper()
    result = wrapped.results(query, configuration.max_search_results)
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search]
