"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from web_research_graph import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    fast_llm_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="groq/llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the fast language model to use for simpler tasks. "
            "Should be in the form: provider/model-name."
        },
    )

    tool_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="groq/llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the fast language model to use for tool tasks. "
            "Should be in the form: provider/model-name."
        },
    )

    long_context_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        field(
            default="anthropic/claude-3-7-sonnet-20250219",
            metadata={
                "description": "The name of the language model to use for tasks requiring longer context. "
                "Should be in the form: provider/model-name."
            },
        )
    )

    max_search_results: int = field(
        default=4,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
