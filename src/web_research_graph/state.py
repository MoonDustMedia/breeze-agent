"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from pydantic import BaseModel, Field
from typing_extensions import Annotated


@dataclass
class Editor:
    """Represents a Wikipedia editor with specific expertise."""

    affiliation: str
    name: str
    role: str
    description: str

    @property
    def persona(self) -> str:
        """Return a formatted string representation of the editor."""
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


@dataclass
class Perspectives:
    """Represents a group of editors with different perspectives."""

    editors: list[Editor] = field(default_factory=list)


class RelatedTopics(BaseModel):
    """Represents related topics for research."""

    topics: list[str] = Field(
        description="List of related topics that are relevant to the main research subject"
    )


class Subsection(BaseModel):
    """Represents a subsection in a Wikipedia article."""

    subsection_title: str = Field(description="The title of the subsection")
    description: str = Field(description="The detailed content of the subsection")

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the subsection."""
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    """Represents a section in a Wikipedia article."""

    section_title: str = Field(description="The title of the section")
    description: str = Field(description="The main content/summary of the section")
    subsections: list[Subsection] = Field(
        description="List of subsections within this section"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of citations supporting the section content",
    )

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the section."""
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join(
            [f"[{i + 1}] {cit}" for i, cit in enumerate(self.citations)]
        )
        return (
            f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


class Outline(BaseModel):
    """Represents a complete Wikipedia-style outline."""

    page_title: str = Field(description="The main title of the Wikipedia article")
    sections: list[Section] = Field(
        description="List of sections that make up the article"
    )

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the outline."""
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


@dataclass
class InputState:
    """Defines the input state for the agent."""

    input: Optional[str] = field(default=None)


@dataclass
class OutputState:
    """Defines the output state for the agent."""

    article: Optional[str] = field(default=None)


class TopicValidation(BaseModel):
    """Structured output for topic validation."""

    is_valid: bool = Field(
        description="Indicates whether the topic is valid for article generation"
    )
    topic: Optional[str] = Field(
        description="The validated and possibly reformulated topic"
    )
    message: Optional[str] = Field(
        description="Feedback message about the topic validation result"
    )


def default_topic_validation() -> TopicValidation:
    """Create a default TopicValidation instance."""
    return TopicValidation(is_valid=False, topic=None, message=None)


# Custom reducer function to merge dictionaries
def merge_dicts(existing: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """Merge two dictionaries, with new keys overriding old keys."""
    return {**existing, **new}


@dataclass
class State(InputState, OutputState):
    """Represents the complete state of the agent."""

    topic: TopicValidation = field(default_factory=default_topic_validation)
    related_topics: Optional[RelatedTopics] = field(default=None)
    is_last_step: IsLastStep = field(default=False)
    outline: Optional[Outline] = field(default=None)
    perspectives: Optional[Perspectives] = field(default=None)
    references: Annotated[dict[str, str], merge_dicts] = field(default_factory=dict)
    article: Optional[str] = field(default=None)
    interviews: list[Annotated[list[AnyMessage], add_messages]] = field(
        default_factory=list
    )
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


@dataclass
class InterviewState(State):
    """State for the interview process between editors and experts."""

    editors: list[Editor] = field(default_factory=list)
    current_editor_index: int = field(default=0)
    interviews: list[Annotated[list[AnyMessage], add_messages]] = field(
        default_factory=list
    )
    is_complete: bool = field(default=False)
