"""Node for generating the full Wikipedia article."""

from typing import Optional, cast

from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import ARTICLE_WRITER_PROMPT, SECTION_WRITER_PROMPT
from web_research_graph.state import Section, State
from web_research_graph.utils import dict_to_section, load_chat_model


async def create_retriever(
    references: Optional[dict[str, str]],
) -> VectorStoreRetriever:
    """Create a retriever from the reference documents."""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    reference_docs = [
        Document(page_content=content, metadata={"source": source})
        for source, content in references.items()
    ]

    vectorstore = InMemoryVectorStore.from_documents(
        reference_docs,
        embedding=embeddings,
    )
    return vectorstore.as_retriever(k=3)


async def generate_section(
    outline_str: str,
    section_title: str,
    topic: str,
    retriever: VectorStoreRetriever,
    config: Optional[RunnableConfig] = None,
) -> Section:
    """Generate a single section of the article."""
    # Get configuration
    configuration = Configuration.from_runnable_config(config)

    # Retrieve relevant documents
    docs = await retriever.ainvoke(f"{topic}: {section_title}")
    formatted_docs = "\n".join(
        f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
        for doc in docs
    )

    # Create the chain
    model = load_chat_model(configuration.long_context_model, max_tokens=2000)
    chain = (SECTION_WRITER_PROMPT | model.with_structured_output(Section)).with_config(
        config
    )

    # Generate the section
    result = await chain.ainvoke(
        {
            "outline": outline_str,
            "section": section_title,
            "docs": formatted_docs,
        }
    )
    assert isinstance(result, Section), f"Unexpected return type: {type(result)}"
    return result


async def generate_article(
    state: State, config: Optional[RunnableConfig] = None
) -> State:
    """Generate the complete Wikipedia article."""
    if not state.outline:
        raise ValueError("No outline found in state")

    # Convert dictionary outline to Outline object if needed
    current_outline = state.outline

    # Create retriever from references in state
    retriever = cast(
        VectorStoreRetriever,
        (await create_retriever(state.references)).with_config(config),
    )

    # Generate each section in parallel
    sections = []
    for section in current_outline.sections:
        section_content = await generate_section(
            current_outline.as_str,
            section.section_title,
            current_outline.page_title,
            retriever,
            config,
        )
        # Convert dictionary to Section object if needed
        if isinstance(section_content, dict):
            section_content = dict_to_section(section_content)
        sections.append(section_content)

    # Format all sections for final article generation
    draft = "\n\n".join(section.as_str for section in sections)

    # Create the chain for generating the final article
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.long_context_model, max_tokens=4000)
    chain = (ARTICLE_WRITER_PROMPT | model | StrOutputParser()).with_config(config)

    # Generate the final article
    final_article = await chain.ainvoke({"draft": draft, "topic": state.topic.topic})

    # Update state with the generated article
    return {
        "article": final_article,
        "is_last_step": True,
    }  # type: ignore
