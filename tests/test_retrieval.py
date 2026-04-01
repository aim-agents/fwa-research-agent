"""Tests for information retrieval module."""

import pytest

from fwa_agent.retrieval import Paper, SearchResult, ResearchRetriever


def test_paper_creation():
    """Test Paper dataclass."""
    paper = Paper(
        title="Test Paper",
        abstract="A test abstract",
        authors=["Author One", "Author Two"],
        url="https://arxiv.org/abs/1234.5678",
        citations=10,
    )
    assert paper.title == "Test Paper"
    assert len(paper.authors) == 2


def test_search_result_creation():
    """Test SearchResult dataclass."""
    result = SearchResult(
        query="test query",
        papers=[],
        total=0,
        source="arxiv",
    )
    assert result.query == "test query"
    assert result.source == "arxiv"


@pytest.mark.asyncio
async def test_retriever_search():
    """Test ResearchRetriever search (may fail without network)."""
    retriever = ResearchRetriever()
    try:
        result = await retriever.search("machine learning", max_results=2)
        assert result.query == "machine learning"
        assert result.source == "combined"
    finally:
        await retriever.close()
