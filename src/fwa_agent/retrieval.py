"""Information retrieval for FieldWorkArena tasks.

Integrates with arXiv and Semantic Scholar APIs for academic paper search.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents an academic paper."""
    title: str
    abstract: str
    authors: list[str]
    url: str
    published: str = ""
    venue: str = ""
    citations: int = 0


@dataclass
class SearchResult:
    """Search result container."""
    query: str
    papers: list[Paper] = field(default_factory=list)
    total: int = 0
    source: str = ""


class ArxivClient:
    """Client for arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str, max_results: int = 5) -> SearchResult:
        """Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResult with papers
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }

        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_arxiv_response(response.text)

            return SearchResult(
                query=query,
                papers=papers,
                total=len(papers),
                source="arxiv",
            )
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return SearchResult(query=query, source="arxiv")

    def _parse_arxiv_response(self, xml_text: str) -> list[Paper]:
        """Parse arXiv XML response into Paper objects."""
        import xml.etree.ElementTree as ET

        papers = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)
                published = entry.find("atom:published", ns)

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text or "")

                papers.append(Paper(
                    title=title.text.strip() if title is not None else "",
                    abstract=summary.text.strip()[:500] if summary is not None else "",
                    authors=authors,
                    url=link.text or "" if link is not None else "",
                    published=published.text or "" if published is not None else "",
                ))
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return papers

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class SemanticScholarClient:
    """Client for Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def search(self, query: str, max_results: int = 5) -> SearchResult:
        """Search Semantic Scholar for papers.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResult with papers
        """
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,venue,citationCount,url",
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            papers = []

            for item in data.get("data", []):
                authors = [a.get("name", "") for a in item.get("authors", [])]
                papers.append(Paper(
                    title=item.get("title", ""),
                    abstract=(item.get("abstract") or "")[:500],
                    authors=authors,
                    url=item.get("url", ""),
                    published=str(item.get("year", "")),
                    venue=item.get("venue", ""),
                    citations=item.get("citationCount", 0),
                ))

            return SearchResult(
                query=query,
                papers=papers,
                total=data.get("total", len(papers)),
                source="semantic_scholar",
            )
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return SearchResult(query=query, source="semantic_scholar")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class ResearchRetriever:
    """Unified research paper retrieval."""

    def __init__(self):
        self.arxiv = ArxivClient()
        self.semantic_scholar = SemanticScholarClient()

    async def search(self, query: str, max_results: int = 5) -> SearchResult:
        """Search across multiple sources.

        Args:
            query: Search query
            max_results: Maximum results per source

        Returns:
            Combined search results
        """
        # Search both sources
        arxiv_result = await self.arxiv.search(query, max_results)
        ss_result = await self.semantic_scholar.search(query, max_results)

        # Combine results, dedup by title similarity
        all_papers = arxiv_result.papers + ss_result.papers
        seen_titles = set()
        unique_papers = []

        for paper in all_papers:
            title_lower = paper.title.lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)

        # Sort by citations (if available)
        unique_papers.sort(key=lambda p: p.citations, reverse=True)

        return SearchResult(
            query=query,
            papers=unique_papers[:max_results],
            total=len(unique_papers),
            source="combined",
        )

    async def close(self):
        """Close all clients."""
        await self.arxiv.close()
        await self.semantic_scholar.close()
