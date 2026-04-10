# ============================================================
# Shared Utility Functions
# Research Paper Pipeline - Utils Module
# ============================================================
"""
Provides shared utilities: API clients, checkpoint management, output directory creation.
API Progressive Strategy: Semantic Scholar -> OpenAlex -> CrossRef
"""

import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Utility 1: Load .env Environment Variables
# ============================================================
def load_env() -> None:
    """
    Load environment variables from .env file into os.environ.
    Uses python-dotenv.
    """
    try:
        from dotenv import load_dotenv

        # Try multiple possible .env paths
        possible_paths = [".env", "../.env", Path(__file__).parent / ".env"]
        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                load_dotenv(path)
                loaded = True
                logger.info(f"Environment variables loaded from {path}")
                break
        if not loaded:
            logger.warning(".env file not found, environment variables may not be loaded")
    except ImportError:
        logger.warning("python-dotenv not installed, cannot auto-load .env file")


load_env()


# ============================================================
# Utility 2: API Clients (Progressive Fallback Strategy)
# ============================================================

def _semantic_scholar_fetch(topic: str, count: int = 20) -> list[dict]:
    """
    Fetch paper metadata from Semantic Scholar API.
    API Docs: https://api.semanticscholar.org/api-docs/graph
    Free, works without API key for limited calls (key recommended for higher limits).
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": topic,
        "limit": min(count, 100),  # Free tier: max 100 per call
        "fields": "title,abstract,authors,year,venue,citationCount,externalIds,openAccessPdf",
        "sort": "citationCount:desc",
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    if response.status_code == 200:
        data = response.json()
        papers = []
        for item in data.get("data", []):
            papers.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstract", "") or "",
                "keywords": [],  # Semantic Scholar doesn't provide keywords directly; extracted from title/abstract
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "year": item.get("year"),
                "venue": item.get("venue", "") or "",
                "citation_count": item.get("citationCount", 0),
            })
        logger.info(f"Semantic Scholar returned {len(papers)} papers")
        return papers
    else:
        logger.warning(
            f"Semantic Scholar API call failed: HTTP {response.status_code} - {response.text[:200]}"
        )
        response.raise_for_status()


def _openalex_fetch(topic: str, count: int = 20) -> list[dict]:
    """
    Fetch paper metadata from OpenAlex API.
    OpenAlex is completely free, no API key required.
    API Docs: https://docs.openalex.org
    """
    url = "https://api.openalex.org/works"
    params = {
        "search": topic,
        "per-page": min(count, 100),
        "sort": "cited_by_count:desc",
        "filter": "type:article",  # Only academic articles
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        data = response.json()
        papers = []
        for item in data.get("results", []):
            # Extract authors
            authors = []
            for au in item.get("authorships", []):
                au_name = au.get("author", {}).get("display_name", "")
                if au_name:
                    authors.append(au_name)

            # Extract keywords (from keywords array)
            keywords = [kw.get("display_name", "") for kw in item.get("keywords", [])]

            # Extract venue
            primary_loc = item.get("primary_location", {})
            source = primary_loc.get("source", {}) or {}
            venue = source.get("display_name", "") or ""

            papers.append({
                "title": item.get("display_name", ""),
                "abstract": item.get("abstract_inverted_index") or "",
                "keywords": keywords,
                "authors": authors,
                "year": item.get("publication_year"),
                "venue": venue,
                "citation_count": item.get("cited_by_count", 0),
            })
        logger.info(f"OpenAlex returned {len(papers)} papers")
        return papers
    else:
        logger.warning(
            f"OpenAlex API call failed: HTTP {response.status_code} - {response.text[:200]}"
        )
        response.raise_for_status()


def _crossref_fetch(topic: str, count: int = 20) -> list[dict]:
    """
    Fetch paper metadata from CrossRef API.
    CrossRef is completely free, no API key required.
    API Docs: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
    """
    url = "https://api.crossref.org/works"
    params = {
        "query": topic,
        "rows": min(count, 100),
        "sort": "is-referenced-by-count:desc",
        "filter": "type:journal-article",
    }
    headers = {
        "User-Agent": "Research-Pipeline/1.0 (mailto:example@example.com)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code == 200:
        data = response.json()
        papers = []
        for item in data.get("message", {}).get("items", []):
            # Extract authors
            authors = []
            for au in item.get("author", []):
                au_name = f"{au.get('given', '')} {au.get('family', '')}".strip()
                if au_name:
                    authors.append(au_name)

            # Extract keywords
            keywords = [kw for kw in item.get("subject", [])]

            # Extract year
            date_parts = item.get("published-print", {}) or item.get("published-online", {}) or item.get("created", {})
            year = None
            if date_parts.get("date-parts"):
                year = date_parts["date-parts"][0][0]

            # Extract venue
            container = item.get("container-title", [])
            venue = container[0] if container else ""

            papers.append({
                "title": item.get("title", [""])[0] if item.get("title") else "",
                "abstract": item.get("abstract", "") or "",
                "keywords": keywords,
                "authors": authors,
                "year": year,
                "venue": venue,
                "citation_count": item.get("is-referenced-by-count", 0),
            })
        logger.info(f"CrossRef returned {len(papers)} papers")
        return papers
    else:
        logger.warning(
            f"CrossRef API call failed: HTTP {response.status_code} - {response.text[:200]}"
        )
        response.raise_for_status()


def fetch_papers(
    topic: str,
    count: int = 20,
    time_range: Optional[dict[str, str]] = None,
    max_retries: int = 3,
) -> list[dict]:
    """
    Main function for fetching paper metadata.

    Progressive strategy: Semantic Scholar -> OpenAlex -> CrossRef
    If any API fails, automatically tries the next one.
    Only raises an exception if all APIs fail.

    Args:
        topic: Research topic keywords
        count: Maximum number of papers
        time_range: Optional, time range {"from": "2020-01-01", "to": "2025-12-31"}
        max_retries: Maximum retries per API (default 3)

    Returns:
        list[dict]: Paper metadata list
    """
    # Semantic Scholar
    try:
        papers = _semantic_scholar_fetch(topic, count)
        if papers:
            return _filter_by_time(papers, time_range)
    except Exception as e:
        logger.warning(f"Semantic Scholar call failed: {e}")

    # OpenAlex
    try:
        papers = _openalex_fetch(topic, count)
        if papers:
            return _filter_by_time(papers, time_range)
    except Exception as e:
        logger.warning(f"OpenAlex call failed: {e}")

    # CrossRef
    try:
        papers = _crossref_fetch(topic, count)
        if papers:
            return _filter_by_time(papers, time_range)
    except Exception as e:
        logger.warning(f"CrossRef call failed: {e}")

    # All APIs failed
    raise RuntimeError(
        f"All academic APIs failed to retrieve papers. Topic: {topic}. "
        f"Please check network connection and API configuration."
    )


def _filter_by_time(
    papers: list[dict], time_range: Optional[dict[str, str]]
) -> list[dict]:
    """
    Filter papers by time range.
    If time_range is None or a paper has no year field, keep all.
    """
    if not time_range:
        return papers

    try:
        from_year = int(time_range.get("from", "1900")[:4])
        to_year = int(time_range.get("to", "2100")[:4])
    except (ValueError, TypeError):
        return papers

    filtered = []
    for p in papers:
        year = p.get("year")
        if year and from_year <= year <= to_year:
            filtered.append(p)

    logger.info(f"Time range filter retained {len(filtered)}/{len(papers)} papers")
    return filtered


# ============================================================
# Utility 3: Checkpoint Manager
# ============================================================

def load_checkpoint(checkpoint_path: str = "checkpoint.json") -> dict[str, Any]:
    """
    Load checkpoint file.
    Returns empty dict if file doesn't exist (for first run).

    Args:
        checkpoint_path: Checkpoint file path

    Returns:
        dict: Checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint file not found, will create new file: {checkpoint_path}")
        return {}

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Checkpoint file read failed: {e}, will use empty dict")
        return {}


def save_checkpoint(
    checkpoint_path: str,
    state: dict[str, Any],
    indent: int = 2,
) -> None:
    """
    Write state to checkpoint file (atomic write).

    Uses temporary file + rename to avoid corruption from interrupted writes.

    Args:
        checkpoint_path: Checkpoint file path
        state: State dict to save
        indent: JSON indent
    """
    temp_path = checkpoint_path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=indent)
        # Atomic replace
        os.replace(temp_path, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    except IOError as e:
        logger.error(f"Checkpoint write failed: {e}")
        raise


def clear_stage_from_checkpoint(
    checkpoint_path: str,
    stage_keys: list[str],
) -> dict[str, Any]:
    """
    Remove specified stage data from checkpoint (used for testing resume).
    Note: This modifies the checkpoint.json file.

    Args:
        checkpoint_path: Checkpoint file path
        stage_keys: List of stage keys to remove

    Returns:
        dict: Updated checkpoint
    """
    state = load_checkpoint(checkpoint_path)
    for key in stage_keys:
        if key in state:
            del state[key]
            logger.info(f"Stage removed from checkpoint: {key}")
    if state:
        save_checkpoint(checkpoint_path, state)
    return state


# ============================================================
# Utility 4: Output Directory Management
# ============================================================

def ensure_output_dir(output_dir: str, research_topic: str) -> str:
    """
    Create output directory (if it doesn't exist).

    Directory structure: output_dir/research_topic/

    Args:
        output_dir: Base output directory
        research_topic: Research topic (used for subdirectory naming)

    Returns:
        str: Created/confirmed output directory absolute path
    """
    # Sanitize research topic as directory name (replace illegal characters)
    safe_topic = research_topic.replace("/", "_").replace("\\", "_")
    safe_topic = safe_topic.replace(":", "_").replace("*", "_")
    safe_topic = safe_topic.replace("?", "_").replace('"', "_")
    safe_topic = safe_topic.replace("<", "_").replace(">", "_")
    safe_topic = safe_topic.replace("|", "_")

    output_path = os.path.join(output_dir, safe_topic)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory confirmed: {output_path}")
    return output_path


def get_output_path(output_dir: str, research_topic: str, filename: str) -> str:
    """
    Get full path of an output file.

    Args:
        output_dir: Base output directory
        research_topic: Research topic
        filename: Filename (with extension)

    Returns:
        str: Full file path
    """
    dir_path = ensure_output_dir(output_dir, research_topic)
    return os.path.join(dir_path, filename)


# ============================================================
# Utility 5: Keyword Extraction
# ============================================================

def extract_keywords_from_papers(papers: list[dict]) -> list[str]:
    """
    Extract all keywords from paper list.
    Prefers paper's keywords field; if empty, extracts from title.

    Args:
        papers: Paper metadata list

    Returns:
        list[str]: Keyword list (may contain duplicates)
    """
    keywords = []
    stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "we", "our", "us", "i", "my", "me", "he", "she", "him",
        "her", "what", "which", "who", "whom", "how", "when", "where", "why",
    }

    for paper in papers:
        # Prefer keywords field
        if paper.get("keywords"):
            for kw in paper["keywords"]:
                kw_clean = kw.lower().strip()
                if kw_clean and kw_clean not in stopwords:
                    keywords.append(kw_clean)
        else:
            # Extract from title (simple tokenization)
            title = paper.get("title", "")
            words = title.replace("-", " ").replace("/", " ").split()
            for word in words:
                word_clean = word.lower().strip(".,;:!?'\"()[]{}")
                if (
                    len(word_clean) > 3
                    and word_clean not in stopwords
                    and not word_clean.isdigit()
                ):
                    keywords.append(word_clean)

    return keywords


def keyword_frequency(keywords: list[str]) -> dict[str, int]:
    """
    Count keyword frequencies.

    Args:
        keywords: Keyword list

    Returns:
        dict[str, int]: Word frequency dict
    """
    freq: dict[str, int] = {}
    for kw in keywords:
        freq[kw] = freq.get(kw, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
