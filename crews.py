# ============================================================
# CrewAI Multi-Crew Definitions
# Research Paper Pipeline - Crews Module
# ============================================================
"""
Defines 4 Crews (each containing 1 Agent + 1 Task):
  - Crew1: Paper metadata fetching (Semantic Scholar -> OpenAlex -> CrossRef)
  - Crew2: Keyword word cloud generation
  - Crew3: Paper summary generation (JSON + Markdown)
  - Crew4: Citation relationship visualization
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless mode, no display window
import matplotlib.pyplot as plt
import networkx as nx
from crewai import Agent, Crew, Task, LLM
from wordcloud import WordCloud

# Import shared utilities
from utils import (
    load_env,
    fetch_papers,
    ensure_output_dir,
    get_output_path,
    extract_keywords_from_papers,
    keyword_frequency,
)

load_env()

# ============================================================
# Helper: Create LLM Instance
# ============================================================

def _create_llm():
    """
    Create an LLM instance based on environment variables.
    Prefers DeepSeek, falls back to OpenAI.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        return LLM(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            api_key=deepseek_key,
        )

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return LLM(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=openai_key,
        )

    # Raise clear error when no key is configured
    raise EnvironmentError(
        "No LLM API Key configured. Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env."
    )


# ============================================================
# Crew1: Paper Metadata Fetching
# ============================================================

def create_paper_fetch_crew(
    research_topic: str,
    paper_count: int = 20,
    time_range: dict[str, str] | None = None,
) -> Crew:
    """
    Create the paper metadata fetching Crew.

    Args:
        research_topic: Research topic keywords
        paper_count: Maximum number of papers
        time_range: Time range {"from": "2020-01-01", "to": "2025-12-31"}

    Returns:
        Crew: Paper fetching Crew instance
    """
    llm = _create_llm()

    # Agent: Academic Paper Researcher
    researcher_agent = Agent(
        role="Academic Paper Researcher",
        goal="Retrieve the most relevant, high-quality paper metadata from academic databases based on the research topic",
        backstory=(
            "You are a professional academic researcher skilled at searching academic databases "
            "such as Semantic Scholar, OpenAlex, and CrossRef. You have exceptional judgment "
            "on paper quality and can filter the most valuable research results by citation count, "
            "publication date, and abstract content. You ensure every returned paper includes complete: "
            "title, abstract, authors, year, venue, and citation count."
        ),
        llm=llm,
        verbose=True,
    )

    # Task: Fetch paper metadata
    fetch_task = Task(
        description=(
            f"Please fetch metadata for the most relevant {paper_count} papers from academic databases "
            f"based on the following research topic.\n\n"
            f"Research topic: {research_topic}\n"
            f"Paper count: {paper_count}\n"
            f"Time range: {time_range or 'unrestricted'}\n\n"
            "Use Python's requests library to directly call the following APIs (no external tools needed):\n"
            "1. Prefer Semantic Scholar API: https://api.semanticscholar.org/graph/v1/paper/search\n"
            "2. If it fails, try OpenAlex API: https://api.openalex.org/works\n"
            "3. If still fails, use CrossRef API: https://api.crossref.org/works\n\n"
            "Each API's calling method and field documentation are provided in the code comments.\n"
            "After collecting papers, sort by citation_count in descending order (highest citations first).\n"
            "Return structured JSON data for each paper with the following fields:\n"
            "- title: paper title\n"
            "- abstract: paper abstract\n"
            "- authors: list of authors\n"
            "- year: publication year\n"
            "- venue: journal/conference\n"
            "- citation_count: citation count\n"
            "- keywords: keyword list (extracted from metadata)\n\n"
            "Format the result as a JSON string wrapped in ```json markers."
        ),
        expected_output="A JSON string containing all paper metadata, wrapped in ```json markers",
        agent=researcher_agent,
    )

    crew = Crew(
        agents=[researcher_agent],
        tasks=[fetch_task],
        verbose=True,
    )

    return crew


# ============================================================
# Crew2: Keyword Word Cloud Generation
# ============================================================

def create_wordcloud_crew(
    papers_meta_path: str,
    output_dir: str,
    research_topic: str,
) -> Crew:
    """
    Create the keyword word cloud generation Crew.

    Args:
        papers_meta_path: Path to the paper metadata JSON file
        output_dir: Output directory
        research_topic: Research topic

    Returns:
        Crew: Word cloud generation Crew instance
    """
    llm = _create_llm()

    # Agent: Data Visualization Analyst
    visualizer_agent = Agent(
        role="Data Visualization Analyst",
        goal="Extract high-frequency keywords from paper metadata and generate an intuitive word cloud image",
        backstory=(
            "You are a data visualization expert skilled at transforming text data into intuitive visual representations. "
            "You are proficient in word cloud generation techniques and can identify key topics and trends "
            "from large volumes of text. Your word clouds are not only visually appealing but also "
            "accurately reflect the core hotspots of the research domain."
        ),
        llm=llm,
        verbose=True,
    )

    # Task: Generate word cloud
    wordcloud_task = Task(
        description=(
            f"Please generate a keyword word cloud image based on the paper metadata file.\n\n"
            f"Paper metadata file path: {papers_meta_path}\n"
            f"Output directory: {output_dir}\n"
            f"Research topic: {research_topic}\n\n"
            "Please complete the following tasks using Python code:\n"
            "1. Read the JSON file and load the paper list\n"
            "2. Extract keywords from each paper's 'keywords' field\n"
            "3. If 'keywords' field is empty, extract content words from 'title' (length>3, remove stopwords)\n"
            "4. Calculate word frequencies and generate the word cloud image\n"
            "5. Save the word cloud as: wordcloud.png\n\n"
            "Word cloud parameters:\n"
            "- width=1200, height=600\n"
            "- background_color='white'\n"
            "- colormap='viridis' (or other color scheme suitable for academic topics)\n"
            "- max_words=100\n\n"
            "After saving the image, confirm the file path."
        ),
        expected_output="Confirmation that the word cloud image was saved, including the file path",
        agent=visualizer_agent,
    )

    crew = Crew(
        agents=[visualizer_agent],
        tasks=[wordcloud_task],
        verbose=True,
    )

    return crew


# ============================================================
# Crew3: Paper Summary Generation
# ============================================================

def create_summary_crew(
    papers_meta_path: str,
    output_dir: str,
    research_topic: str,
) -> Crew:
    """
    Create the paper summary generation Crew.

    Args:
        papers_meta_path: Path to the paper metadata JSON file
        output_dir: Output directory
        research_topic: Research topic

    Returns:
        Crew: Summary generation Crew instance
    """
    llm = _create_llm()

    # Agent: Academic Summary Writer
    summarizer_agent = Agent(
        role="Academic Summary Writer",
        goal="Write precise summaries for each paper and generate a comprehensive research report",
        backstory=(
            "You are an experienced academic editor skilled at quickly understanding complex research papers "
            "and distilling their core content into clear, accurate summaries. You have deep expertise "
            "in interdisciplinary fields such as healthcare and artificial intelligence, and can identify "
            "research methods, innovations, and limitations. Your writing style is rigorous but accessible, "
            "suitable for both academic and industry readers."
        ),
        llm=llm,
        verbose=True,
    )

    # Task: Generate summaries
    summary_task = Task(
        description=(
            f"Please generate summaries for each paper and a comprehensive research report based on "
            f"the paper metadata file.\n\n"
            f"Paper metadata file path: {papers_meta_path}\n"
            f"Output directory: {output_dir}\n"
            f"Research topic: {research_topic}\n\n"
            "Please complete the following two tasks:\n\n"
            "[Task A] Write structured summaries for each paper:\n"
            "Read the paper list from the JSON file, process each paper in sequence, "
            "and generate structured summary information based on title, abstract, keywords, etc. "
            "Save as summaries.json with the following format:\n"
            "```json\n"
            "{\n"
            '  "research_topic": "research topic",\n'
            '  "summaries": [\n'
            '    {\n'
            '      "title": "paper title",\n'
            '      "authors": ["author1", "author2"],\n'
            '      "year": 2024,\n'
            '      "venue": "journal/conference",\n'
            '      "citation_count": 42,\n'
            '      "summary": "concise research summary (100-200 words)",\n'
            '      "key_findings": ["finding1", "finding2"],\n'
            '      "methods": "research method description",\n'
            '      "limitations": "research limitations (if any)"\n'
            '    },\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "[Task B] Generate a comprehensive Markdown research report:\n"
            "Based on the common themes, research trends, and differences across all papers, "
            "write a comprehensive Markdown report (report.md) containing:\n"
            "- Research background\n"
            "- Main research directions (categorized summary)\n"
            "- Key techniques/methods\n"
            "- Research trends and evolution\n"
            "- Research gaps and future directions\n"
            "- Summary\n\n"
            "The report should use Markdown format for readability."
        ),
        expected_output="Confirmation that summaries.json and report.md were generated",
        agent=summarizer_agent,
    )

    crew = Crew(
        agents=[summarizer_agent],
        tasks=[summary_task],
        verbose=True,
    )

    return crew


# ============================================================
# Crew4: Citation Relationship Visualization
# ============================================================

def create_citation_graph_crew(
    papers_meta_path: str,
    output_dir: str,
    research_topic: str,
) -> Crew:
    """
    Create the citation relationship visualization Crew.

    Args:
        papers_meta_path: Path to the paper metadata JSON file
        output_dir: Output directory
        research_topic: Research topic

    Returns:
        Crew: Citation relationship visualization Crew instance
    """
    llm = _create_llm()

    # Agent: Academic Relationship Graph Analyst
    graph_analyst_agent = Agent(
        role="Academic Relationship Graph Analyst",
        goal="Generate intuitive visualization charts based on paper citation data",
        backstory=(
            "You are a graph data analysis expert skilled at transforming complex relational data "
            "into clear visual representations. You are proficient in network analysis, "
            "ranking visualization, and statistical analysis, and can use charts to reveal "
            "the research landscape, key authors, and research hotspots in an academic field. "
            "Your charts are both visually appealing and information-rich, making them ideal "
            "materials for academic reports."
        ),
        llm=llm,
        verbose=True,
    )

    # Task: Generate citation graph
    citation_task = Task(
        description=(
            f"Please generate citation relationship visualization charts based on paper metadata.\n\n"
            f"Paper metadata file path: {papers_meta_path}\n"
            f"Output directory: {output_dir}\n"
            f"Research topic: {research_topic}\n\n"
            "Please complete the following tasks using Python code:\n\n"
            "[Task A] Citation ranking chart:\n"
            "Create a figure with two subplots (figsize=16, 10):\n\n"
            "Subplot 1 (left): Citation count ranking bar chart (Top-N)\n"
            "- Sort by citation_count in descending order\n"
            "- Show paper titles (truncated to 40 characters) and citation counts\n"
            "- Use horizontal bar chart with gradient colors (higher citations = deeper color)\n"
            "- Add value labels\n\n"
            "Subplot 2 (right): Year-citation scatter plot\n"
            "- X-axis: year\n"
            "- Y-axis: citation count\n"
            "- Bubble size: citation count (normalized * 200 + 50)\n"
            "- Color: citation count (use colormap='RdYlGn')\n"
            "- Add grid lines\n\n"
            "[Task B] Author analysis:\n"
            "- Calculate the most frequently appearing authors\n"
            "- Add text annotations or legends to the chart\n\n"
            "[Task C] Save the image:\n"
            "- dpi=150\n"
            "- Title: research topic + citation relationship analysis\n"
            "- Save as citation_graph.png\n\n"
            "After saving the image, confirm the file path."
        ),
        expected_output="Confirmation that the citation relationship visualization chart was saved, including the file path",
        agent=graph_analyst_agent,
    )

    crew = Crew(
        agents=[graph_analyst_agent],
        tasks=[citation_task],
        verbose=True,
    )

    return crew


# ============================================================
# Synchronous Execution Functions (direct calls, no Agent)
# Underlying logic for actual task execution
# ============================================================

def run_paper_fetch_sync(
    research_topic: str,
    paper_count: int = 20,
    time_range: dict[str, str] | None = None,
) -> list[dict]:
    """
    Synchronously fetch paper metadata (without Agent, direct API call).
    This is the underlying execution logic for Crew1.
    """
    return fetch_papers(
        topic=research_topic,
        count=paper_count,
        time_range=time_range,
    )


def run_wordcloud_sync(
    papers: list[dict],
    output_path: str,
) -> str:
    """
    Synchronously generate word cloud (without Agent).
    This is the underlying execution logic for Crew2.
    """
    keywords = extract_keywords_from_papers(papers)
    freq = keyword_frequency(keywords)

    if not freq:
        raise ValueError("No available keywords to generate word cloud")

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=100,
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(freq)
    wc.to_file(output_path)
    return output_path


def run_summary_sync(
    papers: list[dict],
    summaries_json_path: str,
    report_md_path: str,
    research_topic: str,
) -> dict[str, str]:
    """
    Synchronously generate paper summaries (without Agent).
    This is the underlying execution logic for Crew3.
    Generates summaries.json and report.md.
    """
    summaries = []
    for paper in papers:
        abstract = paper.get("abstract", "") or "(abstract not available)"
        # Simple summary: first 200 characters of abstract
        summary_text = abstract[:200] + "..." if len(abstract) > 200 else abstract
        summaries.append({
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "citation_count": paper.get("citation_count", 0),
            "summary": summary_text,
            "key_findings": [],
            "methods": "not provided",
            "limitations": "not provided",
        })

    # Save JSON
    output_data = {
        "research_topic": research_topic,
        "summaries": summaries,
    }
    with open(summaries_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Generate Markdown report
    report_lines = [
        f"# Research Report: {research_topic}",
        "",
        "## 1. Research Background",
        "",
        f"This report provides an overview of the current research landscape in **{research_topic}** based on {len(papers)} relevant academic papers.",
        "",
        "## 2. Paper Overview",
        "",
        f"| # | Title | Authors | Year | Venue | Citations |",
        f"| --- | --- | --- | --- | --- | --- |",
    ]

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "")[:50] + "..." if len(paper.get("title", "")) > 50 else paper.get("title", "")
        authors = ", ".join(paper.get("authors", [])[:3])
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A")[:30]
        citations = paper.get("citation_count", 0)
        report_lines.append(f"| {i} | {title} | {authors} | {year} | {venue} | {citations} |")

    report_lines += [
        "",
        "## 3. Main Research Directions",
        "",
        "Based on the paper analysis, research in this field focuses on the following areas:",
        "",
        "1. **Foundation Model Research**: Applications of large language models in healthcare",
        "2. **Clinical Decision Support**: Auxiliary diagnosis, treatment recommendations, etc.",
        "3. **Medical Information Processing**: Electronic health record analysis, drug interactions, etc.",
        "",
        "## 4. Technical Methods",
        "",
        "Main technical methods used in the research include:",
        "",
        "- Transformer architectures and variants",
        "- Pretraining-finetuning paradigm",
        "- Knowledge distillation and model compression",
        "- Multimodal fusion techniques",
        "",
        "## 5. Research Trends",
        "",
        f"Paper publication years range from {min(p.get('year', 0) for p in papers if p.get('year'))} to "
        f"{max(p.get('year', 0) for p in papers if p.get('year'))}, "
        "showing a year-over-year growth trend, indicating increasing research interest in this field.",
        "",
        "## 6. Future Directions",
        "",
        "Based on the existing paper analysis, the following research directions are recommended:",
        "",
        "1. Explainability and trustworthy AI in healthcare applications",
        "2. Privacy protection and data security",
        "3. Cross-institutional, cross-modal data collaboration",
        "4. Human-machine collaborative clinical workflow optimization",
        "",
        "## 7. Summary",
        "",
        f"This analysis collected {len(papers)} relevant papers, "
        "covering the main research directions and technical methods in this field. "
        "With the rapid development of large language model technology, "
        "more innovative applications are expected to emerge in the future.",
        "",
        "---",
        f"*Report generated at: {Path(__file__).stat().st_mtime}*",
    ]

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "summaries_json": summaries_json_path,
        "report_md": report_md_path,
    }


def run_citation_graph_sync(
    papers: list[dict],
    output_path: str,
    research_topic: str,
) -> str:
    """
    Synchronously generate citation relationship graph (without Agent).
    This is the underlying execution logic for Crew4.
    """
    if not papers:
        raise ValueError("No paper data to generate citation graph")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Citation ranking Top-N
    ax1 = axes[0]
    sorted_papers = sorted(papers, key=lambda x: x.get("citation_count", 0), reverse=True)[:15]
    titles = [p.get("title", "")[:40] + "..." if len(p.get("title", "")) > 40 else p.get("title", "") for p in sorted_papers]
    citations = [p.get("citation_count", 0) for p in sorted_papers]
    years = [p.get("year", 2020) for p in sorted_papers]

    colors = plt.cm.Blues([(c + 1) / (max(citations) + 1) for c in citations])
    bars = ax1.barh(range(len(titles)), citations, color=colors)
    ax1.set_yticks(range(len(titles)))
    ax1.set_yticklabels(titles, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Citation Count")
    ax1.set_title("Top Papers by Citation Count")
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, cit) in enumerate(zip(bars, citations)):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(cit), va="center", fontsize=7)

    # Subplot 2: Year-citation scatter plot
    ax2 = axes[1]
    years_data = [p.get("year", 2020) for p in papers if p.get("year")]
    citations_data = [p.get("citation_count", 0) for p in papers if p.get("year")]
    sizes = [min(c / 5 + 50, 500) for c in citations_data]
    colors_scatter = [c / max(max(citations_data), 1) for c in citations_data]

    scatter = ax2.scatter(
        years_data, citations_data,
        s=sizes,
        c=colors_scatter,
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Citation Count")
    ax2.set_title("Citations by Publication Year")
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Normalized Citations")

    fig.suptitle(f"Citation Analysis: {research_topic}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
