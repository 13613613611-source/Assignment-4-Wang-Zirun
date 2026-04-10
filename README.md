# Research Paper Pipeline

> A multi-stage intelligent agent orchestration pipeline implemented with CrewAI for automated academic paper retrieval, analysis, and visualization.

---

## 1. Project Overview

This pipeline implements an end-to-end academic paper research automation flow:

1. **Paper Retrieval**: Progressive fetching from Semantic Scholar -> OpenAlex -> CrossRef
2. **Word Cloud Generation**: Extract keywords and generate a visual word cloud
3. **Paper Summaries**: Generate per-paper summaries + comprehensive Markdown report
4. **Citation Visualization**: Generate citation ranking charts and year-distribution scatter plots

## 2. Tech Stack

- **Orchestration Framework**: CrewAI 0.80+
- **LLM**: DeepSeek / OpenAI (configured via `.env`)
- **APIs**: Semantic Scholar, OpenAlex, CrossRef (progressive fallback)
- **Visualization**: matplotlib, wordcloud, networkx
- **Language**: Python 3.10+

## 3. Directory Structure

```
├── .env                 # API keys (do not commit)
├── .gitignore
├── config.json          # Research topic configuration
├── crews.py             # 4 Crew definitions
├── pipeline.py          # Python orchestrator
├── utils.py             # Shared utility functions
├── requirements.txt
├── checkpoint.json      # Generated at runtime
├── README.md
├── prompt-log.md
└── output/             # Output directory
    └── {research_topic}/
        ├── papers_meta.json
        ├── wordcloud.png
        ├── summaries.json
        ├── report.md
        ├── citation_graph.png
        └── final_report.md
```

## 4. Quick Start

### 4.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.2 Configure API Keys

Create a `.env` file in the project root:

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

### 4.3 Modify Research Topic

Edit `config.json`:

```json
{
  "research_topic": "large language model in healthcare",
  "paper_count": 20,
  "time_range": {
    "from": "2020-01-01",
    "to": "2025-12-31"
  },
  "checkpoint_path": "checkpoint.json",
  "output_dir": "./output"
}
```

### 4.4 Run the Pipeline

```bash
python pipeline.py
```

## 5. Resume from Checkpoint

The pipeline uses `checkpoint.json` to record the execution status of each stage.

### Normal Run

```bash
python pipeline.py
```

- Completed stages are automatically skipped
- Interrupted stages resume from the checkpoint

### Test Resume

Remove a specific stage from the checkpoint to verify resume functionality:

```bash
# Clear stage 2 (word cloud), rerun from stage 2
python pipeline.py --clear-stage stage_2_wordcloud

# Clear stage 3 (summary), rerun from stage 3
python pipeline.py --clear-stage stage_3_summary
```

## 6. Stages and Crew Responsibilities

| Stage | Crew | Input | Output |
|-------|------|-------|--------|
| 1 | PaperFetchCrew | research_topic | papers_meta.json |
| 2 | WordCloudCrew | papers_meta.json | wordcloud.png |
| 3 | SummaryCrew | papers_meta.json | summaries.json + report.md |
| 4 | CitationGraphCrew | papers_meta.json | citation_graph.png |

## 7. Challenges and Solutions

### Challenge 1: API Reliability

**Problem**: A single academic API may fail due to network issues or rate limiting.

**Solution**: Implemented a three-API progressive fallback strategy (Semantic Scholar -> OpenAlex -> CrossRef). If any API fails, the next one is automatically tried; only when all three fail does it report an error.

### Challenge 2: LLM Agent Execution Instability

**Problem**: LLM Agents may not accurately execute Python code or parse JSON.

**Solution**: Adopted a dual-track strategy (Agent + synchronous function). When the Agent fails, it automatically falls back to directly calling the underlying synchronous function, ensuring output files are always generated.

### Challenge 3: Checkpoint Resume and Data Consistency

**Problem**: Stages have dependencies (e.g., stage 2 needs stage 1 output).

**Solution**: Each stage checks whether its prerequisite stage succeeded first; if not, it throws a clear error, ensuring no stage executes with missing inputs.

## 8. Reliability Features

- **Retry Mechanism**: Up to 3 retries for transient errors, exponential backoff (1s -> 2s -> 4s)
- **Checkpoint Persistence**: Written to JSON file immediately after each stage succeeds
- **Graceful Failure**: Raises RuntimeError after retries exhausted, preserving stage logs
- **Resume from Checkpoint**: Automatically skips completed stages on restart

---

> Version: 1.0 | Date: 2026-04-10
