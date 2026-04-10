# ============================================================
# Research Paper Pipeline -- Orchestrator
# CrewAI Multi-Crew Orchestration Pipeline
# ============================================================
"""
Python orchestrator: sequentially executes 4 Crews for end-to-end research paper analysis.

Features:
  - Sequential execution: Crew1 -> Crew2 -> Crew3 -> Crew4
  - Previous Crew output feeds into the next Crew as input
  - Per-stage success/failure logging
  - Transient errors: up to 3 retries with exponential backoff
  - Checkpoint written after each stage completes
  - Reads checkpoint on startup; completed stages are skipped automatically (resume)

Usage:
  python pipeline.py

  Resume: just rerun; completed stages are skipped.
  Test resume: python pipeline.py --clear-stage stage_2_wordcloud (removes specified stage)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils import (
    load_env,
    load_checkpoint,
    save_checkpoint,
    ensure_output_dir,
    get_output_path,
)
from crews import (
    create_paper_fetch_crew,
    create_wordcloud_crew,
    create_summary_crew,
    create_citation_graph_crew,
    run_paper_fetch_sync,
    run_wordcloud_sync,
    run_summary_sync,
    run_citation_graph_sync,
)

load_env()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global stage definitions (consistent with checkpoint key names)
STAGES = [
    ("stage_1_meta", "Paper Metadata Fetching"),
    ("stage_2_wordcloud", "Keyword Word Cloud Generation"),
    ("stage_3_summary", "Paper Summary Generation"),
    ("stage_4_citation_graph", "Citation Relationship Visualization"),
]

# Maximum retry attempts (meets REQ-RF-01)
MAX_RETRIES = 3


# ============================================================
# Utility Functions
# ============================================================

def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """
    Load and validate the configuration file (simple JSON schema check).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Simple validation for required fields
    required_keys = ["research_topic", "paper_count"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config file missing required field: {key}")

    logger.info(f"Config file loaded: {config_path}")
    return config


def log_stage_status(stage_name: str, status: str, message: str = "") -> None:
    """
    Log stage execution status.
    Meets REQ-OR-04: at minimum records started / completed / failed.
    """
    status_icons = {
        "started": ">",
        "completed": "+",
        "failed": "X",
        "skipped": "o",
    }
    icon = status_icons.get(status, "*")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"{icon} [{timestamp}] [{status.upper()}] {stage_name}"
    if message:
        log_msg += f" -- {message}"

    if status == "started":
        logger.info(log_msg)
    elif status == "completed":
        logger.info(log_msg)
    elif status == "failed":
        logger.error(log_msg)
    else:
        logger.info(log_msg)


def retry_with_backoff(
    func,
    *args,
    stage_name: str = "Unknown Stage",
    **kwargs,
) -> Any:
    """
    Retry with exponential backoff.

    Retries up to MAX_RETRIES times, with delay of 2^attempt seconds between tries.
    Meets REQ-RF-01, REQ-RF-02.

    Args:
        func: Function to execute
        stage_name: Stage name (for logging)
        *args, **kwargs: Arguments passed to func

    Returns:
        Return value of func

    Raises:
        RuntimeError: Raised when retries are exhausted
    """
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"{stage_name} succeeded after {attempt} retries")
            return result

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = 2 ** attempt  # Exponential backoff: 1s -> 2s -> 4s
                logger.warning(
                    f"{stage_name} execution failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}, "
                    f"retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{stage_name} still failed after {MAX_RETRIES} retries: {e}"
                )
                raise RuntimeError(
                    f"{stage_name} retries exhausted. Original error: {e}"
                ) from e

    # Should not reach here, but just in case
    raise RuntimeError(f"{stage_name} execution failed: {last_error}")


# ============================================================
# Per-Stage Execution Functions
# ============================================================

def run_stage_1_meta(
    state: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Stage 1: Paper metadata fetching.
    Uses a Crew + direct API hybrid execution strategy.
    """
    stage_key = "stage_1_meta"
    research_topic = config["research_topic"]
    paper_count = config.get("paper_count", 20)
    time_range = config.get("time_range")

    # Skip already completed stage
    if stage_key in state and state[stage_key].get("success"):
        log_stage_status("Paper Metadata Fetching", "skipped", "found in checkpoint, skipping")
        return state[stage_key]

    log_stage_status("Paper Metadata Fetching", "started")

    # Create output directory
    topic_output_dir = ensure_output_dir(output_dir, research_topic)
    papers_meta_path = os.path.join(topic_output_dir, "papers_meta.json")

    def _fetch():
        # Try LLM Agent first (via Crew)
        # If Agent fails, fall back to direct API call
        try:
            crew = create_paper_fetch_crew(
                research_topic=research_topic,
                paper_count=paper_count,
                time_range=time_range,
            )
            result = crew.kickoff(inputs={
                "research_topic": research_topic,
                "paper_count": str(paper_count),
            })
            raw_output = str(result)
            # Try to parse JSON from Agent output
            papers = _parse_json_from_output(raw_output)
            if papers:
                return papers
        except Exception as e:
            logger.warning(f"Crew1 Agent execution failed, falling back to direct API call: {e}")

        # Direct API call (synchronous version)
        return run_paper_fetch_sync(
            research_topic=research_topic,
            paper_count=paper_count,
            time_range=time_range,
        )

    papers = retry_with_backoff(_fetch, stage_name="Paper Metadata Fetching")

    # Sort by citation count descending
    papers = sorted(papers, key=lambda x: x.get("citation_count", 0), reverse=True)

    # Limit to paper count
    papers = papers[:paper_count]

    # Save to file
    with open(papers_meta_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    result = {
        "papers": papers,
        "papers_meta_path": papers_meta_path,
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "count": len(papers),
    }

    log_stage_status("Paper Metadata Fetching", "completed", f"retrieved {len(papers)} papers")
    return result


def run_stage_2_wordcloud(
    state: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Stage 2: Keyword word cloud generation.
    """
    stage_key = "stage_2_wordcloud"
    research_topic = config["research_topic"]

    # Check prerequisite stage
    if "stage_1_meta" not in state or not state["stage_1_meta"].get("success"):
        raise RuntimeError("Stage 1 (paper metadata) not completed, cannot generate word cloud")

    papers = state["stage_1_meta"]["papers"]
    topic_output_dir = ensure_output_dir(output_dir, research_topic)
    wordcloud_path = os.path.join(topic_output_dir, "wordcloud.png")

    # Skip already completed stage
    if stage_key in state and state[stage_key].get("success"):
        log_stage_status("Keyword Word Cloud Generation", "skipped", "found in checkpoint, skipping")
        return state[stage_key]

    log_stage_status("Keyword Word Cloud Generation", "started")

    def _generate():
        # Try using Agent
        try:
            crew = create_wordcloud_crew(
                papers_meta_path=state["stage_1_meta"]["papers_meta_path"],
                output_dir=topic_output_dir,
                research_topic=research_topic,
            )
            result = crew.kickoff(inputs={
                "papers_meta_path": state["stage_1_meta"]["papers_meta_path"],
                "output_dir": topic_output_dir,
                "research_topic": research_topic,
            })
            raw_output = str(result)
            if "wordcloud.png" in raw_output or "success" in raw_output.lower() or "saved" in raw_output.lower():
                return wordcloud_path
        except Exception as e:
            logger.warning(f"Crew2 Agent execution failed, falling back to direct generation: {e}")

        # Direct word cloud generation (synchronous version)
        return run_wordcloud_sync(papers, wordcloud_path)

    result_path = retry_with_backoff(_generate, stage_name="Keyword Word Cloud Generation")

    result = {
        "path": result_path,
        "success": True,
        "timestamp": datetime.now().isoformat(),
    }

    log_stage_status("Keyword Word Cloud Generation", "completed", f"word cloud saved: {result_path}")
    return result


def run_stage_3_summary(
    state: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Stage 3: Paper summary generation.
    Outputs summaries.json + report.md.
    """
    stage_key = "stage_3_summary"
    research_topic = config["research_topic"]

    # Check prerequisite stage
    if "stage_1_meta" not in state or not state["stage_1_meta"].get("success"):
        raise RuntimeError("Stage 1 (paper metadata) not completed, cannot generate summary")

    papers = state["stage_1_meta"]["papers"]
    topic_output_dir = ensure_output_dir(output_dir, research_topic)
    summaries_json_path = os.path.join(topic_output_dir, "summaries.json")
    report_md_path = os.path.join(topic_output_dir, "report.md")

    # Skip already completed stage
    if stage_key in state and state[stage_key].get("success"):
        log_stage_status("Paper Summary Generation", "skipped", "found in checkpoint, skipping")
        return state[stage_key]

    log_stage_status("Paper Summary Generation", "started")

    def _summarize():
        # Try using Agent
        try:
            crew = create_summary_crew(
                papers_meta_path=state["stage_1_meta"]["papers_meta_path"],
                output_dir=topic_output_dir,
                research_topic=research_topic,
            )
            result = crew.kickoff(inputs={
                "papers_meta_path": state["stage_1_meta"]["papers_meta_path"],
                "output_dir": topic_output_dir,
                "research_topic": research_topic,
            })
            raw_output = str(result)
            # Even if Agent output is imperfect, try to ensure output files exist
            if "summaries.json" in raw_output or "report.md" in raw_output:
                return {
                    "summaries_json": summaries_json_path,
                    "report_md": report_md_path,
                }
        except Exception as e:
            logger.warning(f"Crew3 Agent execution failed, falling back to direct generation: {e}")

        # Direct summary generation (synchronous version)
        return run_summary_sync(
            papers=papers,
            summaries_json_path=summaries_json_path,
            report_md_path=report_md_path,
            research_topic=research_topic,
        )

    result = retry_with_backoff(_summarize, stage_name="Paper Summary Generation")

    # Ensure files exist
    for key in ["summaries_json", "report_md"]:
        if result.get(key) and not os.path.exists(result[key]):
            # If file does not exist, regenerate with fallback function
            fallback = run_summary_sync(
                papers=papers,
                summaries_json_path=summaries_json_path,
                report_md_path=report_md_path,
                research_topic=research_topic,
            )
            result = fallback
            break

    output_result = {
        "summaries_json": result.get("summaries_json", summaries_json_path),
        "report_md": result.get("report_md", report_md_path),
        "success": True,
        "timestamp": datetime.now().isoformat(),
    }

    log_stage_status("Paper Summary Generation", "completed", "summaries.json + report.md generated")
    return output_result


def run_stage_4_citation_graph(
    state: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Stage 4: Citation relationship visualization.
    Outputs citation_graph.png.
    """
    stage_key = "stage_4_citation_graph"
    research_topic = config["research_topic"]

    # Check prerequisite stage
    if "stage_1_meta" not in state or not state["stage_1_meta"].get("success"):
        raise RuntimeError("Stage 1 (paper metadata) not completed, cannot generate citation graph")

    papers = state["stage_1_meta"]["papers"]
    topic_output_dir = ensure_output_dir(output_dir, research_topic)
    citation_graph_path = os.path.join(topic_output_dir, "citation_graph.png")

    # Skip already completed stage
    if stage_key in state and state[stage_key].get("success"):
        log_stage_status("Citation Relationship Visualization", "skipped", "found in checkpoint, skipping")
        return state[stage_key]

    log_stage_status("Citation Relationship Visualization", "started")

    def _visualize():
        # Try using Agent
        try:
            crew = create_citation_graph_crew(
                papers_meta_path=state["stage_1_meta"]["papers_meta_path"],
                output_dir=topic_output_dir,
                research_topic=research_topic,
            )
            result = crew.kickoff(inputs={
                "papers_meta_path": state["stage_1_meta"]["papers_meta_path"],
                "output_dir": topic_output_dir,
                "research_topic": research_topic,
            })
            raw_output = str(result)
            if "citation_graph" in raw_output or "success" in raw_output.lower():
                return citation_graph_path
        except Exception as e:
            logger.warning(f"Crew4 Agent execution failed, falling back to direct generation: {e}")

        # Direct citation graph generation (synchronous version)
        return run_citation_graph_sync(
            papers=papers,
            output_path=citation_graph_path,
            research_topic=research_topic,
        )

    result_path = retry_with_backoff(_visualize, stage_name="Citation Relationship Visualization")

    result = {
        "path": result_path,
        "success": True,
        "timestamp": datetime.now().isoformat(),
    }

    log_stage_status("Citation Relationship Visualization", "completed", f"citation graph saved: {result_path}")
    return result


def _parse_json_from_output(output: str) -> list[dict]:
    """
    Parse JSON data from Agent output.
    Handles various formats (with/without markdown code block markers, etc.).
    """
    # Remove markdown code block markers
    output_clean = output.strip()
    if output_clean.startswith("```json"):
        output_clean = output_clean[7:]
    if output_clean.startswith("```"):
        output_clean = output_clean[3:]
    if output_clean.endswith("```"):
        output_clean = output_clean[:-3]
    output_clean = output_clean.strip()

    try:
        data = json.loads(output_clean)
        # If it's a dict with a "papers" key
        if isinstance(data, dict) and "papers" in data:
            return data["papers"]
        # If it's a direct list of papers
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array using regex
    import re
    match = re.search(r'\[[\s\S]*\]', output_clean)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


# ============================================================
# Final Report Generation
# ============================================================

def generate_final_report(
    state: dict,
    config: dict,
    output_dir: str,
) -> str:
    """
    Generate the final consolidated report (final_report.md).
    Summarizes all stage outputs.
    """
    research_topic = config["research_topic"]
    topic_output_dir = ensure_output_dir(output_dir, research_topic)
    final_report_path = os.path.join(topic_output_dir, "final_report.md")

    # Collect stage outputs
    meta_info = state.get("stage_1_meta", {})
    wc_info = state.get("stage_2_wordcloud", {})
    summary_info = state.get("stage_3_summary", {})
    cg_info = state.get("stage_4_citation_graph", {})

    papers = meta_info.get("papers", [])
    paper_count = len(papers)

    # Statistics
    years = [p.get("year") for p in papers if p.get("year")]
    total_citations = sum(p.get("citation_count", 0) for p in papers)
    avg_citations = total_citations / paper_count if paper_count > 0 else 0
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    report_lines = [
        f"# Research Paper Pipeline -- Final Report",
        "",
        f"**Research Topic**: {research_topic}",
        f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"This report was auto-generated by the Research Paper Pipeline, which completed the full flow from academic database paper retrieval to multi-dimensional analysis.",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Papers Collected | {paper_count} |",
        f"| Paper Year Range | {year_range} |",
        f"| Total Citations | {total_citations} |",
        f"| Average Citations | {avg_citations:.1f} |",
        f"| Time Range | {config.get('time_range', {}).get('from', 'N/A')} ~ {config.get('time_range', {}).get('to', 'N/A')} |",
        "",
        "## 2. Stage Output Checklist",
        "",
        "| Stage | Status | Output File | Timestamp |",
        "| --- | --- | --- | --- |",
    ]

    for stage_key, stage_name in STAGES:
        stage_info = state.get(stage_key, {})
        success = stage_info.get("success", False)
        status_icon = "[OK] Done" if success else "[FAIL] Failed"
        timestamp = stage_info.get("timestamp", "N/A")
        output_file = "N/A"
        if stage_key == "stage_1_meta":
            output_file = meta_info.get("papers_meta_path", "N/A")
        elif stage_key == "stage_2_wordcloud":
            output_file = wc_info.get("path", "N/A")
        elif stage_key == "stage_3_summary":
            summaries = summary_info.get("summaries_json", "N/A")
            report = summary_info.get("report_md", "N/A")
            output_file = f"JSON: {summaries}, MD: {report}"
        elif stage_key == "stage_4_citation_graph":
            output_file = cg_info.get("path", "N/A")
        report_lines.append(
            f"| {stage_name} | {status_icon} | {output_file} | {timestamp} |"
        )

    report_lines += [
        "",
        "## 3. Output File Locations",
        "",
        f"All output files are saved in: `{topic_output_dir}`",
        "",
        "### 3.1 Paper Metadata",
        f"- `papers_meta.json`",
        "",
        "### 3.2 Word Cloud",
        f"- `wordcloud.png`",
        "",
        "### 3.3 Paper Summaries",
        f"- `summaries.json` (structured summary data)",
        f"- `report.md` (Markdown comprehensive report)",
        "",
        "### 3.4 Citation Relationship Visualization",
        f"- `citation_graph.png`",
        "",
        "## 4. Usage Instructions",
        "",
        "### Resume from Checkpoint",
        "The pipeline uses checkpoint.json to record the execution status of each stage.",
        "If the pipeline is interrupted, simply rerun to resume from the checkpoint:",
        "```bash",
        "python pipeline.py",
        "```",
        "",
        "### Test Resume",
        "Remove a specific stage from the checkpoint to verify resume functionality:",
        "```bash",
        "python pipeline.py --clear-stage stage_2_wordcloud",
        "```",
        "",
        "---",
        f"*Auto-generated by Research Paper Pipeline | Version 1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    with open(final_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Final report generated: {final_report_path}")
    return final_report_path


# ============================================================
# Main Orchestrator
# ============================================================

def run_pipeline(
    config_path: str = "config.json",
    checkpoint_path: str = "checkpoint.json",
    clear_stage: Optional[str] = None,
) -> dict:
    """
    Main pipeline entry point.

    Execution flow:
      1. Load config and checkpoint
      2. Sequentially execute 4 stages (Crew1 -> Crew2 -> Crew3 -> Crew4)
      3. Write checkpoint after each stage succeeds
      4. Generate final report

    Args:
        config_path: Config file path
        checkpoint_path: Checkpoint file path
        clear_stage: Optional, clears the specified stage (for testing resume)

    Returns:
        dict: Final state dictionary
    """
    logger.info("=" * 60)
    logger.info("Research Paper Pipeline Started")
    logger.info("=" * 60)

    # Step 1: Load config
    config = load_config(config_path)
    research_topic = config["research_topic"]
    output_dir = config.get("output_dir", "./output")

    logger.info(f"Research topic: {research_topic}")
    logger.info(f"Paper count: {config.get('paper_count', 20)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint file: {checkpoint_path}")

    # Step 2: Load or initialize checkpoint
    state = load_checkpoint(checkpoint_path)

    # Record research_topic in state on first run
    if "research_topic" not in state:
        state["research_topic"] = research_topic

    # Clear specified stage (for testing)
    if clear_stage:
        if clear_stage in state:
            logger.warning(f"Test mode: clearing stage {clear_stage} from checkpoint")
            del state[clear_stage]
            save_checkpoint(checkpoint_path, state)
            logger.info(f"Stage {clear_stage} cleared, resume will start from this stage")

    # Step 3: Execute stages in order
    stage_handlers = {
        "stage_1_meta": run_stage_1_meta,
        "stage_2_wordcloud": run_stage_2_wordcloud,
        "stage_3_summary": run_stage_3_summary,
        "stage_4_citation_graph": run_stage_4_citation_graph,
    }

    for stage_key, stage_name in STAGES:
        handler = stage_handlers[stage_key]

        # Skip already succeeded stages
        if stage_key in state and state[stage_key].get("success"):
            logger.info(f"Skipping completed stage: {stage_name}")
            continue

        # Execute stage
        try:
            result = handler(state=state, config=config, output_dir=output_dir)

            # Update state and save checkpoint
            state[stage_key] = result
            state["research_topic"] = research_topic
            save_checkpoint(checkpoint_path, state)
            logger.info(f"Stage {stage_name} completed, checkpoint saved")

        except Exception as e:
            log_stage_status(stage_name, "failed", str(e))
            logger.error(f"Pipeline execution failed: {e}")
            logger.error("Please check the logs and fix the issue, then rerun (resume will work)")
            raise

    # Step 4: Generate final report
    logger.info("All stages completed, generating final report...")
    final_report_path = generate_final_report(state, config, output_dir)

    logger.info("=" * 60)
    logger.info("Pipeline execution completed!")
    logger.info(f"Final report: {final_report_path}")
    logger.info("=" * 60)

    return state


# ============================================================
# Entry Point
# ============================================================

def main():
    """
    Command-line entry point.
    """
    parser = argparse.ArgumentParser(
        description="Research Paper Pipeline -- CrewAI Multi-Stage Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                    # Normal run
  python pipeline.py --clear-stage stage_2_wordcloud  # Test resume (clears specified stage)
  python pipeline.py --config config.json --checkpoint checkpoint.json
        """,
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Config file path (default: config.json)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.json",
        help="Checkpoint file path (default: checkpoint.json)",
    )
    parser.add_argument(
        "--clear-stage",
        dest="clear_stage",
        default=None,
        help="Clear specified stage from checkpoint (for resume testing), e.g. stage_2_wordcloud",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            clear_stage=args.clear_stage,
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Config error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Pipeline execution error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        logger.info("Checkpoint saved, rerun to resume from checkpoint")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
