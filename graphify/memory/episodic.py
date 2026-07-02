"""
episodic.py — Append-only query log (Phase 5 memory foundation).

Every call to graphify ask / graphify swarm appends one JSON line here.
This log is the raw material for:
  - Token usage analytics
  - Future semantic answer caching
  - Future procedural memory extraction

Schema per line:
  {
    "ts":            "2026-06-30T18:00:00Z",   # ISO-8601 UTC
    "query":         "...",                     # user question
    "repos_searched": ["pipeline", "skills"],   # repos that were searched
    "chunks_used":   8,                         # how many chunks reached the LLM
    "top_score":     0.91,                      # best similarity score
    "min_score":     0.86,                      # worst score that passed threshold
    "threshold":     0.85,                      # score_threshold used
    "provider":      "databricks",              # llm provider
    "model":         "databricks-claude-sonnet-4-6",
    "answer_chars":  512,                       # rough answer length
    "latency_s":     3.2                        # total wall time
  }
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_LOG_FILE = Path("graphify-out") / "memory" / "episodic.jsonl"


def log_query(
    query:          str,
    repos_searched: list[str],
    chunks_used:    int,
    top_score:      float,
    min_score:      float,
    threshold:      float,
    provider:       str,
    model:          str,
    answer_chars:   int,
    latency_s:      float,
) -> None:
    """Append one query record to the episodic log. Silent on failure."""
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query":          query,
            "repos_searched": repos_searched,
            "chunks_used":    chunks_used,
            "top_score":      round(top_score, 4) if top_score else 0.0,
            "min_score":      round(min_score, 4) if min_score else 0.0,
            "threshold":      threshold,
            "provider":       provider,
            "model":          model,
            "answer_chars":   answer_chars,
            "latency_s":      round(latency_s, 2),
        }
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # never break the main flow due to logging


def query_stats(last_n: int = 100) -> dict:
    """Return aggregate stats from the last N queries."""
    if not _LOG_FILE.exists():
        return {}

    lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-last_n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not records:
        return {}

    providers = {}
    for r in records:
        p = r.get("provider", "unknown")
        providers[p] = providers.get(p, 0) + 1

    return {
        "total_queries":    len(records),
        "avg_latency_s":    round(sum(r.get("latency_s", 0) for r in records) / len(records), 2),
        "avg_chunks_used":  round(sum(r.get("chunks_used", 0) for r in records) / len(records), 1),
        "avg_top_score":    round(sum(r.get("top_score", 0) for r in records) / len(records), 3),
        "providers_used":   providers,
    }
