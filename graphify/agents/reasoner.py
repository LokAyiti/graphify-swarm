"""
reasoner.py — Agent 2: synthesise retrieval results into structured findings.

With LLM  — sends merged context to Ollama, parses SUMMARY / FINDINGS /
             RECOMMENDATION sections from the response.

Without LLM — derives structured findings directly from graph metadata
              (pipeline names, activity lists, folder paths) so the swarm
              is still useful when no model is configured.
"""
from __future__ import annotations

import re
from typing import Optional

from graphify.agents.base import BaseAgent, RunContext

_SYSTEM_PROMPT = """\
You are a code-analysis agent inside a multi-agent pipeline.
Read the retrieved context carefully, then answer the task.

Respond in EXACTLY this format — no prose before the labels:

SUMMARY: <one sentence>
FINDINGS:
- <specific finding with file/function name when relevant>
- <another finding>
RECOMMENDATION: <concrete, actionable answer or next step>"""


class ReasonerAgent(BaseAgent):
    name = "reasoner"

    def __init__(self, llm=None) -> None:
        self._llm = llm   # OllamaLLM | None

    def run(self, ctx: RunContext) -> RunContext:
        if not ctx.merged_context and not ctx.graph_contexts:
            ctx.errors.append("Reasoner: no context from Retriever")
            return ctx

        if self._llm is None:
            return _rule_based_reason(ctx)

        prompt = f"Task: {ctx.task}\n\nContext:\n\n{ctx.merged_context}"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw         = self._llm.ask(messages, timeout=120)
            ctx.reasoning = raw
            _parse_llm_response(raw, ctx)
        except Exception as exc:
            ctx.errors.append(f"Reasoner LLM error: {exc}")
            # Degrade gracefully to rule-based
            _rule_based_reason(ctx)

        return ctx


# ── Rule-based fallback (no LLM) ─────────────────────────────────────────────

def _rule_based_reason(ctx: RunContext) -> RunContext:
    """Build structured findings directly from graph metadata — no LLM needed."""

    pipeline_info: list[dict] = []
    for gc in ctx.graph_contexts:
        md = gc.metadata or {}
        if md.get("pipeline_name"):
            pipeline_info.append({
                "name":       md["pipeline_name"],
                "activities": md.get("activity_names", []),
                "count":      md.get("activity_count", 0),
                "folder":     md.get("folder", ""),
                "file":       gc.file_path,
            })

    if pipeline_info:
        ctx.summary = (
            f"Found {len(pipeline_info)} pipeline(s) matching the task"
        )
        findings: list[str] = []
        for p in pipeline_info:
            line = f"{p['name']} — {p['count']} activities"
            if p["folder"]:
                line += f" · folder: {p['folder']}"
            findings.append(line)
            if p["activities"]:
                findings.append("  Activities: " + ", ".join(p["activities"][:8]))
        ctx.findings       = findings
        ctx.recommendation = "Use --llm <model> to generate AI-powered insights."
    else:
        # Fall back to top vector hits
        unique_files = list(dict.fromkeys(h.file_path for h in ctx.vector_hits))
        ctx.summary   = (
            f"Retrieved {len(ctx.vector_hits)} chunks across "
            f"{len(unique_files)} file(s)"
        )
        ctx.findings  = [
            f"[score {h.score:.3f}] {h.file_path}:{h.start_line}"
            for h in ctx.vector_hits[:6]
        ]
        ctx.recommendation = "Use --llm <model> for AI analysis."

    ctx.reasoning = "\n".join([
        f"SUMMARY: {ctx.summary}",
        "FINDINGS:",
        *[f"- {f}" for f in ctx.findings],
        f"RECOMMENDATION: {ctx.recommendation}",
    ])
    return ctx


# ── LLM response parser ───────────────────────────────────────────────────────

def _parse_llm_response(raw: str, ctx: RunContext) -> None:
    m = re.search(r"SUMMARY:\s*(.+?)(?=\nFINDINGS:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        ctx.summary = m.group(1).strip()

    m = re.search(r"FINDINGS:\s*(.+?)(?=\nRECOMMENDATION:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        ctx.findings = [
            b.strip()
            for b in re.findall(r"[-*•]\s+(.+)", m.group(1))
            if b.strip()
        ]

    m = re.search(r"RECOMMENDATION:\s*(.+)\Z", raw, re.DOTALL | re.IGNORECASE)
    if m:
        ctx.recommendation = m.group(1).strip()

    # Fallback: if parsing found nothing useful, use first 200 chars
    if not ctx.summary:
        ctx.summary = raw[:200].strip()
