"""
merger.py — combines vector hits and graph contexts into a single
LLM-ready context string.

Layout
------
  [VECTOR SEARCH RESULTS]
    Result 1 … N  (code blocks, truncated to budget)

  [STRUCTURAL GRAPH CONTEXT]
    Per-file summaries: imports, symbols, ADF metadata, call edges
"""
from __future__ import annotations

from graphify.query.router import GraphContext, VectorHit

# Character budgets
_DEFAULT_TOTAL   = 7_000   # total context chars sent to LLM
_DEFAULT_CHUNK   = 1_000   # max chars per code chunk
_VECTOR_SHARE    = 0.65    # 65 % for code, 35 % for graph context


def format_context(
    hits:           list[VectorHit],
    graph_contexts: list[GraphContext],
    max_total:      int = _DEFAULT_TOTAL,
    max_chunk:      int = _DEFAULT_CHUNK,
) -> str:
    """Return a compact, LLM-ready context string."""
    parts: list[str] = []
    vector_budget    = int(max_total * _VECTOR_SHARE)
    used             = 0

    # ── Section 1: Vector search results ─────────────────────────────────────
    parts.append("[VECTOR SEARCH RESULTS]\n\n")

    for i, hit in enumerate(hits, 1):
        lang  = hit.language or "text"
        label = f"`{hit.name}` — " if hit.name else ""
        header = (
            f"Result {i} (score: {hit.score:.3f})\n"
            f"File: {hit.file_path}   Repo: {hit.repo}   Lines: {hit.start_line}–{hit.end_line}\n"
            f"Type: {hit.chunk_type}   Language: {lang}\n\n"
        )

        body = hit.content
        if len(body) > max_chunk:
            body = body[:max_chunk] + "\n… (truncated)"
        block = f"{header}```{lang}\n{body}\n```\n\n---\n\n"

        if used + len(block) > vector_budget and i > 1:
            parts.append(
                f"*({len(hits) - i + 1} more result(s) omitted to stay within context budget)*\n\n"
            )
            break

        parts.append(block)
        used += len(block)

    # ── Section 2: Structural graph context ───────────────────────────────────
    if graph_contexts:
        parts.append("[STRUCTURAL GRAPH CONTEXT]\n\n")

        for ctx in graph_contexts:
            parts.append(f"File: {ctx.file_path}  (repo: {ctx.repo}, lang: {ctx.language})\n")

            # ADF / JSON metadata
            md = ctx.metadata or {}
            if md.get("pipeline_name"):
                parts.append(f"  Pipeline name : {md['pipeline_name']}\n")
            if md.get("activity_count"):
                acts = md.get("activity_names", [])
                act_str = ", ".join(acts[:10]) + ("…" if len(acts) > 10 else "")
                parts.append(f"  Activities    : {md['activity_count']}  ({act_str})\n")
            if md.get("folder"):
                parts.append(f"  Folder        : {md['folder']}\n")
            if md.get("pipeline_type"):
                parts.append(f"  ADF type      : {md['pipeline_type']}\n")
            if md.get("top_keys"):
                parts.append(f"  Top-level keys: {', '.join(md['top_keys'][:8])}\n")

            # Symbols
            funcs    = [s.name for s in ctx.contains if s.node_type == "function"][:10]
            classes  = [s.name for s in ctx.contains if s.node_type == "class"][:6]
            sections = [s.name for s in ctx.contains if s.node_type == "section"][:6]

            if funcs:
                parts.append(f"  Functions     : {', '.join(funcs)}\n")
            if classes:
                parts.append(f"  Classes       : {', '.join(classes)}\n")
            if sections:
                parts.append(f"  Sections      : {', '.join(sections)}\n")

            # Edges
            if ctx.imports:
                parts.append(f"  Imports       : {', '.join(ctx.imports[:12])}\n")
            if ctx.imported_by:
                parts.append(f"  Used by       : {', '.join(ctx.imported_by[:5])}\n")
            if ctx.calls_out:
                parts.append(f"  Call edges    : {'; '.join(ctx.calls_out[:8])}\n")

            parts.append("\n")

    return "".join(parts)


def build_llm_messages(
    context:   str,
    question:  str,
    provider:  str = "ollama",
    model:     str = "",
    repos:     list[str] | None = None,
    threshold: float = 0.0,
) -> list[dict]:
    """Return a messages list for any provider's chat endpoint.

    Uses the structured Graphify system prompt that enforces repo citation,
    adapts verbosity based on provider, and structures swarm-mode output.
    """
    repo_names   = ", ".join(repos) if repos else "(all indexed repos)"
    provider_str = "ollama" if provider == "ollama" else "api"
    threshold_str = f"{threshold:.2f}" if threshold else "none"

    system = (
        f"You are Graphify, a multi-repo code knowledge assistant.\n\n"
        f"CONTEXT SOURCE: {repo_names}\n"
        f"PROVIDER: {provider_str}\n"
        f"MODEL: {model or provider}\n"
        f"MIN_SIMILARITY_THRESHOLD: {threshold_str}\n\n"
        "Below is the retrieved context, grouped by repo. Each chunk is tagged with "
        "its source repo, file path, and cosine similarity score. "
        f"Only use chunks with a similarity score >= {threshold_str if threshold_str != 'none' else '0.0'}. "
        "If no chunks meet this threshold, say so explicitly instead of guessing.\n\n"
        "Rules:\n"
        "1. Always cite which repo and file a fact came from, "
        "e.g. [repo: pipeline, file: cashier.py].\n"
        "2. Never blend logic from two different repos unless the user explicitly asks "
        "for a cross-repo comparison.\n"
        "3. If the user's query matches multiple repos, list which repos were searched "
        "and which ones returned relevant context.\n"
        "4. If PROVIDER is 'ollama' (free local fallback), keep answers concise since "
        "local models have smaller context windows.\n"
        "5. If PROVIDER is 'api', you may give a more detailed, multi-step analysis.\n"
        "6. For 'swarm' mode queries, structure your output as: "
        "Findings → Evidence (with repo/file tags) → Suggested next action."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"{context}\n\nQuestion: {question}"},
    ]
