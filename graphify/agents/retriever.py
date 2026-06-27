"""
retriever.py — Agent 1: vector search + graph expansion.

Wraps Phase 3's Router.  Outputs vector_hits, graph_contexts,
and the merged_context string onto RunContext.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from graphify.agents.base import BaseAgent, RunContext
from graphify.query.merger import format_context
from graphify.query.router import Router


class RetrieverAgent(BaseAgent):
    name = "retriever"

    def __init__(
        self,
        qdrant_dir:      Path,
        embedder_cache:  Path,
        graph_json_path: Optional[Path] = None,
    ) -> None:
        self._router = Router(qdrant_dir, embedder_cache, graph_json_path)

    def run(self, ctx: RunContext) -> RunContext:
        hits, graph_contexts = self._router.route(
            ctx.task,
            top_k=ctx.top_k,
            repo_filter=ctx.repo_filter,
        )
        ctx.vector_hits    = hits
        ctx.graph_contexts = graph_contexts
        ctx.merged_context = format_context(hits, graph_contexts)
        return ctx
