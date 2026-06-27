"""
router.py — Phase 3 dual-mode query router.

For a given question the router runs two parallel searches:

1. Vector path  — embed question → Qdrant cosine search → ranked chunks
2. Graph path   — load graph.json → locate file nodes that match the top
                  vector hits → BFS one hop → extract structural context
                  (imports, contained symbols, ADF metadata, …)

The two result sets are kept separate so the merger can format them
however it likes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import networkx as nx
from networkx.readwrite import json_graph


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class VectorHit:
    score:      float
    repo:       str
    file_path:  str
    language:   str
    chunk_type: str
    name:       str
    content:    str
    start_line: int
    end_line:   int


@dataclass
class SymbolInfo:
    node_type:  str    # function | class | section
    name:       str
    start_line: int = 0
    end_line:   int = 0


@dataclass
class GraphContext:
    """Structural context for one file node in the graph."""
    file_path:        str
    repo:             str
    language:         str
    contains:         List[SymbolInfo] = field(default_factory=list)
    imports:          List[str]        = field(default_factory=list)
    imported_by:      List[str]        = field(default_factory=list)  # other files → this
    calls_out:        List[str]        = field(default_factory=list)  # func names called
    metadata:         dict             = field(default_factory=dict)


# ── Router ─────────────────────────────────────────────────────────────────────

class Router:
    """Lazy-loading dual query router (vector + graph)."""

    def __init__(
        self,
        qdrant_dir:       Path,
        embedder_cache:   Path,
        graph_json_path:  Optional[Path] = None,
    ) -> None:
        self._qdrant_dir     = qdrant_dir
        self._embedder_cache = embedder_cache
        self._graph_path     = graph_json_path

        self._store    = None
        self._embedder = None
        self._graph:   nx.MultiDiGraph | None = None
        self._graph_loaded = False

    # ── Lazy accessors ────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from graphify.indexer.qdrant_store import QdrantStore
            self._store = QdrantStore(self._qdrant_dir)
        return self._store

    def _get_embedder(self):
        if self._embedder is None:
            from graphify.indexer.embedder import Embedder
            self._embedder = Embedder(self._embedder_cache)
        return self._embedder

    def _get_graph(self) -> nx.MultiDiGraph | None:
        if not self._graph_loaded:
            self._graph_loaded = True
            if self._graph_path and self._graph_path.exists():
                try:
                    data        = json.loads(self._graph_path.read_text(encoding="utf-8"))
                    self._graph = json_graph.node_link_graph(
                        data, directed=True, multigraph=True, edges="links"
                    )
                except Exception:
                    self._graph = None
        return self._graph

    def index_count(self) -> int:
        return self._get_store().count()

    # ── Vector search ─────────────────────────────────────────────────────────

    def vector_search(
        self,
        question:    str,
        top_k:       int          = 8,
        repo_filter: str | None   = None,
        lang_filter: str | None   = None,
    ) -> List[VectorHit]:
        embedder  = self._get_embedder()
        store     = self._get_store()
        query_vec = embedder.embed([question])[0]

        raw = store.search(
            query_vec,
            top_k=top_k,
            repo_filter=repo_filter,
            language_filter=lang_filter,
        )
        return [
            VectorHit(
                score      = r.get("score", 0.0),
                repo       = r.get("repo", ""),
                file_path  = r.get("file_path", ""),
                language   = r.get("language", ""),
                chunk_type = r.get("chunk_type", ""),
                name       = r.get("name", ""),
                content    = r.get("content", ""),
                start_line = r.get("start_line", 0),
                end_line   = r.get("end_line", 0),
            )
            for r in raw
        ]

    # ── Graph traversal ───────────────────────────────────────────────────────

    def graph_expand(self, hits: List[VectorHit]) -> List[GraphContext]:
        """For each unique file in *hits*, return its 1-hop graph neighbourhood."""
        G = self._get_graph()
        if G is None:
            return []

        seen:     set[str]            = set()
        contexts: List[GraphContext]  = []

        for hit in hits:
            fid = f"file:{hit.repo}:{hit.file_path}"
            if fid in seen or fid not in G:
                continue
            seen.add(fid)

            node_data = G.nodes[fid]
            ctx = GraphContext(
                file_path = hit.file_path,
                repo      = hit.repo,
                language  = hit.language,
                metadata  = node_data.get("metadata") or {},
            )

            # Outgoing edges: contains / imports
            for succ in G.successors(fid):
                edge_bundle = G.get_edge_data(fid, succ) or {}
                for edge_attrs in edge_bundle.values():
                    etype     = edge_attrs.get("type", "")
                    succ_data = G.nodes.get(succ) or {}
                    succ_type = succ_data.get("type", "")
                    succ_name = succ_data.get("name", succ.split(":")[-1])

                    if etype == "imports":
                        ctx.imports.append(succ_name)
                    elif etype == "contains":
                        ctx.contains.append(SymbolInfo(
                            node_type  = succ_type,
                            name       = succ_name,
                            start_line = succ_data.get("start_line", 0),
                            end_line   = succ_data.get("end_line", 0),
                        ))

            # Incoming edges: other files that import / reference this one
            for pred in G.predecessors(fid):
                pred_data = G.nodes.get(pred) or {}
                if pred_data.get("type") == "file":
                    fp = pred_data.get("file_path", "")
                    if fp:
                        ctx.imported_by.append(fp)

            # Calls made by functions in this file
            for symbol in ctx.contains:
                if symbol.node_type == "function":
                    func_id = f"func:{hit.repo}:{hit.file_path}:{symbol.name}:{symbol.start_line}"
                    if func_id in G:
                        for callee in G.successors(func_id):
                            edge_bundle = G.get_edge_data(func_id, callee) or {}
                            for ea in edge_bundle.values():
                                if ea.get("type") == "calls":
                                    callee_name = (G.nodes.get(callee) or {}).get("name", "")
                                    if callee_name:
                                        ctx.calls_out.append(f"{symbol.name} → {callee_name}")

            # Deduplicate lists
            ctx.imports    = sorted(set(ctx.imports))
            ctx.imported_by = sorted(set(ctx.imported_by))
            ctx.calls_out  = sorted(set(ctx.calls_out))

            contexts.append(ctx)

        return contexts

    # ── Combined route ────────────────────────────────────────────────────────

    def route(
        self,
        question:    str,
        top_k:       int        = 8,
        repo_filter: str | None = None,
        lang_filter: str | None = None,
    ) -> tuple[List[VectorHit], List[GraphContext]]:
        """Run vector search then graph expansion.  Returns (hits, contexts)."""
        hits     = self.vector_search(question, top_k=top_k,
                                      repo_filter=repo_filter, lang_filter=lang_filter)
        contexts = self.graph_expand(hits)
        return hits, contexts
