"""
builder.py — assembles a NetworkX MultiDiGraph from extracted nodes and edges.

Deduplication rules
-------------------
  Nodes  : unique by id  (first-seen wins for attributes)
  Edges  : unique by (source, target, type)  — no parallel edges of the same type
"""
from __future__ import annotations

from typing import List

import networkx as nx

from graphify.graph.extractor import GEdge, GNode


def build_graph(
    all_nodes: List[GNode],
    all_edges: List[GEdge],
    repo_names: List[str] | None = None,
) -> nx.MultiDiGraph:
    """Build and return a deduplicated MultiDiGraph."""
    G = nx.MultiDiGraph(name="graphify")
    seen_edges: set[tuple[str, str, str]] = set()

    # Add repo nodes first
    for repo in (repo_names or []):
        G.add_node(
            f"repo:{repo}",
            type="repo", name=repo, repo=repo,
            file_path="", language="",
            start_line=0, end_line=0,
            metadata={},
        )

    # Deduplicated nodes
    for node in all_nodes:
        if node.id not in G:
            G.add_node(
                node.id,
                **{k: v for k, v in node.__dict__.items() if k != "id"},
            )

    # Wire files to their repo nodes
    for nid, data in list(G.nodes(data=True)):
        if data.get("type") == "file":
            repo     = data.get("repo", "")
            repo_nid = f"repo:{repo}"
            if repo_nid in G:
                key = (repo_nid, nid, "contains")
                if key not in seen_edges:
                    G.add_edge(repo_nid, nid, type="contains")
                    seen_edges.add(key)

    # Deduplicated extracted edges
    for edge in all_edges:
        if edge.source not in G or edge.target not in G:
            continue
        key = (edge.source, edge.target, edge.type)
        if key not in seen_edges:
            attrs = {k: v for k, v in edge.__dict__.items() if k not in ("source", "target")}
            G.add_edge(edge.source, edge.target, **attrs)
            seen_edges.add(key)

    return G


def graph_stats(G: nx.MultiDiGraph) -> dict:
    """Return a summary statistics dict for the graph."""
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}

    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    for _, _, data in G.edges(data=True):
        t = data.get("type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types":  node_types,
        "edge_types":  edge_types,
    }


def top_nodes_by_degree(G: nx.MultiDiGraph, n: int = 10) -> list[tuple[str, int, dict]]:
    """Return (node_id, degree, attrs) sorted descending by total degree."""
    ranked = [
        (nid, G.degree(nid), data)
        for nid, data in G.nodes(data=True)
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:n]


def most_imported(G: nx.MultiDiGraph, n: int = 10) -> list[tuple[str, int, dict]]:
    """Return import nodes sorted by in-degree (how many files import them)."""
    imports = [
        (nid, G.in_degree(nid), data)
        for nid, data in G.nodes(data=True)
        if data.get("type") == "import"
    ]
    imports.sort(key=lambda x: x[1], reverse=True)
    return imports[:n]


def isolated_nodes(G: nx.MultiDiGraph) -> list[tuple[str, dict]]:
    """Return nodes with degree == 0 (no connections)."""
    return [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if G.degree(nid) == 0
    ]
