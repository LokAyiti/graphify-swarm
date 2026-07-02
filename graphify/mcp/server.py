"""
graphify/mcp/server.py — MCP (Model Context Protocol) server for Graphify.

Exposes your indexed code repos to any MCP-compatible AI tool:
  VS Code Copilot, Claude Desktop, Cursor, Windsurf, ChatGPT, etc.

Protocol: JSON-RPC 2.0 over stdio (MCP standard).

Tools exposed
-------------
  search_codebase(query, top_k=8, repo=None, threshold=0.0)
      Semantic search across all indexed repos.  Returns the top matching
      code/text chunks with scores, file paths, and content.

  get_file_context(file_path, repo)
      Returns the graph neighbourhood for a specific file: its functions,
      classes, imports, and the files that import it.

  list_repos()
      Returns all indexed repos with file counts and chunk counts.

Usage
-----
Run directly (VS Code / Claude Desktop will do this automatically):

    python -m graphify.mcp.server          # from graphify-swarm root
    graphify mcp                           # via CLI

VS Code setup  →  .vscode/mcp.json
Claude Desktop →  see claude_desktop_config.json instructions in README
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# MCP tool definitions (sent during tools/list)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "search_codebase",
        "description": (
            "Semantic search across all indexed code repositories. "
            "Returns the most relevant code chunks, documentation sections, "
            "or pipeline definitions matching the query. "
            "Use this to answer questions about how code works, find functions, "
            "or locate patterns across multiple repos."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question or keyword search",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 8, max: 20)",
                    "default": 8,
                },
                "repo": {
                    "type": "string",
                    "description": "Restrict search to a specific repo name (optional)",
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum cosine similarity score 0.0-1.0 (default: 0.0 = off)",
                    "default": 0.0,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file_context",
        "description": (
            "Get the structural graph context for a specific file: "
            "its functions, classes, imports, and which other files import it. "
            "Use this after search_codebase to dive deeper into a specific file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative file path within the repo (e.g. 'src/utils.py')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repo name containing the file",
                },
            },
            "required": ["file_path", "repo"],
        },
    },
    {
        "name": "list_repos",
        "description": (
            "List all indexed repositories with their file counts, "
            "chunk counts, and local paths. "
            "Use this to understand what codebases are available to search."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _search_codebase(query: str, top_k: int = 8, repo: str | None = None,
                     threshold: float = 0.0) -> str:
    """Run vector search and return formatted results."""
    from graphify.indexer.embedder import Embedder
    from graphify.indexer.qdrant_store import QdrantStore
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path=Path(".env"), override=False)

    qdrant_url = os.environ.get("QDRANT_URL")
    api_key    = os.environ.get("QDRANT_API_KEY") or None
    cache_dir  = Path("graphify-out") / "cache"
    qdrant_dir = Path("graphify-out") / "qdrant"

    store    = QdrantStore(qdrant_dir, url=qdrant_url, api_key=api_key)
    embedder = Embedder(cache_dir)

    query_vec = embedder.embed([query])[0]
    results   = store.search(
        query_vec,
        top_k      = min(top_k, 20),
        repo_filter = repo or None,
        score_threshold = threshold if threshold > 0 else None,
    )

    if not results:
        return "No results found above the threshold."

    lines = [f"Found {len(results)} result(s) for: {query!r}\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"--- Result {i} ---\n"
            f"Score    : {r['score']:.3f}\n"
            f"Repo     : {r.get('repo', '')}\n"
            f"File     : {r.get('file_path', '')}\n"
            f"Lines    : {r.get('start_line')}–{r.get('end_line')}\n"
            f"Type     : {r.get('chunk_type', '')}  /  {r.get('language', '')}\n"
            f"Symbol   : {r.get('name', '') or '(none)'}\n"
            f"Content  :\n{r.get('content', '')[:800]}\n"
        )
    return "\n".join(lines)


def _get_file_context(file_path: str, repo: str) -> str:
    """Return graph neighbourhood for a file."""
    import json as _json
    from graphify.query.router import Router

    cache_dir  = Path("graphify-out") / "cache"
    qdrant_dir = Path("graphify-out") / "qdrant"
    graph_path = Path("graphify-out") / "graph.json"

    from dotenv import load_dotenv
    import os
    load_dotenv(dotenv_path=Path(".env"), override=False)
    qdrant_url = os.environ.get("QDRANT_URL")
    api_key    = os.environ.get("QDRANT_API_KEY") or None

    router = Router(
        qdrant_dir,
        cache_dir,
        graph_json_path = graph_path if graph_path.exists() else None,
        qdrant_url      = qdrant_url,
        qdrant_api_key  = api_key,
    )

    graph = router._get_graph()
    if graph is None:
        return (
            "No graph.json found. Run `graphify graph --all` first "
            "to enable structural context."
        )

    fid = f"file:{repo}:{file_path}"
    if fid not in graph:
        return f"File '{file_path}' in repo '{repo}' not found in graph."

    node = graph.nodes[fid]
    lines = [
        f"File: {file_path}  (repo: {repo})\n",
        f"Language : {node.get('language', 'unknown')}",
    ]

    # Contained symbols
    contains = []
    for succ in graph.successors(fid):
        bundle = graph.get_edge_data(fid, succ) or {}
        for edge_attrs in bundle.values():
            if edge_attrs.get("type") == "contains":
                d = graph.nodes.get(succ, {})
                contains.append(f"  {d.get('type','?'):12} {d.get('name', succ)}")
    if contains:
        lines.append("\nContains:")
        lines.extend(contains[:30])

    # Imports
    imports = []
    for succ in graph.successors(fid):
        bundle = graph.get_edge_data(fid, succ) or {}
        for edge_attrs in bundle.values():
            if edge_attrs.get("type") == "imports":
                d = graph.nodes.get(succ, {})
                imports.append(f"  {d.get('name', succ)}")
    if imports:
        lines.append("\nImports:")
        lines.extend(imports[:20])

    # Imported by
    imported_by = []
    for pred in graph.predecessors(fid):
        d = graph.nodes.get(pred, {})
        if d.get("type") == "file":
            imported_by.append(f"  {d.get('file_path', pred)}")
    if imported_by:
        lines.append("\nImported by:")
        lines.extend(imported_by[:10])

    # ADF metadata
    md = node.get("metadata") or {}
    if md.get("pipeline_name"):
        lines.append(f"\nPipeline   : {md['pipeline_name']}")
    if md.get("activity_count"):
        acts = ", ".join(md.get("activity_names", [])[:8])
        lines.append(f"Activities : {md['activity_count']}  ({acts})")

    return "\n".join(lines)


def _list_repos() -> str:
    """Return all indexed repos from summaries.json."""
    summaries_path = Path("graphify-out") / "summaries.json"
    if not summaries_path.exists():
        return "No repos indexed yet. Run `graphify index --all` first."

    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    if not summaries:
        return "No repos indexed yet."

    lines = [f"Indexed repositories ({len(summaries)} total):\n"]
    for name, info in summaries.items():
        lines.append(
            f"  {name}\n"
            f"    Files  : {info.get('total_files', 0)}\n"
            f"    Chunks : {info.get('total_chunks', 0)}\n"
            f"    Path   : {info.get('repo_path', '(not set)')}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------

def _make_result(request_id, result: object) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _make_error(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _handle(msg: dict) -> dict | None:
    method     = msg.get("method", "")
    request_id = msg.get("id")
    params     = msg.get("params") or {}

    # Notifications (no id) — acknowledge but don't reply
    if request_id is None and method.startswith("notifications/"):
        return None

    # ── initialize ────────────────────────────────────────────────────────
    if method == "initialize":
        return _make_result(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name":    "graphify",
                "version": "1.0.0",
            },
        })

    # ── tools/list ────────────────────────────────────────────────────────
    if method == "tools/list":
        return _make_result(request_id, {"tools": _TOOLS})

    # ── tools/call ────────────────────────────────────────────────────────
    if method == "tools/call":
        name      = params.get("name", "")
        arguments = params.get("arguments") or {}

        try:
            if name == "search_codebase":
                text = _search_codebase(
                    query     = arguments["query"],
                    top_k     = int(arguments.get("top_k", 8)),
                    repo      = arguments.get("repo"),
                    threshold = float(arguments.get("threshold", 0.0)),
                )
            elif name == "get_file_context":
                text = _get_file_context(
                    file_path = arguments["file_path"],
                    repo      = arguments["repo"],
                )
            elif name == "list_repos":
                text = _list_repos()
            else:
                return _make_error(request_id, -32601, f"Unknown tool: {name!r}")

            return _make_result(request_id, {
                "content": [{"type": "text", "text": text}],
            })

        except KeyError as exc:
            return _make_error(request_id, -32602, f"Missing argument: {exc}")
        except Exception as exc:
            return _make_error(request_id, -32603, f"Tool error: {exc}")

    # ── initialized notification (no response needed) ─────────────────────
    if method == "notifications/initialized":
        return None

    return _make_error(request_id, -32601, f"Method not found: {method!r}")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve() -> None:
    """Run the MCP server — reads from stdin, writes to stdout."""
    # Reopen stdin/stdout in binary mode to avoid Windows encoding issues
    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    for raw_line in stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        response = _handle(msg)
        if response is not None:
            line = json.dumps(response, ensure_ascii=False) + "\n"
            stdout.write(line.encode("utf-8"))
            stdout.flush()


if __name__ == "__main__":
    serve()
