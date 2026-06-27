# Graphify — Workspace Instructions for GitHub Copilot

This repository contains **Graphify**, a local code knowledge graph tool built in Python.
It indexes git repositories into a searchable vector store (Qdrant) and knowledge graph
(NetworkX), then provides CLI commands for semantic search, graph traversal, and
multi-agent analysis.

## What This Repo Is

- `graphify/` — the CLI tool source (Python 3.11)
- `graphify-out/` — pre-built knowledge outputs committed to git
- `.graphify.json` — registry of indexed repos + settings
- `requirements.txt` / `pyproject.toml` — dependencies

## CLI Commands (13 total)

```
Phase 1 – Index     : graphify index, graphify status, graphify clear
Phase 2 – Graph     : graphify graph, graphify visualize, graphify report
Phase 3 – Query     : graphify query, graphify ask
Phase 4 – Swarm     : graphify swarm
Orchestration       : graphify init, graphify add-repo, graphify rebuild, graphify sync
```

## Key Files to Know

| File | Purpose |
|---|---|
| `graphify/cli.py` | All CLI commands — typer app with 13 commands |
| `graphify/config.py` | `.graphify.json` load/save/upsert helpers |
| `graphify/indexer/chunker.py` | Python AST + markdown + sliding-window chunking |
| `graphify/indexer/embedder.py` | sentence-transformers with SHA256 disk cache |
| `graphify/indexer/qdrant_store.py` | Local Qdrant CRUD + `query_points` API |
| `graphify/graph/extractor.py` | AST/regex → GNode + GEdge for graph building |
| `graphify/graph/visualizer.py` | Writes graph.json + self-contained graph.html |
| `graphify/query/router.py` | Dual router: vector search + graph BFS expansion |
| `graphify/query/merger.py` | Formats merged context for LLM consumption |
| `graphify/query/llm.py` | Ollama HTTP client with streaming `/api/chat` |
| `graphify/agents/swarm.py` | 4-agent orchestrator: retriever→reasoner→editor→validator |

## Important Design Decisions

- **qdrant-client 1.18+**: use `client.query_points()` not `client.search()` (removed)
- **UTF-8 BOM**: all JSON/file reads use `encoding="utf-8-sig"` to strip BOM
- **chunks.jsonl**: chunk content saved during `index` so team members can `rebuild` without original files
- **embedding cache**: SHA256-keyed numpy `.npy` files under `graphify-out/cache/embeddings/`
- **NetworkX node_link format**: always pass `edges="links"` to `node_link_data/graph` calls
- **graph.html**: self-contained HTML with vis-network from cdnjs CDN — no build step
- **Qdrant local storage**: gitignored (`graphify-out/qdrant/`), rebuilt via `graphify rebuild`

## Coding Patterns in This Codebase

- All CLI commands are `@app.command()` functions in `graphify/cli.py`
- Lazy imports inside command functions (avoids slow startup)
- `OUT_DIR = Path("graphify-out")` — all outputs relative to CWD
- `RunContext` dataclass flows through all 4 agents (mutated in-place)
- `BaseAgent` ABC with `run(ctx) -> ctx` interface
- Error handling: agents catch exceptions and append to `ctx.errors` rather than raising

## Indexed Content

The `graphify-out/` directory contains pre-indexed knowledge from:
- See `graphify-out/summaries.json` for the list of indexed repos

When answering questions about the codebase, you can reference the graph structure
and chunk content without needing to read every file.
