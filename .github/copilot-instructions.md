# Graphify — Workspace Instructions for GitHub Copilot

This repository contains **Graphify**, a local code knowledge graph tool built in Python.
It indexes **any** git repository into a searchable vector store (Qdrant) and knowledge
graph (NetworkX), then provides CLI commands for semantic search, graph traversal, and
multi-agent analysis using **any LLM** (Databricks, OpenAI, Anthropic, Google, or Ollama).

## What This Repo Is

- `graphify/` — the CLI tool source (Python 3.11)
- `graphify-out/` — generated outputs (gitignored except `chunks.jsonl`, `graph.*`, `summaries.json`)
- `.graphify.json` — registry of indexed repos + settings
- `.env` — LLM API keys + Qdrant connection (never committed)
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
| `graphify/indexer/qdrant_store.py` | Qdrant CRUD + `query_points` API + `score_threshold` |
| `graphify/graph/extractor.py` | AST/regex → GNode + GEdge for graph building |
| `graphify/graph/visualizer.py` | Writes graph.json + self-contained graph.html |
| `graphify/query/router.py` | Dual router: vector search (with threshold) + graph BFS |
| `graphify/query/merger.py` | Formats merged context + builds structured system prompt |
| `graphify/query/llm.py` | Multi-provider LLM: Databricks, OpenAI, Anthropic, Google, Ollama |
| `graphify/agents/swarm.py` | 4-agent orchestrator: retriever→reasoner→editor→validator |
| `graphify/memory/episodic.py` | Append-only query log (JSONL) for analytics + future caching |

## LLM Provider System

`graphify/query/llm.py` provides:
- `BaseLLM` — abstract base with `ask_stream()` + `ask()` interface
- `OllamaLLM` — local Ollama (no API key)
- `OpenAICompatibleLLM` — OpenAI, Groq, Together, Azure OpenAI
- `AnthropicLLM` — direct Anthropic Messages API
- `GoogleLLM` — Gemini via Generative Language API
- `DatabricksLLM` — Databricks Model Serving (invocations endpoint)
- `build_llm(provider, model, ollama_host)` — factory, auto-detects from `.env`
- `detect_provider()` — returns first configured provider or `None`

**Auto-detection priority:** `DATABRICKS_TOKEN` → `OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → `GOOGLE_API_KEY` → Ollama fallback

## Important Design Decisions

- **qdrant-client 1.18+**: use `client.query_points()` not `client.search()` (removed)
- **score_threshold**: passed directly into `query_points()` — irrelevant chunks never reach LLM
- **UTF-8 BOM**: all JSON/file reads use `encoding="utf-8-sig"` to strip BOM
- **chunks.jsonl**: chunk content saved during `index` so team members can `rebuild` without original files
- **embedding cache**: SHA256-keyed numpy `.npy` files under `graphify-out/cache/embeddings/` (gitignored)
- **NetworkX node_link format**: always pass `edges="links"` to `node_link_data/graph` calls
- **graph.html**: self-contained HTML with vis-network from cdnjs CDN — no build step
- **Qdrant storage**: gitignored (`graphify-out/qdrant/`), managed by Docker, rebuilt via `graphify rebuild`
- **episodic log**: `graphify-out/memory/episodic.jsonl` — append-only, never blocks main flow
- **system prompt**: `merger.py::build_llm_messages()` uses structured template with repo citation rules

## Coding Patterns in This Codebase

- All CLI commands are `@app.command()` functions in `graphify/cli.py`
- Lazy imports inside command functions (avoids slow startup)
- `OUT_DIR = Path("graphify-out")` — all outputs relative to CWD
- `_qdrant_url(cfg)` helper: env var `QDRANT_URL` overrides `.graphify.json`
- `build_llm(provider, model)` factory in `cli.py` — always use this, never instantiate LLM classes directly
- `RunContext` dataclass flows through all 4 agents (mutated in-place)
- `BaseAgent` ABC with `run(ctx) -> ctx` interface
- Error handling: agents catch exceptions and append to `ctx.errors` rather than raising

## Indexed Repos

See `graphify-out/summaries.json` for the current list of indexed repos.
Graphify works with **any** git repository — no special config required.

