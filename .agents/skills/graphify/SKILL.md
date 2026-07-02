# Graphify Skill

A local code knowledge graph tool that indexes **any** git repository and enables
semantic search, graph traversal, and multi-agent analysis using **any LLM**.

Uses **Qdrant in Docker** as the vector store and auto-detects the LLM provider
from `.env` (Databricks → OpenAI → Anthropic → Google → Ollama fallback).

## When to Use This Skill

Use this skill when the user asks to:
- Index any git repo into the knowledge graph
- Ask questions across one or many repos
- Extract and visualize code structure
- Run multi-agent analysis on code
- Switch or configure LLM providers
- Push knowledge outputs to GitHub for team sharing
- Rebuild the index on a fresh clone

## Command Quick Reference

```bash
# ── Docker (start Qdrant first) ────────────────────────────────────────────
docker compose up -d                       # start Qdrant container
docker compose stop                        # stop container
docker logs graphify-qdrant               # view Qdrant logs

# ── Setup ──────────────────────────────────────────────────────────────────
graphify init                              # create .graphify.json + .gitignore
graphify add-repo <path>                   # register + index + graph any repo
graphify add-repo <path> --name <alias>   # register with a custom name
graphify add-repo <path> --no-index       # register only, skip indexing
graphify rebuild                           # rebuild Qdrant from chunks.jsonl (no source needed)

# ── Phase 1: Index ─────────────────────────────────────────────────────────
graphify index <path1> <path2>            # index specific repos
graphify index --all                       # index all repos in .graphify.json
graphify index --all --reindex            # force full re-index
graphify status                            # show indexed repos + chunk counts
graphify clear --repo <name>              # remove one repo from index
graphify clear --yes                      # remove all index data

# ── Phase 2: Graph ─────────────────────────────────────────────────────────
graphify graph <path>                      # extract graph from a repo
graphify graph --all                       # extract from all registered repos
graphify visualize                         # open graph.html in browser
graphify report                            # regenerate GRAPH_REPORT.md

# ── Phase 3: Query ─────────────────────────────────────────────────────────
graphify query "<question>"                # pure vector search (fast, no LLM)
graphify ask "<question>"                  # auto-detect provider from .env
graphify ask "<question>" --provider databricks          # force Databricks
graphify ask "<question>" --provider openai --llm gpt-4o # force OpenAI
graphify ask "<question>" --provider anthropic           # force Anthropic
graphify ask "<question>" --provider google              # force Google
graphify ask "<question>" --provider ollama --llm llama3 # force Ollama
graphify ask "<question>" --threshold 0.85               # strict similarity filter
graphify ask "<question>" --repo <name>                  # filter to one repo
graphify ask "<question>" --context                      # print context only

# ── Phase 4: Swarm ─────────────────────────────────────────────────────────
graphify swarm "<task>"                                  # auto-detect provider
graphify swarm "<task>" --provider databricks            # force Databricks
graphify swarm "<task>" --mode edit                      # propose edits
graphify swarm "<task>" --mode edit --apply --yes        # apply edits

# ── Team Sync ──────────────────────────────────────────────────────────────
graphify sync                              # commit + push knowledge outputs
graphify sync --message "custom msg"      # custom commit message
graphify sync --no-push                   # commit only
```

## LLM Provider Auto-Detection

Graphify reads `.env` and picks the first configured provider:

| Priority | Provider | Env var needed |
|---|---|---|
| 1 | Databricks | `DATABRICKS_TOKEN` + endpoint |
| 2 | OpenAI | `OPENAI_API_KEY` |
| 3 | Anthropic | `ANTHROPIC_API_KEY` |
| 4 | Google | `GOOGLE_API_KEY` |
| 5 | Ollama | *(none — local fallback)* |

API providers auto-apply `--threshold 0.85` so only high-confidence chunks reach the LLM.

## Architecture

```
graphify/
├── indexer/     Phase 1 — chunk, embed (all-MiniLM-L6-v2), store in Qdrant
├── graph/       Phase 2 — AST/regex extraction → NetworkX → graph.html
├── query/       Phase 3 — vector search + score_threshold + graph BFS + any LLM
├── agents/      Phase 4 — Retriever → Reasoner → Editor → Validator
└── memory/      Episodic query log (graphify-out/memory/episodic.jsonl)
```

## Team Clone Workflow

```bash
git clone https://github.com/LokAyiti/graphify-swarm
pip install -e .
docker compose up -d     # start Qdrant in Docker Desktop
graphify rebuild         # rebuilds Qdrant from committed chunks.jsonl
graphify ask "question"  # auto-uses API key from .env
```

## Configuration (.graphify.json)

```json
{
  "repos": [{"name": "my-repo", "path": "C:\\path\\to\\any-repo"}],
  "default_llm": null,
  "ollama_host": "http://localhost:11434",
  "qdrant_url": null
}
```

## Qdrant + LLM Connection (.env)

```ini
# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333   # local Docker (active)
# QDRANT_URL=https://<cluster>.aws.cloud.qdrant.io:6333   # cloud
# QDRANT_API_KEY=your-qdrant-key

# ── LLM Providers (set ONE or more; first configured wins) ──────────────────
DATABRICKS_TOKEN=dapi...
DATABRICKS_SONNET_ENDPOINT=https://adb-....azuredatabricks.net/serving-endpoints/.../invocations
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=AIza...
```

## What Gets Committed to GitHub

| File | Why |
|---|---|
| `graphify-out/chunks.jsonl` | Chunk content for offline rebuild |
| `graphify-out/summaries.json` | Repo metadata |
| `graphify-out/graph.json` | Knowledge graph |
| `graphify-out/graph.html` | Interactive visualization |
| `graphify-out/GRAPH_REPORT.md` | Analysis report |
| `.graphify.json` | Repo registry |

**Gitignored:** `graphify-out/qdrant/`, `graphify-out/cache/embeddings/`, `graphify-out/memory/`, `.env`

## Supported Repo Types (any git repo works)

| Type | Extraction |
|---|---|
| Python | Functions, classes, imports, call graph, inheritance |
| JavaScript / TypeScript | Functions, classes, imports, exports |
| Markdown | Heading sections, internal links |
| JSON / ADF pipelines | Top-level keys; pipeline name, activities, folder, type |
| YAML, Shell, SQL, Terraform, Bicep, Rust, Go, ... | File node + sliding-window chunks |

## Key Technical Notes

- `build_llm(provider, model)` factory — always use this, never instantiate LLM classes directly
- `score_threshold` is enforced at the Qdrant `query_points()` call — irrelevant chunks never reach LLM
- qdrant-client 1.18+: use `query_points()` not `search()` (API removed)
- Files with UTF-8 BOM are handled automatically (`utf-8-sig` encoding)
- All NetworkX node_link calls need `edges="links"` kwarg
- Episodic log never blocks the main flow — failures are silently ignored


## Command Quick Reference

```bash
# ── Docker (start Qdrant first) ────────────────────────────────────────────
docker compose up -d                       # start Qdrant container
docker compose stop                        # stop container
docker logs graphify-qdrant               # view Qdrant logs

# ── Setup ──────────────────────────────────────────────────────────────────
graphify init                              # create .graphify.json + .gitignore
graphify add-repo <path>                   # register + index + graph a repo
graphify add-repo <path> --name <alias>   # register with a custom name
graphify add-repo <path> --no-index       # register only, skip indexing
graphify rebuild                           # rebuild Qdrant from chunks.jsonl (no source files)

# ── Phase 1: Index ─────────────────────────────────────────────────────────
graphify index <path1> <path2>            # index specific repos
graphify index --all                       # index all repos in .graphify.json
graphify index --all --reindex            # force full re-index
graphify status                            # show indexed repos + chunk counts
graphify clear --repo <name>              # remove one repo from index
graphify clear --yes                      # remove all index data

# ── Phase 2: Graph ─────────────────────────────────────────────────────────
graphify graph <path>                      # extract graph from a repo
graphify graph --all                       # extract from all registered repos
graphify visualize                         # open graph.html in browser
graphify report                            # regenerate GRAPH_REPORT.md

# ── Phase 3: Query ─────────────────────────────────────────────────────────
graphify query "<question>"                # pure vector search (fast)
graphify ask "<question>"                  # vector + graph context
graphify ask "<question>" --llm llama3    # + Ollama LLM answer (streaming)
graphify ask "<question>" --repo <name>   # filter to one repo
graphify ask "<question>" --context       # print context only (for any LLM)

# ── Phase 4: Swarm ─────────────────────────────────────────────────────────
graphify swarm "<task>"                    # 4-agent analyze (no LLM)
graphify swarm "<task>" --llm llama3      # with LLM reasoning
graphify swarm "<task>" --mode edit --llm llama3          # propose edits
graphify swarm "<task>" --mode edit --llm llama3 --apply  # apply edits

# ── Team Sync ──────────────────────────────────────────────────────────────
graphify sync                              # commit + push knowledge outputs
graphify sync --message "custom msg"      # custom commit message
graphify sync --no-push                   # commit only
```

## Architecture

```
graphify/
├── indexer/     Phase 1 — chunk, embed (all-MiniLM-L6-v2), store in Qdrant
├── graph/       Phase 2 — AST/regex extraction → NetworkX → graph.html
├── query/       Phase 3 — vector search + graph BFS + Ollama LLM
└── agents/      Phase 4 — Retriever → Reasoner → Editor → Validator
```

## Team Clone Workflow

```bash
git clone https://github.com/LokAyiti/graphify-swarm
pip install -e .
docker compose up -d     # start Qdrant in Docker Desktop
graphify rebuild         # rebuilds Qdrant from committed chunks.jsonl
graphify ask "question"
```

## Configuration (.graphify.json)

```json
{
  "repos": [{"name": "pipeline", "path": "C:\\path\\to\\pipeline"}],
  "default_llm": "llama3",
  "ollama_host": "http://localhost:11434",
  "qdrant_url": null
}
```

## Qdrant Connection (.env)

```ini
# Option A — Local Docker (default)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Option B — Qdrant Cloud
# QDRANT_URL=https://<cluster>.aws.cloud.qdrant.io:6333
# QDRANT_API_KEY=your-api-key

# Leave both commented out for embedded disk mode (no Docker needed)
```

> QDRANT_URL from `.env` takes priority over `qdrant_url` in `.graphify.json`.

## What Gets Committed to GitHub

| File | Why |
|---|---|
| `graphify-out/chunks.jsonl` | Chunk content for offline rebuild |
| `graphify-out/summaries.json` | Repo metadata |
| `graphify-out/graph.json` | Knowledge graph |
| `graphify-out/graph.html` | Interactive visualization |
| `graphify-out/GRAPH_REPORT.md` | Analysis report |
| `.graphify.json` | Repo registry |

`graphify-out/qdrant/` and `graphify-out/cache/embeddings/` are **gitignored** (large binary files, auto-regenerated locally).

## Supported Repo Types

- Azure Data Factory pipelines (JSON with `.properties.activities`) — rich metadata extraction
- Python repos — function/class/import/call graph
- JavaScript/TypeScript — function/class/import extraction
- Any git repo — sliding-window chunking for all other files

## Key Technical Notes

- qdrant-client 1.18+: use `query_points()` not `search()` (API removed)
- Files with UTF-8 BOM are handled automatically (`utf-8-sig` encoding)
- All NetworkX node_link calls need `edges="links"` kwarg
- Embedding cache is content-addressed (SHA256) — same content = cached automatically
