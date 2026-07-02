# Graphify — Local Code Knowledge Graph

> Index **any** git repository into a searchable, queryable knowledge graph.  
> Works with **any LLM** — Databricks, OpenAI, Anthropic, Google, or local Ollama.  
> Runs locally via Docker (Qdrant) or connects to Qdrant Cloud for team sharing.

```
graphify add-repo  ./my-repo          # register + index + extract graph (any repo)
graphify ask       "how does auth work?"    # auto-uses API key from .env, or Ollama
graphify swarm     "find duplicate patterns" # 4-agent analysis pipeline
graphify sync                                # push knowledge to GitHub for the team
```

> **Works with any repo.** Python, JavaScript, TypeScript, Rust, Go, Markdown, JSON,
> YAML, SQL, Terraform, Bicep, shell scripts — just run `graphify add-repo <path>`
> and start querying. No config needed.

---

## What It Does

Graphify turns local git repositories into a living knowledge base your whole team can query. You index repos once, the outputs are committed to this repository, and anyone who clones it can start asking questions immediately — without touching the original source files.

```
Your repos (local)          graphify-swarm (GitHub)          Any team member
──────────────────          ────────────────────────         ─────────────────
pipeline/                   graphify-out/                    git clone …
app-code/      ──index──▶   ├── chunks.jsonl    ──push──▶   pip install -e .
infra-tf/                   ├── graph.json                   graphify rebuild
                            ├── graph.html                   graphify ask "…"
                            ├── summaries.json               graphify swarm "…"
                            └── cache/embeddings/
```

---

## Architecture — Four Phases + Memory

```
Phase 1 · Indexer        Phase 2 · Graph          Phase 3 · Router         Phase 4 · Swarm
─────────────────────    ─────────────────────    ─────────────────────    ─────────────────────
walk repo files          Python AST parsing        vector search (Qdrant)   Retriever Agent
  ↓                      JS/TS regex                 +                      Reasoner Agent
chunk by logic unit      Markdown headings         graph traversal          Editor Agent
  ↓                      JSON/ADF metadata           ↓                     Validator Agent
embed (384-dim)            ↓                       score threshold            ↓
  ↓                      NetworkX graph              ↓                     structured findings
store in Qdrant          graph.json               any LLM provider         proposed edits
save chunks.jsonl        graph.html (vis.js)      streaming answer         diff + apply
                                                   episodic log
```

**Embedding model:** `all-MiniLM-L6-v2` — 22 M params, 384-dim, ~90 MB download once, then cached.  
**Vector store:** Qdrant — runs in Docker (`docker compose up -d`) or embedded disk mode (fallback).  
**Graph library:** NetworkX MultiDiGraph.  
**LLM:** Auto-detected from `.env`. Priority: Databricks → OpenAI → Anthropic → Google → Ollama.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 + | 3.11 recommended |
| Docker Desktop | any | for Qdrant vector store (`docker compose up -d`) |
| git | any | for `graphify sync` |
| Ollama | any | **optional** — only needed if you have no API keys in `.env` |

---

## Installation

```bash
# Clone this repo
git clone https://github.com/LokAyiti/graphify-swarm
cd graphify-swarm

# Install graphify and all dependencies
pip install -e .

# Start Qdrant in Docker Desktop (must be running first)
docker compose up -d

# If this is a fresh clone (no original source files needed):
graphify rebuild     # rebuilds Qdrant from chunks.jsonl + embedding cache

# Verify
graphify status
graphify ask "what pipelines exist here"
```

**First-time setup on a new machine (you have the source repos):**

```bash
pip install -e .
docker compose up -d                   # start Qdrant
graphify init                          # creates .graphify.json, updates .gitignore
graphify add-repo C:\path\to\repo1    # registers, indexes, and graphs in one step
graphify add-repo C:\path\to\repo2
graphify sync                          # push updated knowledge to GitHub
```

---

## Command Reference

### Orchestration (start here)

| Command | What it does |
|---|---|
| `graphify init` | Create `.graphify.json` + update `.gitignore` |
| `graphify add-repo <path>` | Register repo, auto-index + auto-graph it |
| `graphify rebuild` | Rebuild Qdrant from `chunks.jsonl` + cache (no source files needed) |
| `graphify sync` | Git add/commit/push knowledge outputs to GitHub |

### Phase 1 — Index (vectors)

```bash
# Index specific paths
graphify index ./repo1 ./repo2

# Index all repos registered in .graphify.json
graphify index --all

# Force full re-index
graphify index --all --reindex

# Check what's indexed
graphify status
```

### Phase 2 — Graph (structure)

```bash
# Extract code graph from specific paths
graphify graph ./repo1

# Extract from all registered repos
graphify graph --all

# Open the interactive vis.js visualization in browser
graphify visualize

# Regenerate the text report from existing graph.json
graphify report
```

### Phase 3 — Query (ask questions)

```bash
# Pure vector search — fast, no LLM
graphify query "what does the cashier pipeline do"

# Ask with auto-detected provider (reads .env)
graphify ask "what activities run in the EDR pipelines"

# Explicit provider override
graphify ask "explain error handling" --provider databricks
graphify ask "explain error handling" --provider openai --llm gpt-5
graphify ask "explain error handling" --provider anthropic --llm claude-3-5-sonnet-latest
graphify ask "explain error handling" --provider google  --llm gemini-1.5-pro

# Ollama (local, no API key needed)
graphify ask "explain error handling" --provider ollama --llm llama3

# Strict threshold (0.85 auto-applied for API providers)
graphify ask "find all retry patterns" --threshold 0.88

# Filter to one repo
graphify ask "find all Python classes" --repo myrepo --lang python

# Print merged context only (paste into any external LLM)
graphify ask "question" --context
```

### Phase 4 — Swarm (multi-agent)

```bash
# Analyze with auto-detected provider
graphify swarm "what activities run in the cashier pipelines"

# Explicit provider
graphify swarm "find inconsistencies across EDR pipelines" --provider databricks
graphify swarm "find inconsistencies across EDR pipelines" --provider openai --llm gpt-5

# Edit mode — propose file changes
graphify swarm "add error handling" --mode edit --provider databricks

# Edit mode + apply validated diffs
graphify swarm "fix naming inconsistency" --mode edit --apply --yes
```

### Phase 5 — Memory & Learning

```bash
# After any `ask` or `swarm` — give feedback on the answer
graphify feedback good
graphify feedback bad
graphify feedback corrected --correction "The correct answer is ..."

# Inspect what Graphify has learned
graphify memory status          # DB stats
graphify memory patterns        # learned retrieval patterns
graphify memory feedback        # feedback breakdown
graphify patterns               # shortcut for memory patterns

# Run learning cycle
graphify evolve                 # extract patterns from episodic log
graphify evolve --deep          # full engine: promote rules, decay, prune, drift detection

# System health check
graphify health                 # Qdrant + LLM + Memory all in one
```

### Maintenance

```bash
# Remove one repo from the index
graphify clear --repo pipeline

# Remove everything
graphify clear --yes
```

---

## Output Files

All outputs live in `graphify-out/` and are committed to this repo (except Qdrant binary).

| File | Committed | Description |
|---|---|---|
| `chunks.jsonl` | ✅ | All chunk content — used by `graphify rebuild` |
| `cache/embeddings/` | ❌ | SHA256-keyed numpy vectors — gitignored (large binary files), auto-regenerated |
| `summaries.json` | ✅ | Repo metadata (file counts, chunk counts) |
| `graph.json` | ✅ | NetworkX node-link graph |
| `graph.html` | ✅ | Interactive vis.js visualization |
| `GRAPH_REPORT.md` | ✅ | Top nodes, most-imported modules, suggested questions |
| `qdrant/` | ❌ | Binary Qdrant storage — gitignored, rebuilt locally |

---

## Configuration — `.graphify.json`

```json
{
  "repos": [
    {
      "name": "pipeline",
      "path": "C:\\Users\\you\\Downloads\\pipeline"
    },
    {
      "name": "app-code",
      "path": "C:\\work\\app-code"
    }
  ],
  "default_llm": "llama3",
  "ollama_host": "http://localhost:11434",
  "qdrant_url": null
}
```

> **Qdrant URL:** Controlled via `.env` (takes priority over config).  
> Set `QDRANT_URL=http://localhost:6333` for Docker, or the Qdrant Cloud URL for cloud mode.  
> Leave unset (or `null`) to fall back to embedded local disk mode.

> **Portability note:** `path` is machine-specific (absolute Windows path). Each team member
> runs `graphify add-repo <their-local-path> --name <same-name>` once to set their path.
> Everyone else uses `graphify rebuild` which needs no paths at all.

---

## Team Workflow

### Owner / Maintainer

```bash
# One-time setup
docker compose up -d
graphify init
graphify add-repo C:\path\to\pipeline
graphify add-repo C:\path\to\another-repo

# After adding a new repo or updating existing ones
graphify add-repo C:\path\to\new-repo
graphify sync

# Routine update (re-index changed files)
graphify index --all --reindex
graphify graph --all
graphify sync
```

### Team Member (clone and use)

```bash
git clone https://github.com/LokAyiti/graphify-swarm
cd graphify-swarm
pip install -e .
docker compose up -d
graphify rebuild

# Now query anything
graphify ask "what pipelines are in the ACQ/Sybase folder"
graphify swarm "which pipelines are missing error handling"
graphify visualize
```

### Team Member (has local copy of source repos)

```bash
git clone https://github.com/LokAyiti/graphify-swarm
cd graphify-swarm
pip install -e .
docker compose up -d
graphify add-repo C:\my-local\pipeline --name pipeline    # registers + re-indexes (fast — cache hit)
graphify ask "question"
```

---

## LLM Integration

Graphify auto-detects the LLM provider from `.env` — **no flag required** if keys are set.

### Provider priority (first configured wins)

| Priority | Provider | Env var | Notes |
|---|---|---|---|
| 1 | **Databricks** | `DATABRICKS_TOKEN` + endpoint | Claude via Azure Databricks |
| 2 | **OpenAI** | `OPENAI_API_KEY` | GPT-5, GPT-5-mini, etc. |
| 3 | **Anthropic** | `ANTHROPIC_API_KEY` | Claude direct API |
| 4 | **Google** | `GOOGLE_API_KEY` | Gemini 1.5 Pro/Flash |
| 5 | **Ollama** | *(none)* | Local fallback, free |

### CLI examples

```bash
# Auto-detect from .env (recommended)
graphify ask "how does auth work"
graphify swarm "find all error handling patterns"

# Explicit provider + model
graphify ask "q" --provider databricks --llm databricks-claude-sonnet-4-6
graphify ask "q" --provider openai     --llm gpt-5
graphify ask "q" --provider anthropic  --llm claude-3-5-sonnet-latest
graphify ask "q" --provider google     --llm gemini-1.5-pro
graphify ask "q" --provider ollama     --llm llama3

# Enforce similarity threshold (only high-confidence chunks reach the LLM)
# Auto-set to 0.85 for API providers, off for Ollama
graphify ask "q" --threshold 0.88

# Print context only (paste into any LLM yourself)
graphify ask "q" --context
```

### Configuring `.env`

```ini
# Set the provider you want — leave others commented out
DATABRICKS_TOKEN=dapi...
DATABRICKS_SONNET_ENDPOINT=https://adb-....azuredatabricks.net/serving-endpoints/...

# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=AIza...
```

### Episodic memory

Every `ask` / `swarm` call appends a record to `graphify-out/memory/episodic.jsonl`:

```json
{"ts": "2026-07-02T10:00:00Z", "query": "...", "provider": "databricks",
 "model": "databricks-claude-sonnet-4-6", "chunks_used": 8, "top_score": 0.91,
 "threshold": 0.85, "latency_s": 2.4}
```

This log is the foundation for future semantic caching and pattern learning.

---

## Supported File Types

Graphify extracts rich structure from:

| Language | Extraction |
|---|---|
| Python `.py` | Functions, classes, imports, call graph, inheritance |
| JavaScript/TypeScript `.js .ts .jsx .tsx` | Functions, classes, imports, exports |
| Markdown `.md .mdx` | Heading sections, internal links |
| JSON `.json` | Top-level keys; **ADF pipeline metadata** (name, activities, folder, type) |
| YAML `.yaml .yml` | File node + sliding-window chunks |
| Shell, PowerShell, SQL, Terraform, Bicep, ... | File node + sliding-window chunks |

**Azure Data Factory pipelines** get special treatment: the `properties.activities` array is parsed and stored as graph metadata, so you can query `"what activities run in the cashier pipelines"` and get structured answers even without an LLM.

---

## Project Structure

```
graphify-swarm/
├── graphify/
│   ├── cli.py              ← 13-command CLI entry point
│   ├── config.py           ← .graphify.json management
│   ├── indexer/
│   │   ├── walker.py       ← repo file discovery (.gitignore aware)
│   │   ├── chunker.py      ← Python AST + markdown + sliding-window chunking
│   │   ├── embedder.py     ← sentence-transformers + SHA256 disk cache
│   │   └── qdrant_store.py ← local Qdrant CRUD + semantic search
│   ├── graph/
│   │   ├── extractor.py    ← AST/regex → GNode + GEdge
│   │   ├── builder.py      ← NetworkX graph assembly + stats
│   │   └── visualizer.py   ← graph.json + graph.html (vis.js) + GRAPH_REPORT.md
│   ├── query/
│   │   ├── router.py       ← vector search + graph BFS expansion
│   │   ├── merger.py       ← format merged context + structured system prompt
│   │   └── llm.py          ← multi-provider LLM: Databricks, OpenAI, Anthropic, Google, Ollama
│   ├── memory/
│   │   ├── episodic.py          ← append-only query log (JSONL)
│   │   ├── memory_store.py      ← SQLite: patterns, feedback, rules tables
│   │   ├── feedback_loop.py     ← 3-state feedback + boost/decay
│   │   └── evolution_engine.py  ← self-learning: promote, decay, prune, drift detection
│   └── agents/
│       ├── base.py         ← RunContext, ProposedEdit, BaseAgent
│       ├── retriever.py    ← Agent 1: vector + graph retrieval
│       ├── reasoner.py     ← Agent 2: LLM synthesis or rule-based metadata
│       ├── editor.py       ← Agent 3: LLM diff generation
│       ├── validator.py    ← Agent 4: syntax + path containment + ADF regression check
│       └── swarm.py        ← orchestrator + Rich terminal display
├── graphify-out/
│   ├── chunks.jsonl        ← portable chunk content (committed)
│   ├── cache/embeddings/   ← numpy vector cache (committed)
│   ├── graph.json          ← knowledge graph (committed)
│   ├── graph.html          ← interactive visualization (committed)
│   ├── GRAPH_REPORT.md     ← text report (committed)
│   ├── summaries.json      ← repo metadata (committed)
│   ├── qdrant/             ← Qdrant binary (gitignored — managed by Docker)
│   └── cache/embeddings/   ← numpy vectors (gitignored — auto-regenerated)
├── .graphify.json          ← repo registry + default settings
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

---

## Dependencies

```
qdrant-client>=1.9.0        vector store (Docker or embedded)
sentence-transformers>=3.0.0  all-MiniLM-L6-v2 embeddings
networkx>=3.0               graph data structure
typer[all]>=0.12.0          CLI framework
rich>=13.0.0                terminal display
pathspec>=0.12.0            .gitignore parsing
numpy>=1.24.0               embedding cache (numpy .npy files)
python-dotenv>=1.0.0        .env file loading
```

---

## Troubleshooting

**`graphify rebuild` — "Server disconnected" or connection refused**  
The Qdrant Docker container is not running. Start it with: `docker compose up -d`

**`graphify rebuild` — "chunks.jsonl not found"**  
Someone with the source repos needs to run `graphify index <path>` and `graphify sync` first.

**Ollama not available**  
Ollama is the local fallback — you don't need it if you have an API key in `.env`.
If you do want Ollama: run `ollama serve` in a separate terminal, then `ollama pull llama3`.

**Qdrant is corrupted / behaves unexpectedly**  
```bash
graphify clear --yes
docker compose restart   # restart the container
graphify rebuild
```

**Switch between Local Docker and Qdrant Cloud**  
Edit `.env` — uncomment Option A (Docker) or Option B (Cloud). Re-run `graphify index --all` after switching.

**File content is garbled (UTF-16 / BOM)**  
The chunker and JSON extractor both use `utf-8-sig` encoding which strips UTF-8 BOMs automatically. UTF-16 files (exported from some tools) should be converted first: `Get-Content file.json | Set-Content -Encoding UTF8 file.json`.

**`graphify sync` push fails**  
Ensure your GitHub SSH key or Personal Access Token is configured:  
`git remote -v` → check the remote URL is correct.

---

*Built with Python 3.11 · sentence-transformers · Qdrant · NetworkX · vis.js · Ollama*
