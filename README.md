# Graphify — Local Code Knowledge Graph

> Index any number of git repos into a searchable, queryable knowledge graph.  
> Works **100 % offline**. No API keys. No cloud. No Docker.

```
graphify add-repo  ./my-repo          # register + index + extract graph
graphify ask       "how does auth work?"     # semantic search + graph context
graphify swarm     "find duplicate patterns" # 4-agent analysis pipeline
graphify sync                               # push knowledge to GitHub for the team
```

---

## What It Does

Graphify turns local git repositories into a living knowledge base your whole team can query. You index repos once, the outputs are committed to this repository, and anyone who clones it can start asking questions immediately — without touching the original source files.

```
Your repos (local)          CICD_Automation (GitHub)         Any team member
──────────────────          ────────────────────────         ─────────────────
pipeline/                   graphify-out/                    git clone …
app-code/      ──index──▶   ├── chunks.jsonl    ──push──▶   pip install -e .
infra-tf/                   ├── graph.json                   graphify rebuild
                            ├── graph.html                   graphify ask "…"
                            ├── summaries.json               graphify swarm "…"
                            └── cache/embeddings/
```

---

## Architecture — Four Phases

```
Phase 1 · Indexer        Phase 2 · Graph          Phase 3 · Router         Phase 4 · Swarm
─────────────────────    ─────────────────────    ─────────────────────    ─────────────────────
walk repo files          Python AST parsing        vector search (Qdrant)   Retriever Agent
  ↓                      JS/TS regex                 +                      Reasoner Agent
chunk by logic unit      Markdown headings         graph traversal          Editor Agent
  ↓                      JSON/ADF metadata           ↓                     Validator Agent
embed (384-dim)            ↓                       merged context             ↓
  ↓                      NetworkX graph              ↓                     structured findings
store in Qdrant          graph.json               Ollama LLM (optional)    proposed edits
save chunks.jsonl        graph.html (vis.js)      streaming answer         diff + apply
```

**Embedding model:** `all-MiniLM-L6-v2` — 22 M params, 384-dim, ~90 MB download once, then cached.  
**Vector store:** Qdrant local disk mode — no server, no Docker.  
**Graph library:** NetworkX MultiDiGraph.  
**LLM (optional):** Any Ollama model via `http://localhost:11434`.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 + | 3.11 recommended |
| git | any | for `graphify sync` |
| Ollama | any | **optional** — only for `--llm` flag |

---

## Installation

```bash
# Clone this repo
git clone https://github.com/LokAyiti/CICD_Automation
cd CICD_Automation

# Install graphify and all dependencies
pip install -e .

# If this is a fresh clone (no original source files needed):
graphify rebuild     # rebuilds Qdrant from chunks.jsonl + cached embeddings

# Verify
graphify status
graphify ask "what pipelines exist here"
```

**First-time setup on a new machine (you have the source repos):**

```bash
pip install -e .
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

# Vector + graph context (no LLM answer)
graphify ask "what activities run in the EDR pipelines"

# Vector + graph + Ollama answer (streaming)
graphify ask "how do the pipelines handle errors" --llm llama3

# Filter to one repo, restrict language
graphify ask "find all Python classes" --repo myrepo --lang python

# Print only the merged context (for pasting into any LLM)
graphify ask "question" --context
```

### Phase 4 — Swarm (multi-agent)

```bash
# Analyze mode (no edits) — rule-based, works without LLM
graphify swarm "what activities run in the cashier pipelines"

# Analyze with LLM-powered reasoning
graphify swarm "find inconsistencies across EDR pipelines" --llm llama3

# Edit mode — propose file changes
graphify swarm "add error handling to all cashier pipelines" --mode edit --llm llama3

# Edit mode + apply validated diffs to disk
graphify swarm "fix naming inconsistency" --mode edit --llm llama3 --apply --yes
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
| `cache/embeddings/` | ✅ | SHA256-keyed numpy vectors — makes rebuild instant |
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
  "ollama_host": "http://localhost:11434"
}
```

> **Portability note:** `path` is machine-specific (absolute Windows path). Each team member
> runs `graphify add-repo <their-local-path> --name <same-name>` once to set their path.
> Everyone else uses `graphify rebuild` which needs no paths at all.

---

## Team Workflow

### Owner / Maintainer

```bash
# One-time setup
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
git clone https://github.com/LokAyiti/CICD_Automation
cd CICD_Automation
pip install -e .
graphify rebuild

# Now query anything
graphify ask "what pipelines are in the PBNA_FC/ACQ/Sybase folder"
graphify swarm "which pipelines are missing error handling"
graphify visualize
```

### Team Member (has local copy of source repos)

```bash
git clone https://github.com/LokAyiti/CICD_Automation
cd CICD_Automation
pip install -e .
graphify add-repo C:\my-local\pipeline --name pipeline    # registers + re-indexes (fast — cache hit)
graphify ask "question"
```

---

## LLM Integration (Ollama)

Graphify works fully without a model. The `--llm` flag enables AI-powered answers.

```bash
# Install Ollama: https://ollama.com
ollama pull llama3       # or codellama, mistral, phi3, etc.
ollama serve

# Use with any command
graphify ask "explain the duplicate-run check pattern" --llm llama3
graphify swarm "find pipelines missing Wait_For_API_Propagation" --llm llama3

# Custom Ollama host
graphify ask "question" --llm llama3 --host http://192.168.1.10:11434
```

**Available models that work well:**

| Model | Size | Good for |
|---|---|---|
| `llama3` | 4.7 GB | General Q&A, analysis |
| `codellama` | 3.8 GB | Code-specific questions |
| `mistral` | 4.1 GB | Fast, good reasoning |
| `phi3` | 2.3 GB | Lightweight, quick answers |

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
CICD_Automation/
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
│   │   ├── merger.py       ← format merged context for LLM
│   │   └── llm.py          ← Ollama HTTP client (streaming)
│   └── agents/
│       ├── base.py         ← RunContext, ProposedEdit, BaseAgent
│       ├── retriever.py    ← Agent 1: vector + graph retrieval
│       ├── reasoner.py     ← Agent 2: LLM synthesis or rule-based metadata
│       ├── editor.py       ← Agent 3: LLM diff generation
│       ├── validator.py    ← Agent 4: JSON/Python syntax + ADF regression check
│       └── swarm.py        ← orchestrator + Rich terminal display
├── graphify-out/
│   ├── chunks.jsonl        ← portable chunk content (committed)
│   ├── cache/embeddings/   ← numpy vector cache (committed)
│   ├── graph.json          ← knowledge graph (committed)
│   ├── graph.html          ← interactive visualization (committed)
│   ├── GRAPH_REPORT.md     ← text report (committed)
│   ├── summaries.json      ← repo metadata (committed)
│   └── qdrant/             ← Qdrant binary (gitignored — rebuilt locally)
├── .graphify.json          ← repo registry + default settings
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

---

## Dependencies

```
qdrant-client>=1.9.0        vector store (local disk mode)
sentence-transformers>=3.0.0  all-MiniLM-L6-v2 embeddings
networkx>=3.0               graph data structure
typer[all]>=0.12.0          CLI framework
rich>=13.0.0                terminal display
pathspec>=0.12.0            .gitignore parsing
numpy>=1.24.0               embedding cache (numpy .npy files)
```

---

## Troubleshooting

**`graphify rebuild` — "chunks.jsonl not found"**  
Someone with the source repos needs to run `graphify index <path>` and `graphify sync` first.

**Ollama not available**  
Run `ollama serve` in a separate terminal. Check `graphify ask "q" --host http://localhost:11434`.

**Model not installed**  
Run `ollama pull llama3` (or whichever model you want to use).

**Qdrant is corrupted / behaves unexpectedly**  
```bash
graphify clear --yes
graphify rebuild
```

**File content is garbled (UTF-16 / BOM)**  
The chunker and JSON extractor both use `utf-8-sig` encoding which strips UTF-8 BOMs automatically. UTF-16 files (exported from some tools) should be converted first: `Get-Content file.json | Set-Content -Encoding UTF8 file.json`.

**`graphify sync` push fails**  
Ensure your GitHub SSH key or Personal Access Token is configured:  
`git remote -v` → check the remote URL is correct.

---

*Built with Python 3.11 · sentence-transformers · Qdrant · NetworkX · vis.js · Ollama*
