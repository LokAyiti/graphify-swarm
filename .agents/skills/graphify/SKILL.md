# Graphify Skill

A local code knowledge graph tool that indexes git repositories and enables
semantic search, graph traversal, and multi-agent analysis across all of them.

## When to Use This Skill

Use this skill when the user asks to:
- Index git repositories into the knowledge graph
- Ask questions across multiple repos
- Extract and visualize code structure
- Run multi-agent analysis on code
- Push knowledge outputs to GitHub for team sharing
- Rebuild the index on a fresh clone

## Command Quick Reference

```bash
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
git clone https://github.com/LokAyiti/CICD_Automation
pip install -e .
graphify rebuild     # rebuilds Qdrant from committed chunks.jsonl + embeddings
graphify ask "question"
```

## Configuration (.graphify.json)

```json
{
  "repos": [{"name": "pipeline", "path": "C:\\path\\to\\pipeline"}],
  "default_llm": "llama3",
  "ollama_host": "http://localhost:11434"
}
```

## What Gets Committed to GitHub

| File | Why |
|---|---|
| `graphify-out/chunks.jsonl` | Chunk content for offline rebuild |
| `graphify-out/cache/embeddings/` | Numpy vector cache (portable) |
| `graphify-out/graph.json` | Knowledge graph |
| `graphify-out/graph.html` | Interactive visualization |
| `graphify-out/summaries.json` | Repo metadata |
| `graphify-out/GRAPH_REPORT.md` | Analysis report |
| `.graphify.json` | Repo registry |

`graphify-out/qdrant/` is **gitignored** (binary, non-portable — rebuilt locally).

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
