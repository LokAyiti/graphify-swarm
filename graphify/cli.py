"""
graphify CLI â€” Phase 1 + Phase 2

Commands
--------
  graphify index <repo1> [<repo2> â€¦]   index one or more local repos (vectors)
  graphify query "<question>"           semantic search across all indexed repos
  graphify status                       show what's indexed
  graphify clear [--repo NAME]          remove index data

  graphify graph <repo1> [<repo2> â€¦]   extract code graph â†’ graph.json + graph.html + GRAPH_REPORT.md
  graphify visualize                    open graph.html in the default browser
  graphify report                       regenerate GRAPH_REPORT.md from existing graph.json

  graphify ask "<question>"             Phase 3: vector + graph context + optional Ollama answer
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

# Load .env from the cwd where `graphify` is invoked (repo root)
load_dotenv(dotenv_path=Path(".env"), override=False)

app = typer.Typer(
    name="graphify",
    help="Local code knowledge graph â€” index repos, query them semantically.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# ---------------------------------------------------------------------------
# Shared paths  (relative to cwd where `graphify` is invoked)
# ---------------------------------------------------------------------------
OUT_DIR        = Path("graphify-out")
QDRANT_DIR     = OUT_DIR / "qdrant"
CACHE_DIR      = OUT_DIR / "cache"
SUMMARIES_FILE = OUT_DIR / "summaries.json"
CHUNKS_FILE    = OUT_DIR / "chunks.jsonl"     # portable chunk content for rebuild
CONFIG_FILE    = Path(".graphify.json")

_EMBED_BATCH = 64

# Files committed to GitHub (relative to repo root)
_GIT_TRACKED = [
    ".graphify.json",
    "graphify-out/summaries.json",
    "graphify-out/chunks.jsonl",
    "graphify-out/graph.json",
    "graphify-out/graph.html",
    "graphify-out/GRAPH_REPORT.md",
    "graphify-out/cache",        # embedding cache (numpy, portable)
    "graphify/",                 # tool source
    "requirements.txt",
    "pyproject.toml",
]


def _qdrant_url(cfg=None) -> Optional[str]:
    """Env var QDRANT_URL takes priority over .graphify.json."""
    return os.environ.get("QDRANT_URL") or (cfg.qdrant_url if cfg else None)


def _qdrant_api_key() -> Optional[str]:
    return os.environ.get("QDRANT_API_KEY") or None


def _store():
    from graphify.config import load_config
    from graphify.indexer.qdrant_store import QdrantStore
    cfg = load_config(CONFIG_FILE)
    return QdrantStore(QDRANT_DIR, url=_qdrant_url(cfg), api_key=_qdrant_api_key())


def _embedder():
    from graphify.indexer.embedder import Embedder
    return Embedder(CACHE_DIR)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command()
def index(
    repos:   List[str]     = typer.Argument(None,  help="Repo paths (omit to use all repos in .graphify.json)"),
    reindex: bool          = typer.Option(False, "--reindex", help="Clear and re-index even if already indexed"),
    all_:    bool          = typer.Option(False, "--all",     help="Index all repos registered in .graphify.json"),
):
    """Index one or more local repos (chunk â†’ embed â†’ store in Qdrant)."""
    from graphify.config import load_config
    from graphify.indexer.chunker import chunk_file
    from graphify.indexer.embedder import Embedder
    from graphify.indexer.qdrant_store import QdrantStore
    from graphify.indexer.walker import walk_repo

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve repo list
    if all_ or not repos:
        cfg = load_config(CONFIG_FILE)
        if not cfg.repos:
            console.print("[red]No repos in .graphify.json â€” run: graphify add-repo <path>[/]")
            raise typer.Exit(1)
        repo_strs = [r.path for r in cfg.repos if r.path]
        if not repo_strs:
            console.print("[red]Repos in config have no local paths on this machine.\nRun: graphify add-repo <name> <path>[/]")
            raise typer.Exit(1)
    else:
        repo_strs = list(repos)
        cfg = load_config(CONFIG_FILE)

    store    = QdrantStore(QDRANT_DIR, url=_qdrant_url(cfg), api_key=_qdrant_api_key())
    embedder = Embedder(CACHE_DIR)

    summaries: dict = {}
    if SUMMARIES_FILE.exists():
        try:
            summaries = json.loads(SUMMARIES_FILE.read_text())
        except json.JSONDecodeError:
            summaries = {}

    # Load existing chunks.jsonl (to preserve other repos when reindexing one)
    existing_chunks: list[dict] = []
    if CHUNKS_FILE.exists():
        try:
            existing_chunks = [
                json.loads(line)
                for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            existing_chunks = []

    for repo_str in repo_strs:
        repo_path = Path(repo_str).resolve()
        repo_name = repo_path.name

        console.print(Panel(
            f"[white]{repo_path}[/]",
            title=f"[bold cyan]Indexing â€” {repo_name}[/]",
        ))

        if reindex:
            store.delete_repo(repo_name)
            console.print(f"  [yellow]Cleared previous index for '{repo_name}'[/]")

        try:
            files = list(walk_repo(repo_path))
        except (FileNotFoundError, NotADirectoryError) as exc:
            console.print(f"  [red]Error: {exc}[/]")
            continue

        if not files:
            console.print("  [yellow]No indexable files found â€” skipping.[/]")
            continue

        console.print(f"  Found [bold]{len(files)}[/] files\n")

        all_chunks = []
        file_summaries: dict = {}

        # --- Chunking pass ---
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task("Chunking filesâ€¦", total=len(files))
            for fpath in files:
                try:
                    chunks = chunk_file(fpath, repo_path, repo_name)
                    all_chunks.extend(chunks)
                    rel = fpath.relative_to(repo_path).as_posix()
                    file_summaries[rel] = {
                        "chunks": len(chunks),
                        "language": chunks[0].language if chunks else "unknown",
                    }
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]  chunking error {fpath.name}: {exc}[/]")
                prog.advance(task)

        console.print(f"  Chunks produced: [bold green]{len(all_chunks)}[/]")
        console.print("  Embedding (first run downloads ~90 MB model)â€¦\n")

        # --- Embedding + upsert pass ---
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task("Embedding & storingâ€¦", total=len(all_chunks))
            for i in range(0, len(all_chunks), _EMBED_BATCH):
                batch  = all_chunks[i: i + _EMBED_BATCH]
                texts  = [c.content for c in batch]
                vecs   = embedder.embed(texts)
                store.upsert(batch, vecs)
                prog.advance(task, len(batch))

        summaries[repo_name] = {
            "repo_path":    str(repo_path),
            "total_files":  len(files),
            "total_chunks": len(all_chunks),
            "files":        file_summaries,
        }
        SUMMARIES_FILE.write_text(json.dumps(summaries, indent=2))

        # â”€â”€ Save chunk content to chunks.jsonl (portable rebuild source) â”€â”€
        # Remove old entries for this repo, then append fresh ones
        kept = [c for c in existing_chunks if c.get("repo") != repo_name]
        fresh = [
            {
                "repo":       c.repo,
                "file_path":  c.file_path,
                "language":   c.language,
                "chunk_type": c.chunk_type,
                "name":       c.name,
                "content":    c.content,
                "start_line": c.start_line,
                "end_line":   c.end_line,
            }
            for c in all_chunks
        ]
        all_serialised = kept + fresh
        CHUNKS_FILE.write_text(
            "\n".join(json.dumps(c) for c in all_serialised),
            encoding="utf-8",
        )
        existing_chunks = all_serialised   # keep in sync for next repo loop

        console.print(
            f"\n  [bold green]âœ“[/] Done â€” [bold]{len(all_chunks)}[/] chunks stored "
            f"for [cyan]{repo_name}[/]\n"
        )

    console.print(f"[bold]Total vectors in store: {store.count()}[/]")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question"),
    top_k: int    = typer.Option(8, "--top-k", "-k", help="Number of results"),
    repo:   Optional[str] = typer.Option(None, "--repo", "-r", help="Restrict to one repo"),
    lang:   Optional[str] = typer.Option(None, "--lang", "-l", help="Restrict to one language"),
    show_content: bool = typer.Option(True, "--content/--no-content", help="Print best-match content"),
):
    """Semantic search across all indexed repos."""
    store    = _store()
    embedder = _embedder()

    if store.count() == 0:
        console.print("[red]Nothing indexed yet.  Run:  graphify index <path>[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Query:[/] {question}\n")

    query_vec = embedder.embed([question])[0]
    results   = store.search(
        query_vec,
        top_k=top_k,
        repo_filter=repo,
        language_filter=lang,
    )

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit()

    table = Table(title="Results", show_lines=True, expand=False)
    table.add_column("Score",    style="cyan",   width=7,  no_wrap=True)
    table.add_column("Repo",     style="yellow",  width=16)
    table.add_column("File",     style="white",   width=40)
    table.add_column("Symbol",   style="green",   width=22)
    table.add_column("Lines",    width=10)

    for r in results:
        table.add_row(
            f"{r['score']:.3f}",
            r.get("repo", ""),
            r.get("file_path", ""),
            r.get("name", "") or f"[{r.get('chunk_type','')}]",
            f"{r.get('start_line')}â€“{r.get('end_line')}",
        )

    console.print(table)

    if show_content and results:
        top = results[0]
        console.print(Panel(
            top.get("content", "")[:2000],
            title=f"[bold]Best match  Â·  {top.get('file_path')}:{top.get('start_line')}[/]",
            border_style="green",
        ))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status():
    """Show what has been indexed."""
    store = _store()
    total = store.count()

    if not SUMMARIES_FILE.exists() or total == 0:
        console.print("[yellow]No repos indexed yet.  Run:  graphify index <path>[/]")
        return

    summaries = json.loads(SUMMARIES_FILE.read_text())

    table = Table(title="Indexed Repositories", show_lines=True)
    table.add_column("Repo",        style="cyan")
    table.add_column("Path",        style="white")
    table.add_column("Files",       style="yellow",  justify="right")
    table.add_column("Chunks",      style="green",   justify="right")

    for name, info in summaries.items():
        table.add_row(
            name,
            info.get("repo_path", ""),
            str(info.get("total_files",  0)),
            str(info.get("total_chunks", 0)),
        )

    console.print(table)
    console.print(f"\n[bold]Total vectors in Qdrant: {total}[/]")


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

@app.command()
def clear(
    repo: Optional[str] = typer.Option(None, "--repo", help="Clear a specific repo"),
    yes:  bool           = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove index data (default: all)."""
    target_label = f"repo '{repo}'" if repo else "ALL indexed data"
    if not yes:
        ok = typer.confirm(f"Delete {target_label}?")
        if not ok:
            raise typer.Exit()

    store = _store()

    if repo:
        store.delete_repo(repo)
        # Update summaries
        if SUMMARIES_FILE.exists():
            s = json.loads(SUMMARIES_FILE.read_text())
            s.pop(repo, None)
            SUMMARIES_FILE.write_text(json.dumps(s, indent=2))
        console.print(f"[green]Cleared repo '{repo}'.[/]")
    else:
        import shutil
        if QDRANT_DIR.exists():
            shutil.rmtree(QDRANT_DIR)
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        if SUMMARIES_FILE.exists():
            SUMMARIES_FILE.unlink()
        console.print("[green]All index data cleared.[/]")


# ---------------------------------------------------------------------------
# graph  â€” Phase 2
# ---------------------------------------------------------------------------

GRAPH_JSON_FILE   = OUT_DIR / "graph.json"
GRAPH_HTML_FILE   = OUT_DIR / "graph.html"
GRAPH_REPORT_FILE = OUT_DIR / "GRAPH_REPORT.md"


@app.command()
def graph(
    repos: List[str] = typer.Argument(None,  help="Repo paths (omit to use all in .graphify.json)"),
    all_:  bool      = typer.Option(False, "--all", help="Extract graphs from all registered repos"),
):
    """Extract code graph (nodes + edges) â†’ graph.json, graph.html, GRAPH_REPORT.md."""
    from graphify.config           import load_config
    from graphify.graph.builder    import build_graph, graph_stats
    from graphify.graph.extractor  import extract_file
    from graphify.graph.visualizer import write_graph_html, write_graph_json, write_graph_report
    from graphify.indexer.walker   import walk_repo

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve repo list
    if all_ or not repos:
        cfg = load_config(CONFIG_FILE)
        repo_strs = [r.path for r in cfg.repos if r.path]
        if not repo_strs:
            console.print("[red]No repos with local paths. Run: graphify add-repo <path>[/]")
            raise typer.Exit(1)
    else:
        repo_strs = list(repos)

    all_nodes, all_edges = [], []
    repo_names: list[str] = []

    for repo_str in repo_strs:
        repo_path = Path(repo_str).resolve()
        repo_name = repo_path.name
        repo_names.append(repo_name)

        console.print(Panel(
            f"[white]{repo_path}[/]",
            title=f"[bold cyan]Graph extraction â€” {repo_name}[/]",
        ))

        try:
            files = list(walk_repo(repo_path))
        except (FileNotFoundError, NotADirectoryError) as exc:
            console.print(f"  [red]{exc}[/]")
            continue

        if not files:
            console.print("  [yellow]No indexable files found.[/]")
            continue

        console.print(f"  Extracting graph from [bold]{len(files)}[/] filesâ€¦\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task("Extractingâ€¦", total=len(files))
            for fpath in files:
                try:
                    nodes, edges = extract_file(fpath, repo_path, repo_name)
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]  error {fpath.name}: {exc}[/]")
                prog.advance(task)

    if not all_nodes:
        console.print("[red]No nodes extracted â€” nothing to build.[/]")
        raise typer.Exit(1)

    console.print("  Building graphâ€¦")
    G = build_graph(all_nodes, all_edges, repo_names=repo_names)

    stats = graph_stats(G)
    console.print(
        f"  [bold green]âœ“[/] Graph built â€” "
        f"[bold]{stats['total_nodes']}[/] nodes Â· "
        f"[bold]{stats['total_edges']}[/] edges"
    )

    # Print node-type breakdown
    table = Table(show_header=False, box=None, padding=(0, 2))
    for t, c in sorted(stats["node_types"].items()):
        table.add_row(f"  {t}", f"[bold cyan]{c}[/]")
    console.print(table)

    # Write outputs
    write_graph_json(G, GRAPH_JSON_FILE)
    console.print(f"\n  [dim]â†’[/] {GRAPH_JSON_FILE}")

    title = ", ".join(repo_names)
    write_graph_html(G, GRAPH_HTML_FILE, title=title)
    console.print(f"  [dim]â†’[/] {GRAPH_HTML_FILE}")

    write_graph_report(G, GRAPH_REPORT_FILE, repos=repo_names)
    console.print(f"  [dim]â†’[/] {GRAPH_REPORT_FILE}")

    console.print(f"\n[bold green]Done.[/]  Open [cyan]{GRAPH_HTML_FILE}[/] in your browser to explore.")


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------

@app.command()
def visualize():
    """Open graph.html in the default browser."""
    if not GRAPH_HTML_FILE.exists():
        console.print("[red]No graph.html found.  Run:  graphify graph <path>[/]")
        raise typer.Exit(1)

    import webbrowser
    url = GRAPH_HTML_FILE.resolve().as_uri()
    webbrowser.open(url)
    console.print(f"[green]Opened[/] {url}")


# ---------------------------------------------------------------------------
# report  (regenerate from existing graph.json)
# ---------------------------------------------------------------------------

@app.command()
def report():
    """Regenerate GRAPH_REPORT.md from the existing graph.json."""
    if not GRAPH_JSON_FILE.exists():
        console.print("[red]No graph.json found.  Run:  graphify graph <path>[/]")
        raise typer.Exit(1)

    import networkx as nx
    from networkx.readwrite import json_graph

    from graphify.graph.visualizer import write_graph_report

    data  = json.loads(GRAPH_JSON_FILE.read_text())
    G     = json_graph.node_link_graph(data, directed=True, multigraph=True, edges="links")
    repos = sorted({d.get("repo", "") for _, d in G.nodes(data=True) if d.get("type") == "repo"})
    if not repos:
        repos = sorted({d.get("repo", "") for _, d in G.nodes(data=True) if d.get("repo")})

    write_graph_report(G, GRAPH_REPORT_FILE, repos=repos)
    console.print(f"[green]Report written â†’[/] {GRAPH_REPORT_FILE}")


# ---------------------------------------------------------------------------
# ask  â€” Phase 3
# ---------------------------------------------------------------------------

@app.command()
def ask(
    question:     str           = typer.Argument(..., help="Natural-language question"),
    top_k:        int           = typer.Option(8,     "--top-k",   "-k",  help="Vector results"),
    repo:         Optional[str] = typer.Option(None,  "--repo",    "-r",  help="Filter to one repo"),
    llm:          Optional[str] = typer.Option(None,  "--llm",     "-m",  help="Model name (e.g. gpt-4o, llama3, claude-3-5-sonnet-latest)"),
    provider:     Optional[str] = typer.Option(None,  "--provider","-p",  help="LLM provider: databricks|openai|anthropic|google|ollama (auto-detected from .env if omitted)"),
    threshold:    float         = typer.Option(0.0,   "--threshold","-t", help="Min similarity score (0.0=off, 0.85=strict). Auto-set to 0.85 for API providers."),
    no_graph:     bool          = typer.Option(False, "--no-graph",        help="Skip graph traversal"),
    context_only: bool          = typer.Option(False, "--context",         help="Print context and exit"),
    ollama_host:  str           = typer.Option("http://localhost:11434", "--host", help="Ollama host URL"),
):
    """Phase 3 â€” vector search + graph context + LLM answer (auto-detects provider from .env)."""
    import time as _time
    import sys

    from graphify.config import load_config
    from graphify.memory.episodic import log_query
    from graphify.query.llm import build_llm, detect_provider
    from graphify.query.merger import build_llm_messages, format_context
    from graphify.query.router import Router

    t_start = _time.perf_counter()

    # â”€â”€ Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph_path = None
    if not no_graph and GRAPH_JSON_FILE.exists():
        graph_path = GRAPH_JSON_FILE

    cfg    = load_config(CONFIG_FILE)
    router = Router(QDRANT_DIR, CACHE_DIR, graph_json_path=graph_path,
                    qdrant_url=_qdrant_url(cfg), qdrant_api_key=_qdrant_api_key())

    if router.index_count() == 0:
        console.print("[red]Nothing indexed yet.  Run:  graphify index <path>[/]")
        raise typer.Exit(1)

    # â”€â”€ Resolve provider + threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    resolved_provider = provider or detect_provider() or "ollama"
    # Auto-apply 0.85 threshold for API providers unless the user explicitly set one
    effective_threshold = threshold if threshold > 0 else (0.85 if resolved_provider != "ollama" else None)

    console.print(f"\n[bold cyan]Question:[/] {question}")
    console.print(f"[dim]Provider: {resolved_provider}  |  Threshold: {effective_threshold or 'off'}[/]\n")

    # â”€â”€ Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with console.status("[dim]Searching vectorsâ€¦[/]", spinner="dots"):
        hits = router.vector_search(
            question,
            top_k=top_k,
            repo_filter=repo,
            score_threshold=effective_threshold,
        )

    if not hits:
        msg = "No results found"
        if effective_threshold:
            msg += f" above similarity threshold {effective_threshold}"
        console.print(f"[yellow]{msg}.[/]")
        raise typer.Exit()

    # â”€â”€ Graph expand â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph_contexts = []
    if graph_path:
        with console.status("[dim]Expanding graph contextâ€¦[/]", spinner="dots"):
            graph_contexts = router.graph_expand(hits)

    # â”€â”€ Results table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    table = Table(title="Vector Search Results", show_lines=True)
    table.add_column("Score",  style="cyan",   width=7,  no_wrap=True)
    table.add_column("Repo",   style="yellow",  width=15)
    table.add_column("File",   style="white",   width=40)
    table.add_column("Symbol", style="green",   width=22)
    table.add_column("Lines",  width=10)
    for h in hits:
        table.add_row(
            f"{h.score:.3f}",
            h.repo,
            h.file_path,
            h.name or f"[{h.chunk_type}]",
            f"{h.start_line}â€“{h.end_line}",
        )
    console.print(table)

    if graph_contexts:
        console.print(f"\n[dim]Graph traversal:[/] expanded [bold]{len(graph_contexts)}[/] file(s)")
    elif not no_graph and not GRAPH_JSON_FILE.exists():
        console.print("\n[dim yellow]No graph.json â€” run [white]graphify graph <path>[/white] to enable structural context[/]")

    # â”€â”€ Format merged context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ctx_text = format_context(hits, graph_contexts)

    repos_searched = list({h.repo for h in hits})

    if context_only or (llm is None and resolved_provider == "ollama"):
        console.print(Panel(
            ctx_text,
            title="[bold]Merged Context[/]  [dim](add --llm <model> or set an API key in .env)[/]",
            border_style="blue",
        ))
        if llm is None:
            return

    # â”€â”€ Build LLM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    backend = build_llm(
        provider=resolved_provider if resolved_provider != "ollama" else (provider or None),
        model=llm,
        ollama_host=ollama_host,
    )
    resolved_model = getattr(backend, "model", str(backend.__class__.__name__))

    # Ollama validation
    if resolved_provider == "ollama":
        if not backend.is_available():
            console.print(f"\n[red]Ollama not available at {ollama_host}[/]\nStart with: [white]ollama serve[/]")
            raise typer.Exit(1)
        if llm and not backend.model_exists():
            avail = backend.list_models()
            console.print(
                f"\n[red]Model '{llm}' not installed.[/]\n"
                + (f"Available: {', '.join(avail[:6])}\n" if avail else "")
                + f"Pull it: [white]ollama pull {llm}[/]"
            )
            raise typer.Exit(1)

    messages = build_llm_messages(
        context=ctx_text,
        question=question,
        provider=resolved_provider,
        model=resolved_model,
        repos=repos_searched,
        threshold=effective_threshold or 0.0,
    )

    console.rule(f"[bold green]Answer  Â·  {resolved_provider} / {resolved_model}[/]")
    console.print()

    answer_chars = 0
    try:
        for token in backend.ask_stream(messages):
            sys.stdout.write(token)
            sys.stdout.flush()
            answer_chars += len(token)
        sys.stdout.write("\n\n")
        sys.stdout.flush()
    except ConnectionError as exc:
        console.print(f"\n[red]{exc}[/]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"\n[red]LLM error: {exc}[/]")
        raise typer.Exit(1)

    console.rule()

    # â”€â”€ Episodic log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    scores = [h.score for h in hits]
    log_query(
        query=question,
        repos_searched=repos_searched,
        chunks_used=len(hits),
        top_score=max(scores) if scores else 0.0,
        min_score=min(scores) if scores else 0.0,
        threshold=effective_threshold or 0.0,
        provider=resolved_provider,
        model=resolved_model,
        answer_chars=answer_chars,
        latency_s=_time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# swarm  â€” Phase 4
# ---------------------------------------------------------------------------

@app.command()
def swarm(
    task:        str           = typer.Argument(...,   help="Task or question for the swarm"),
    mode:        str           = typer.Option("analyze", "--mode", "-x",  help="analyze | edit"),
    llm:         Optional[str] = typer.Option(None,   "--llm",  "-m",     help="Model name"),
    provider:    Optional[str] = typer.Option(None,   "--provider", "-p", help="LLM provider: databricks|openai|anthropic|google|ollama"),
    threshold:   float         = typer.Option(0.0,    "--threshold", "-t", help="Min similarity score (auto-set to 0.85 for API providers)"),
    top_k:       int           = typer.Option(6,      "--top-k", "-k"),
    repo:        Optional[str] = typer.Option(None,   "--repo", "-r"),
    apply:       bool          = typer.Option(False,  "--apply",           help="Apply validated edits to files (edit mode)"),
    yes:         bool          = typer.Option(False,  "--yes",  "-y",      help="Skip apply confirmation"),
    ollama_host: str           = typer.Option("http://localhost:11434", "--host"),
):
    """Phase 4 â€” multi-agent swarm: Retrieve â†’ Reason â†’ Edit â†’ Validate."""
    from graphify.agents.swarm import build_swarm
    from graphify.config import load_config
    from graphify.query.llm import build_llm, detect_provider

    cfg = load_config(CONFIG_FILE)
    resolved_provider   = provider or cfg.default_provider or detect_provider() or "ollama"
    cfg_threshold       = cfg.score_threshold
    effective_threshold = (
        threshold if threshold > 0
        else cfg_threshold
        if cfg_threshold is not None
        else (0.85 if resolved_provider != "ollama" else None)
    )

    # â”€â”€ Load repo paths from summaries.json â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    repo_paths: dict = {}
    if SUMMARIES_FILE.exists():
        try:
            summaries = json.loads(SUMMARIES_FILE.read_text())
            repo_paths = {
                name: Path(info["repo_path"])
                for name, info in summaries.items()
                if info.get("repo_path")
            }
        except Exception:
            pass

    if not repo_paths:
        console.print("[red]Nothing indexed yet.  Run:  graphify index <path>[/]")
        raise typer.Exit(1)

    # â”€â”€ Validate mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if mode not in ("analyze", "edit"):
        console.print("[red]--mode must be 'analyze' or 'edit'[/]")
        raise typer.Exit(1)

    # â”€â”€ LLM availability check (Ollama only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if llm and resolved_provider == "ollama":
        from graphify.query.llm import OllamaLLM
        backend = OllamaLLM(model=llm, host=ollama_host)
        if not backend.is_available():
            console.print(f"[red]Ollama not available at {ollama_host}[/]\nStart with: [white]ollama serve[/]")
            raise typer.Exit(1)
        if not backend.model_exists():
            avail = backend.list_models()
            console.print(
                f"[red]Model '{llm}' not installed.[/]\n"
                + (f"Available: {', '.join(avail[:6])}\n" if avail else "")
                + f"Pull with: [white]ollama pull {llm}[/]"
            )
            raise typer.Exit(1)

    # â”€â”€ Build LLM backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    llm_backend = None
    if llm or resolved_provider != "ollama":
        llm_backend = build_llm(
            provider=resolved_provider if resolved_provider != "ollama" else None,
            model=llm,
            ollama_host=ollama_host,
        )

    # â”€â”€ Build graph path if available â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph_path = GRAPH_JSON_FILE if GRAPH_JSON_FILE.exists() else None

    # â”€â”€ Build and run swarm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    s = build_swarm(
        qdrant_dir       = QDRANT_DIR,
        embedder_cache   = CACHE_DIR,
        graph_json_path  = graph_path,
        repo_paths       = repo_paths,
        llm_backend      = llm_backend,
        qdrant_url       = _qdrant_url(cfg),
        qdrant_api_key   = _qdrant_api_key(),
        score_threshold  = effective_threshold,
        console          = console,
    )

    ctx = s.run(
        task=task,
        mode=mode,
        llm_model=llm,
        top_k=top_k,
        repo_filter=repo,
    )

    # â”€â”€ Episodic log for swarm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    import time as _swarm_time
    from graphify.memory.episodic import log_query as _log_swarm
    _hits   = ctx.vector_hits or []
    _scores = [h.score for h in _hits]
    _repos  = list({h.repo for h in _hits})
    _model  = getattr(llm_backend, "model", llm or "")
    _log_swarm(
        query          = task,
        repos_searched = _repos,
        chunks_used    = len(_hits),
        top_score      = max(_scores) if _scores else 0.0,
        min_score      = min(_scores) if _scores else 0.0,
        threshold      = effective_threshold or 0.0,
        provider       = resolved_provider,
        model          = _model,
        answer_chars   = len(ctx.summary or "") + len(ctx.reasoning or ""),
        latency_s      = sum(ctx.agent_times.values()),
    )

    # â”€â”€ Apply edits if requested â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    validated_edits = [e for e in ctx.proposed_edits if e.validated]
    if apply and validated_edits:
        console.print(Rule("[bold yellow]Apply Edits[/]"))

        if not yes:
            ok = typer.confirm(
                f"Apply {len(validated_edits)} validated edit(s) to disk?"
            )
            if not ok:
                console.print("[yellow]Aborted â€” no files changed.[/]")
                raise typer.Exit()

        for edit in validated_edits:
            root      = repo_paths.get(edit.repo, Path(".")).resolve()
            full_path = (root / edit.file_path).resolve()

            # â”€â”€ Security: path containment check (issue #2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if not full_path.is_relative_to(root):
                console.print(
                    f"  [red]âœ—[/] {edit.file_path} â€” "
                    f"path escapes repo root; skipping (possible prompt injection)"
                )
                continue

            if not full_path.exists():
                console.print(f"  [red]âœ—[/] {edit.file_path} â€” file not found")
                continue

            current = full_path.read_text(encoding="utf-8-sig", errors="replace")

            # â”€â”€ Safety: reject empty BEFORE which would prepend blindly (issue #6)
            if not edit.before:
                console.print(
                    f"  [yellow]âš [/] {edit.file_path} â€” "
                    f"BEFORE block is empty; skipping to prevent file corruption"
                )
                continue

            if edit.before not in current:
                console.print(
                    f"  [yellow]âš [/] {edit.file_path} â€” "
                    f"original text not found verbatim; skipping"
                )
                continue

            new_content = current.replace(edit.before, edit.after, 1)
            full_path.write_text(new_content, encoding="utf-8")
            console.print(f"  [green]âœ“[/] Applied to {full_path}")

    elif apply and not validated_edits:
        console.print("[yellow]No validated edits to apply.[/]")


# ---------------------------------------------------------------------------
# init  â€” set up graphify in the current git repo
# ---------------------------------------------------------------------------

@app.command()
def init():
    """Initialise graphify: create .graphify.json and update .gitignore."""
    from graphify.config import GraphifyConfig, save_config

    # Create config if missing
    if CONFIG_FILE.exists():
        console.print(f"[yellow].graphify.json already exists[/]")
    else:
        save_config(GraphifyConfig())
        console.print(f"[green]Created[/] .graphify.json")

    # Update .gitignore
    gi = Path(".gitignore")
    lines_to_add = [
        "",
        "# graphify â€” generated outputs (non-portable; rebuilt locally)",
        "graphify-out/qdrant/",
        "graphify-out/cache/embeddings/",
        "graphify-out/memory/",
        "graphify_local.egg-info/",
        "*.egg-info/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".env",
    ]
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    new_lines = [l for l in lines_to_add if l and l not in existing]
    if new_lines:
        with gi.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines_to_add) + "\n")
        console.print(f"[green]Updated[/] .gitignore  (+{len(new_lines)} entries)")
    else:
        console.print("[dim].gitignore already up to date[/]")

    console.print(
        "\n[bold]Next steps:[/]\n"
        "  [cyan]graphify add-repo <path/to/repo>[/]   register a repo\n"
        "  [cyan]graphify index --all[/]               index all registered repos\n"
        "  [cyan]graphify graph --all[/]               extract graphs\n"
        "  [cyan]graphify sync[/]                      push to GitHub\n"
    )


# ---------------------------------------------------------------------------
# add-repo  â€” register a repo and optionally index + graph it
# ---------------------------------------------------------------------------

@app.command(name="add-repo")
def add_repo(
    path:     str           = typer.Argument(..., help="Local path to the git repo"),
    name:     Optional[str] = typer.Option(None, "--name", "-n", help="Override repo name"),
    no_index: bool          = typer.Option(False, "--no-index", help="Skip auto-indexing"),
    no_graph: bool          = typer.Option(False, "--no-graph", help="Skip auto-graph extraction"),
):
    """Register a repo in .graphify.json, then index and graph it."""
    from graphify.config import load_config, save_config, upsert_repo

    repo_path = Path(path).resolve()
    if not repo_path.exists():
        console.print(f"[red]Path not found: {repo_path}[/]")
        raise typer.Exit(1)

    cfg   = load_config(CONFIG_FILE)
    entry = upsert_repo(cfg, repo_path, name=name)
    save_config(cfg)
    console.print(f"[green]Registered[/] '{entry.name}' â†’ {entry.path}")

    if not no_index:
        console.print()
        index(repos=[entry.path], reindex=False, all_=False)

    if not no_graph:
        console.print()
        graph(repos=[entry.path], all_=False)


# ---------------------------------------------------------------------------
# rebuild  â€” rebuild Qdrant from chunks.jsonl + embedding cache (no source files)
# ---------------------------------------------------------------------------

@app.command()
def rebuild():
    """Rebuild Qdrant index from chunks.jsonl + embedding cache (works after git clone)."""
    from graphify.config import load_config
    from graphify.indexer.chunker import Chunk
    from graphify.indexer.embedder import Embedder
    from graphify.indexer.qdrant_store import QdrantStore

    cfg = load_config(CONFIG_FILE)

    if not CHUNKS_FILE.exists():
        console.print(
            "[red]chunks.jsonl not found.[/]\n"
            "Run [white]graphify index <path>[/] to build it first, "
            "then commit and push with [white]graphify sync[/]."
        )
        raise typer.Exit(1)

    raw_chunks = [
        json.loads(line)
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not raw_chunks:
        console.print("[yellow]chunks.jsonl is empty â€” nothing to rebuild.[/]")
        raise typer.Exit()

    console.print(f"  Rebuilding from [bold]{len(raw_chunks)}[/] chunksâ€¦")
    console.print("  (cached embeddings used where available; new ones computed otherwise)\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Wipe and recreate Qdrant (embedded only â€” Docker/Cloud data persists)
    import shutil
    if not _qdrant_url(cfg) and QDRANT_DIR.exists():
        shutil.rmtree(QDRANT_DIR)

    store    = QdrantStore(QDRANT_DIR, url=_qdrant_url(cfg), api_key=_qdrant_api_key())
    embedder = Embedder(CACHE_DIR)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("Embedding & storingâ€¦", total=len(raw_chunks))
        for i in range(0, len(raw_chunks), _EMBED_BATCH):
            batch_data = raw_chunks[i: i + _EMBED_BATCH]
            chunks = [
                Chunk(
                    repo=c["repo"], file_path=c["file_path"],
                    language=c.get("language", ""), chunk_type=c.get("chunk_type", "window"),
                    name=c.get("name", ""), content=c["content"],
                    start_line=c.get("start_line", 0), end_line=c.get("end_line", 0),
                )
                for c in batch_data
            ]
            texts = [c.content for c in chunks]
            vecs  = embedder.embed(texts)
            store.upsert(chunks, vecs)
            prog.advance(task, len(chunks))

    console.print(
        f"\n[bold green]âœ“[/] Rebuilt â€” [bold]{store.count()}[/] vectors in Qdrant\n"
        f"[dim]Run [white]graphify status[/] or [white]graphify ask \"question\"[/] to verify.[/]"
    )


# ---------------------------------------------------------------------------
# sync  â€” commit graphify outputs and push to GitHub
# ---------------------------------------------------------------------------

@app.command()
def sync(
    message: str  = typer.Option(
        "", "--message", "-m",
        help="Commit message (auto-generated if omitted)",
    ),
    push:    bool = typer.Option(True,  "--push/--no-push", help="Push after commit"),
    remote:  str  = typer.Option("origin", "--remote"),
    branch:  str  = typer.Option("",       "--branch",
                                 help="Branch to push to (default: current branch)"),
):
    """Commit graphify knowledge outputs and push to GitHub."""
    import subprocess

    def _git(*args: str) -> tuple[int, str]:
        r = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, cwd=Path(".").resolve()
        )
        return r.returncode, (r.stdout + r.stderr).strip()

    # â”€â”€ Sanity checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    code, out = _git("rev-parse", "--is-inside-work-tree")
    if code != 0:
        console.print("[red]Not inside a git repository.[/]")
        raise typer.Exit(1)

    # â”€â”€ Stage tracked files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    staged: list[str] = []
    for pattern in _GIT_TRACKED:
        p = Path(pattern)
        if p.exists():
            _git("add", pattern)
            staged.append(pattern)
        else:
            console.print(f"  [dim]skip (not found): {pattern}[/]")

    if not staged:
        console.print("[yellow]Nothing to stage.[/]")
        raise typer.Exit()

    console.print(f"  Staged [bold]{len(staged)}[/] path(s)")

    # â”€â”€ Check if anything actually changed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    code, diff_out = _git("diff", "--cached", "--stat")
    if not diff_out.strip():
        console.print("[yellow]Nothing changed since last commit â€” skipping.[/]")
        raise typer.Exit()

    # â”€â”€ Build auto commit message â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not message:
        if SUMMARIES_FILE.exists():
            try:
                s = json.loads(SUMMARIES_FILE.read_text())
                repo_names = list(s.keys())
                total_chunks = sum(v.get("total_chunks", 0) for v in s.values())
                message = (
                    f"graphify: update knowledge graph "
                    f"[{', '.join(repo_names)}] "
                    f"â€” {total_chunks} chunks"
                )
            except Exception:
                message = "graphify: update knowledge graph"
        else:
            message = "graphify: update knowledge graph"

    # â”€â”€ Commit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    code, out = _git("commit", "-m", message)
    if code != 0:
        console.print(f"[red]git commit failed:[/]\n{out}")
        raise typer.Exit(1)
    console.print(f"[green]Committed:[/] {message}")

    if not push:
        raise typer.Exit()

    # â”€â”€ Push â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    push_args = ["push", remote]
    if branch:
        push_args.append(branch)
    code, out = _git(*push_args)
    if code != 0:
        console.print(f"[red]git push failed:[/]\n{out}")
        console.print("[dim]Tip: check your GitHub auth (SSH key or token)[/]")
        raise typer.Exit(1)
    console.print(f"[bold green]âœ“ Pushed to {remote}[/]")
    console.print(
        f"\n[dim]Team members can now:[/]\n"
        f"  git clone https://github.com/LokAyiti/graphify-swarm\n"
        f"  pip install -e .\n"
        f"  graphify rebuild            # rebuild Qdrant from chunks.jsonl\n"
        f"  graphify ask \"your question\"\n"
    )


# ---------------------------------------------------------------------------
# feedback  â€” Phase 5B
# ---------------------------------------------------------------------------

@app.command()
def feedback(
    rating:     str           = typer.Argument(..., help="good | bad | corrected"),
    correction: Optional[str] = typer.Option(None, "--correction", "-c",
                                             help="Your corrected answer (use with 'corrected')"),
):
    """Record feedback on the last answer to improve future retrievals."""
    from graphify.memory.episodic    import get_last_query
    from graphify.memory.feedback_loop import submit_feedback

    if rating not in ("good", "bad", "corrected"):
        console.print("[red]Rating must be: good | bad | corrected[/]")
        raise typer.Exit(1)

    if rating == "corrected" and not correction:
        console.print("[red]--correction is required when rating is 'corrected'[/]")
        raise typer.Exit(1)

    last_entry = get_last_query()
    if last_entry is None:
        console.print("[yellow]No queries logged yet â€” run graphify ask first.[/]")
        raise typer.Exit()

    result = submit_feedback(
        query=last_entry.get("query", ""),
        repos=last_entry.get("repos_searched", []),
        provider=last_entry.get("provider", ""),
        model=last_entry.get("model", ""),
        rating=rating,
        correction=correction,
    )

    icon  = {"good": "âœ“", "bad": "âœ—", "corrected": "â†º"}.get(rating, "?")
    color = {"good": "green", "bad": "red", "corrected": "yellow"}.get(rating, "white")
    console.print(f"[{color}]{icon}[/] Feedback recorded â€” [{color}]{rating}[/] (id={result['feedback_id']})")
    if result["promotions"]:
        for p in result["promotions"]:
            console.print(f"  [bold cyan]ðŸŽ“ Promoted:[/] {p}")

    console.print(f"\n[dim]Query:[/] {last_entry.get('query', '')[:80]}")


# ---------------------------------------------------------------------------
# memory  â€” Phase 5B
# ---------------------------------------------------------------------------

@app.command(name="memory")
def memory_cmd(
    subcommand: str = typer.Argument("status", help="status | patterns | feedback"),
    repo:       Optional[str] = typer.Option(None, "--repo", "-r"),
):
    """Inspect Graphify's learned memory (status / patterns / feedback)."""
    from graphify.memory.memory_store import feedback_stats, get_patterns, memory_summary

    sub = subcommand.lower()

    if sub == "status":
        info = memory_summary()
        console.print("\n[bold cyan]Graphify Memory Status[/]\n")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("Patterns learned",  f"[bold]{info['patterns']}[/]")
        table.add_row("Rules promoted",    f"[bold]{info['rules']}[/]")
        table.add_row("Feedback total",    f"[bold]{info['feedback_total']}[/]")
        table.add_row(
            "  good / bad",
            f"[green]{info['feedback_good']}[/] / [red]{info['feedback_bad']}[/]"
        )
        table.add_row("DB path", f"[dim]{info['db_path']}[/]")
        console.print(table)

    elif sub == "patterns":
        patterns = get_patterns(repo=repo, min_hits=1)
        if not patterns:
            console.print("[yellow]No patterns learned yet.  Run some queries to build up history.[/]")
            return
        table = Table(title=f"Learned Patterns{' â€” ' + repo if repo else ''}", show_lines=True)
        table.add_column("Repo",       style="cyan",  width=14)
        table.add_column("Keywords",   style="white", width=30)
        table.add_column("Chunks",     style="yellow")
        table.add_column("Hits",       style="green",  justify="right")
        table.add_column("Avg score",  style="cyan",   justify="right")
        for p in patterns[:30]:
            table.add_row(
                p["repo"],
                ", ".join(p["keywords"][:5]),
                ", ".join(p["chunk_types"][:4]),
                str(p["hit_count"]),
                f"{p['avg_score']:.3f}",
            )
        console.print(table)

    elif sub == "feedback":
        stats = feedback_stats()
        console.print(
            f"\n[bold]Feedback summary:[/]  "
            f"[green]{stats['good']} good[/]  "
            f"[red]{stats['bad']} bad[/]  "
            f"[yellow]{stats['corrected']} corrected[/]  "
            f"(total: {stats['total']})"
        )
    else:
        console.print(f"[red]Unknown subcommand '{subcommand}'. Use: status | patterns | feedback[/]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# patterns  â€” Phase 5B (shortcut to `memory patterns`)
# ---------------------------------------------------------------------------

@app.command()
def patterns(
    repo:     Optional[str] = typer.Option(None, "--repo", "-r"),
    min_hits: int            = typer.Option(2, "--min-hits", help="Min hit count to show"),
):
    """Show all discovered retrieval patterns (shortcut for `memory patterns`)."""
    from graphify.memory.memory_store import get_patterns
    found = get_patterns(repo=repo, min_hits=min_hits)
    if not found:
        console.print("[yellow]No patterns match the filter.[/]")
        return
    table = Table(title="Learned Patterns", show_lines=True)
    table.add_column("Repo",      style="cyan",  width=16)
    table.add_column("Keywords",  style="white", width=32)
    table.add_column("Hits",      style="green", justify="right")
    table.add_column("Avg score", style="cyan",  justify="right")
    for p in found:
        table.add_row(p["repo"], ", ".join(p["keywords"][:6]),
                      str(p["hit_count"]), f"{p['avg_score']:.3f}")
    console.print(table)


# ---------------------------------------------------------------------------
# evolve  â€” Phase 5B: trigger batch learning from episodic log
# ---------------------------------------------------------------------------

@app.command()
def evolve(
    deep:      bool  = typer.Option(False, "--deep",       help="Run full evolution engine (promote, decay, prune, drift detection)"),
    min_score: float = typer.Option(0.80,  "--min-score",  help="Minimum avg score to record a pattern (shallow mode only)"),
    top_k:     int   = typer.Option(100,   "--top-k",      help="Number of recent episodes to analyse (shallow mode only)"),
):
    """Analyse episodic log and extract reusable retrieval patterns.

    Use --deep to run the full evolution engine: pattern promotion,
    rule decay, pruning, and drift detection.
    """
    if deep:
        from graphify.memory.evolution_engine import run_evolution
        with console.status("[dim]Running full evolution engineâ€¦[/]", spinner="dots"):
            report = run_evolution()

        console.print(f"\n[bold green]âœ“[/] Evolution complete in [bold]{report['duration_s']}s[/]\n")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("Episodes analysed",  f"[bold]{report['episodes_analysed']}[/]")
        table.add_row("Patterns upserted",  f"[bold]{report['patterns_upserted']}[/]")
        table.add_row("Rules promoted",     f"[bold green]{report['rules_promoted']}[/]")
        table.add_row("Rules decayed",      f"[dim]{report['rules_decayed']}[/]")
        table.add_row("Rules pruned",       f"[dim red]{report['rules_pruned']}[/]")
        table.add_row("Total patterns",     f"[bold]{report['total_patterns']}[/]")
        table.add_row("Total rules",        f"[bold]{report['total_rules']}[/]")
        fb = report["feedback_summary"]
        table.add_row(
            "Feedback",
            f"[green]{fb['good']} good[/]  [red]{fb['bad']} bad[/]  (total: {fb['total']})"
        )
        console.print(table)

        if report["repos_drifting"]:
            console.print("\n[bold yellow]âš  Drift detected â€” consider re-indexing:[/]")
            for d in report["repos_drifting"]:
                console.print(
                    f"  [yellow]â€¢[/] [cyan]{d['repo']}[/]  "
                    f"avg_score dropped {d['drop_pct']}%  "
                    f"({d['old_avg']} â†’ {d['new_avg']})"
                )
            console.print(
                "\n[dim]Run: [white]graphify index --all --reindex[/] to refresh affected repos[/]"
            )
        return

    # â”€â”€ Shallow mode (original behaviour) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from graphify.memory.episodic    import get_last_query
    from graphify.memory.memory_store import record_pattern
    from graphify.memory.episodic    import _LOG_FILE

    if not _LOG_FILE.exists():
        console.print("[yellow]No episodic log found.  Run some queries first.[/]")
        raise typer.Exit()

    lines = [l for l in _LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    episodes = []
    for line in lines[-top_k:]:
        try:
            episodes.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not episodes:
        console.print("[yellow]No usable episodes in log.[/]")
        raise typer.Exit()

    new_patterns = 0
    for ep in episodes:
        if ep.get("top_score", 0) < min_score:
            continue
        repos    = ep.get("repos_searched", [])
        query    = ep.get("query", "")
        keywords = list(dict.fromkeys(
            w for w in query.lower().split() if len(w) > 3
        ))[:8]
        if not keywords or not repos:
            continue
        for repo in repos:
            record_pattern(
                repo=repo,
                keywords=keywords,
                chunk_types=[],
                languages=[],
                avg_score=ep.get("top_score", 0.0),
            )
            new_patterns += 1

    console.print(
        f"[bold green]âœ“[/] Evolved â€” processed [bold]{len(episodes)}[/] episodes, "
        f"updated [bold]{new_patterns}[/] pattern(s).\n"
        f"[dim]Run [white]graphify patterns[/] to see results.  "
        f"Use [white]--deep[/] for full promotion + drift detection.[/]"
    )


# ---------------------------------------------------------------------------
# health  â€” Phase 5B: system health check
# ---------------------------------------------------------------------------

@app.command()
def health():
    """Check Qdrant, LLM provider, and memory are all reachable and configured."""
    from graphify.config import load_config
    from graphify.memory.memory_store import memory_summary
    from graphify.query.llm import detect_provider

    cfg      = load_config(CONFIG_FILE)
    ok_icon  = "[bold green]âœ“[/]"
    fail_icon = "[bold red]âœ—[/]"
    warn_icon = "[bold yellow]âš [/]"

    console.print("\n[bold cyan]Graphify Health Check[/]\n")

    # â”€â”€ Qdrant â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    qdrant_url = _qdrant_url(cfg) or "embedded (local disk)"
    try:
        store = _store()
        count = store.count()
        console.print(f"  {ok_icon}  Qdrant     {qdrant_url}  â†’  {count} vectors")
    except Exception as exc:
        console.print(f"  {fail_icon}  Qdrant     {qdrant_url}  â†’  [red]{exc}[/]")

    # â”€â”€ LLM provider â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    provider = detect_provider()
    if provider:
        console.print(f"  {ok_icon}  LLM        Provider detected: [cyan]{provider}[/]")
    else:
        # Check if partial Databricks config exists (token but no endpoint)
        import os as _os
        if _os.environ.get("DATABRICKS_TOKEN") and not (
            _os.environ.get("DATABRICKS_SONNET_ENDPOINT")
            or _os.environ.get("DATABRICKS_OPUS_ENDPOINT")
        ):
            console.print(f"  {warn_icon}  LLM        DATABRICKS_TOKEN found but no endpoint set â€” add DATABRICKS_SONNET_ENDPOINT to .env")
        else:
            console.print(f"  {warn_icon}  LLM        No API key found in .env â€” will use Ollama (local)")
        import urllib.request as _ur
        try:
            _ur.urlopen("http://localhost:11434/api/tags", timeout=2)
            console.print(f"  {ok_icon}  Ollama     running at http://localhost:11434")
        except Exception:
            console.print(f"  {fail_icon}  Ollama     not reachable â€” start with: [white]ollama serve[/]")

    # â”€â”€ Memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        info = memory_summary()
        console.print(
            f"  {ok_icon}  Memory     {info['patterns']} patterns  "
            f"{info['feedback_total']} feedback entries  "
            f"({info['db_path']})"
        )
    except Exception as exc:
        console.print(f"  {warn_icon}  Memory     not initialised yet ({exc})")

    # â”€â”€ Episodic log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from graphify.memory.episodic import _LOG_FILE
    if _LOG_FILE.exists():
        lines = [l for l in _LOG_FILE.read_text().splitlines() if l.strip()]
        console.print(f"  {ok_icon}  Episodic   {len(lines)} queries logged  ({_LOG_FILE})")
    else:
        console.print(f"  {warn_icon}  Episodic   no queries logged yet")

    console.print()


# ---------------------------------------------------------------------------
# chat  â€” interactive conversation loop (Phase 6)
# ---------------------------------------------------------------------------

@app.command()
def chat(
    provider:    Optional[str] = typer.Option(None,  "--provider", "-p",
                                              help="LLM provider (auto-detected from .env)"),
    llm:         Optional[str] = typer.Option(None,  "--llm",      "-m",
                                              help="Model name"),
    repo:        Optional[str] = typer.Option(None,  "--repo",     "-r",
                                              help="Restrict search to one repo"),
    threshold:   float         = typer.Option(0.0,   "--threshold", "-t",
                                              help="Similarity threshold (auto-set for API providers)"),
    top_k:       int           = typer.Option(6,     "--top-k",    "-k"),
    no_graph:    bool          = typer.Option(False,  "--no-graph",
                                              help="Skip graph context"),
    ollama_host: str           = typer.Option("http://localhost:11434", "--host"),
):
    """Interactive chat â€” type questions freely, Ctrl+C or 'exit' to quit.

    Keeps the last 6 exchanges as conversation history so follow-up
    questions like 'what about the second file?' work correctly.
    """
    import sys
    import time as _time
    from graphify.config import load_config
    from graphify.memory.episodic import log_query
    from graphify.query.llm import build_llm, detect_provider
    from graphify.query.merger import build_llm_messages, format_context
    from graphify.query.router import Router
    from rich.rule import Rule

    cfg                 = load_config(CONFIG_FILE)
    resolved_provider   = provider or cfg.default_provider or detect_provider() or "ollama"
    cfg_threshold       = cfg.score_threshold
    effective_threshold = (
        threshold if threshold > 0
        else cfg_threshold if cfg_threshold is not None
        else (0.85 if resolved_provider != "ollama" else None)
    )

    graph_path = None if no_graph else (GRAPH_JSON_FILE if GRAPH_JSON_FILE.exists() else None)
    router     = Router(QDRANT_DIR, CACHE_DIR, graph_json_path=graph_path,
                        qdrant_url=_qdrant_url(cfg), qdrant_api_key=_qdrant_api_key())

    if router.index_count() == 0:
        console.print("[red]Nothing indexed yet.  Run:  graphify index <path>[/]")
        raise typer.Exit(1)

    try:
        backend = build_llm(
            provider    = resolved_provider if resolved_provider != "ollama" else None,
            model       = llm,
            ollama_host = ollama_host,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    resolved_model = getattr(backend, "model", resolved_provider)

    console.print(Rule("[bold cyan]Graphify Chat[/]"))
    console.print(
        f"  [dim]Provider:[/] [cyan]{resolved_provider}[/]  "
        f"[dim]Model:[/] [cyan]{resolved_model}[/]  "
        f"[dim]Threshold:[/] [cyan]{effective_threshold or 'off'}[/]"
        + (f"  [dim]Repo:[/] [cyan]{repo}[/]" if repo else "")
    )
    console.print("[dim]  Type your question. Press Ctrl+C or type 'exit' to quit.[/]\n")

    # Conversation history: list of {"role": ..., "content": ...}
    history: list[dict] = []
    _MAX_HISTORY = 6   # keep last 3 exchanges (6 messages: 3 user + 3 assistant)

    while True:
        try:
            question = input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        t_start = _time.perf_counter()

        # â”€â”€ Retrieve context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with console.status("[dim]Searchingâ€¦[/]", spinner="dots"):
            hits = router.vector_search(
                question, top_k=top_k,
                repo_filter=repo, score_threshold=effective_threshold,
            )
            graph_contexts = router.graph_expand(hits) if graph_path else []

        if not hits:
            console.print("[yellow]No relevant context found for that question.[/]\n")
            continue

        repos_found = list({h.repo for h in hits})
        ctx_text    = format_context(hits, graph_contexts)

        # â”€â”€ Build messages with history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # System + context in the first user message; then interleave history
        system_msgs = build_llm_messages(
            context   = ctx_text,
            question  = question,
            provider  = resolved_provider,
            model     = resolved_model,
            repos     = repos_found,
            threshold = effective_threshold or 0.0,
        )

        if history:
            # Inject conversation history between system and current user message
            messages = [system_msgs[0]] + history[-_MAX_HISTORY:] + [system_msgs[-1]]
        else:
            messages = system_msgs

        # â”€â”€ Stream answer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        console.print(f"\n[bold cyan]Graphify[/] [dim]({resolved_provider})[/]")
        answer_tokens: list[str] = []
        try:
            for token in backend.ask_stream(messages):
                sys.stdout.write(token)
                sys.stdout.flush()
                answer_tokens.append(token)
            sys.stdout.write("\n\n")
            sys.stdout.flush()
        except ConnectionError as exc:
            console.print(f"\n[red]{exc}[/]\n")
            continue
        except Exception as exc:
            console.print(f"\n[red]LLM error: {exc}[/]\n")
            continue

        answer = "".join(answer_tokens)

        # â”€â”€ Update history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > _MAX_HISTORY:
            history = history[-_MAX_HISTORY:]

        # â”€â”€ Episodic log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        scores = [h.score for h in hits]
        log_query(
            query          = question,
            repos_searched = repos_found,
            chunks_used    = len(hits),
            top_score      = max(scores) if scores else 0.0,
            min_score      = min(scores) if scores else 0.0,
            threshold      = effective_threshold or 0.0,
            provider       = resolved_provider,
            model          = resolved_model,
            answer_chars   = len(answer),
            latency_s      = _time.perf_counter() - t_start,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()

