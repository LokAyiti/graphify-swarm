"""
swarm.py — Phase 4 orchestrator.

Chains the four agents in order, tracks timing, and renders a
clean Rich terminal report after each step and at the end.

Usage (from CLI):
    from graphify.agents.swarm import Swarm, build_swarm
    swarm = build_swarm(...)
    ctx   = swarm.run(task="...", mode="analyze")
"""
from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Dict, Optional

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from graphify.agents.base import RunContext
from graphify.agents.editor import EditorAgent
from graphify.agents.reasoner import ReasonerAgent
from graphify.agents.retriever import RetrieverAgent
from graphify.agents.validator import ValidatorAgent


class Swarm:
    """Orchestrates the four Phase-4 agents and handles display."""

    def __init__(
        self,
        retriever: RetrieverAgent,
        reasoner:  ReasonerAgent,
        editor:    EditorAgent,
        validator: ValidatorAgent,
        console:   Console,
    ) -> None:
        self._agents   = [retriever, reasoner, editor, validator]
        self._labels   = ["Retriever", "Reasoner", "Editor", "Validator"]
        self._console  = console

    # ── Main run loop ─────────────────────────────────────────────────────

    def run(self, task: str, mode: str = "analyze", **kwargs) -> RunContext:
        ctx = RunContext(
            task=task,
            mode=mode,
            llm_model=kwargs.get("llm_model"),
            top_k=kwargs.get("top_k", 6),
            repo_filter=kwargs.get("repo_filter"),
        )

        c = self._console

        c.print(Rule("[bold cyan]Graphify Swarm[/]"))
        c.print(f"[bold]Task:[/] {task}")
        c.print(
            f"[dim]Mode:[/] {mode}  "
            f"[dim]Top-K:[/] {ctx.top_k}  "
            f"[dim]LLM:[/] {ctx.llm_model or 'none (rule-based)'}\n"
        )

        for i, (agent, label) in enumerate(zip(self._agents, self._labels), 1):
            with c.status(f"[{i}/4] {label}…", spinner="dots"):
                t0 = time.perf_counter()
                try:
                    ctx = agent.run(ctx)
                except Exception as exc:
                    ctx.errors.append(f"{agent.name}: unhandled error: {exc}")
                elapsed = time.perf_counter() - t0
                ctx.agent_times[agent.name] = round(elapsed, 2)

            self._print_step(i, label, agent.name, ctx, elapsed)

        c.print()
        self._print_result(ctx)
        return ctx

    # ── Step summary line ─────────────────────────────────────────────────

    def _print_step(
        self,
        step: int,
        label: str,
        agent_name: str,
        ctx: RunContext,
        elapsed: float,
    ) -> None:
        c = self._console
        t = f"{elapsed:.1f}s"

        if agent_name == "retriever":
            n_files = len({h.file_path for h in ctx.vector_hits})
            n_gc    = len(ctx.graph_contexts)
            detail  = (
                f"{len(ctx.vector_hits)} chunks · {n_files} file(s) · "
                f"{n_gc} graph context(s)"
            )
            status = "[green]✓[/]"

        elif agent_name == "reasoner":
            if ctx.summary:
                detail = ctx.summary[:80]
                status = "[green]✓[/]"
            else:
                detail = "no output"
                status = "[yellow]⚠[/]"

        elif agent_name == "editor":
            if ctx.mode != "edit":
                detail = "skipped (analyze mode)"
                status = "[dim]⊘[/]"
            elif ctx.proposed_edits:
                detail = f"{len(ctx.proposed_edits)} edit(s) proposed"
                status = "[green]✓[/]"
            else:
                detail = "no edits proposed"
                status = "[dim]—[/]"

        elif agent_name == "validator":
            n_ok = len([e for e in ctx.proposed_edits if e.validated])
            n_bad = len([e for e in ctx.proposed_edits if not e.validated and ctx.proposed_edits])
            detail = (
                f"{ctx.files_checked} file(s) scanned"
                + (f" · {len(ctx.syntax_issues)} issue(s)" if ctx.syntax_issues else " · no issues")
                + (f" · {n_ok}/{len(ctx.proposed_edits)} edits validated" if ctx.proposed_edits else "")
            )
            status = "[green]✓[/]" if not ctx.syntax_issues else "[yellow]⚠[/]"

        else:
            detail = ""
            status = "[green]✓[/]"

        # Agent errors override status
        agent_errors = [e for e in ctx.errors if e.startswith(agent_name)]
        if agent_errors:
            status = "[red]✗[/]"
            detail = agent_errors[-1]

        label_pad = label.ljust(12)
        c.print(f"  [{step}/4] {status} [bold]{label_pad}[/]  {detail}  [dim]{t}[/]")

    # ── Final result panel ────────────────────────────────────────────────

    def _print_result(self, ctx: RunContext) -> None:
        c = self._console
        c.print(Rule("[bold]Result[/]"))

        # ── Summary + Findings ──
        if ctx.summary:
            c.print(f"\n[bold cyan]Summary[/]\n{ctx.summary}\n")

        if ctx.findings:
            c.print("[bold cyan]Findings[/]")
            for f in ctx.findings:
                indent = "   " if f.startswith("  ") else " ● "
                c.print(f"{indent}{f.strip()}")
            c.print()

        if ctx.recommendation:
            c.print(f"[bold cyan]Recommendation[/]\n{ctx.recommendation}\n")

        # ── Proposed edits ──
        if ctx.proposed_edits:
            c.print(Rule("[bold yellow]Proposed Edits[/]"))
            for i, edit in enumerate(ctx.proposed_edits, 1):
                status = "[green]✓ validated[/]" if edit.validated else f"[red]✗ {edit.validation_error}[/]"
                c.print(f"\n  Edit {i}: [bold]{edit.file_path}[/]  {status}")
                c.print(f"  [dim]{edit.description}[/]")

                diff = list(difflib.unified_diff(
                    edit.before.splitlines(keepends=True),
                    edit.after.splitlines(keepends=True),
                    fromfile=f"a/{edit.file_path}",
                    tofile=f"b/{edit.file_path}",
                    n=3,
                ))
                if diff:
                    diff_text = "".join(diff)
                    c.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))
            c.print()

        # ── Syntax issues ──
        if ctx.syntax_issues:
            c.print(Rule("[bold red]Syntax Issues Found[/]"))
            for issue in ctx.syntax_issues[:20]:
                c.print(f"  [red]•[/] {issue}")
            c.print()

        # ── Errors ──
        if ctx.errors:
            c.print(Rule("[bold red]Agent Errors[/]"))
            for err in ctx.errors:
                c.print(f"  [red]•[/] {err}")
            c.print()

        # ── Timing ──
        if ctx.agent_times:
            total = sum(ctx.agent_times.values())
            times = "  ".join(
                f"{name}: {t}s" for name, t in ctx.agent_times.items()
            )
            c.print(f"[dim]Total: {total:.1f}s  |  {times}[/]")

        c.print(Rule())


# ── Factory ───────────────────────────────────────────────────────────────────

def build_swarm(
    qdrant_dir:      Path,
    embedder_cache:  Path,
    graph_json_path: Optional[Path],
    repo_paths:      Dict[str, Path],
    llm_model:       Optional[str],
    ollama_host:     str,
    console:         Console,
) -> Swarm:
    """Construct a fully-wired Swarm ready to run."""

    llm = None
    if llm_model:
        from graphify.query.llm import OllamaLLM
        llm = OllamaLLM(model=llm_model, host=ollama_host)

    return Swarm(
        retriever = RetrieverAgent(qdrant_dir, embedder_cache, graph_json_path),
        reasoner  = ReasonerAgent(llm=llm),
        editor    = EditorAgent(llm=llm),
        validator = ValidatorAgent(repo_paths=repo_paths),
        console   = console,
    )
