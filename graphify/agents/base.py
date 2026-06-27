"""
base.py — shared data structures for the Phase 4 agent pipeline.

RunContext flows through every agent.  Each agent reads what it needs
and writes its outputs back onto the same object.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ProposedEdit:
    """One file change proposed by the Editor agent."""
    file_path:        str        # repo-relative path
    repo:             str        # repo name
    description:      str        # human-readable summary of the change
    before:           str        # exact original text to replace
    after:            str        # replacement text
    validated:        bool = False
    validation_error: str  = ""


@dataclass
class RunContext:
    """Mutable state that flows through the entire agent pipeline."""

    # ── Input ─────────────────────────────────────────────────────────────
    task:        str
    mode:        str             # "analyze" | "edit"
    llm_model:   Optional[str]
    top_k:       int
    repo_filter: Optional[str]

    # ── Retriever output ──────────────────────────────────────────────────
    vector_hits:    list = field(default_factory=list)   # list[VectorHit]
    graph_contexts: list = field(default_factory=list)   # list[GraphContext]
    merged_context: str  = ""

    # ── Reasoner output ───────────────────────────────────────────────────
    reasoning:      str       = ""
    summary:        str       = ""
    findings:       List[str] = field(default_factory=list)
    recommendation: str       = ""

    # ── Editor output ─────────────────────────────────────────────────────
    proposed_edits: List[ProposedEdit] = field(default_factory=list)

    # ── Validator output ──────────────────────────────────────────────────
    syntax_issues:     List[str] = field(default_factory=list)
    files_checked:     int       = 0
    validation_passed: bool      = True

    # ── Diagnostics ───────────────────────────────────────────────────────
    agent_times: dict      = field(default_factory=dict)
    errors:      List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str = "base"

    @abstractmethod
    def run(self, ctx: RunContext) -> RunContext:
        """Mutate *ctx* with this agent's outputs and return it."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
