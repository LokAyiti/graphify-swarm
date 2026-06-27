"""
validator.py — Agent 4: syntax + regression checks.

Two duties
----------
1. Validate any ProposedEdits from the Editor (JSON/Python syntax,
   ADF required-field regression check).
2. Static syntax scan of every file that appeared in the Retriever's
   top results — surfaces pre-existing issues in the retrieved files.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict

from graphify.agents.base import BaseAgent, ProposedEdit, RunContext


class ValidatorAgent(BaseAgent):
    name = "validator"

    def __init__(self, repo_paths: Dict[str, Path]) -> None:
        """
        repo_paths: mapping from repo name → absolute path of that repo's root.
        Loaded from graphify-out/summaries.json at CLI startup.
        """
        self._repos = repo_paths

    # ── Main entry ────────────────────────────────────────────────────────

    def run(self, ctx: RunContext) -> RunContext:
        all_issues: list[str] = []

        # ① Validate proposed edits
        for edit in ctx.proposed_edits:
            issues = self._validate_edit(edit)
            if issues:
                edit.validation_error = "; ".join(issues)
                all_issues.extend(f"[edit:{edit.file_path}] {i}" for i in issues)
            else:
                edit.validated = True

        # ② Static scan of retrieved files (catches pre-existing bugs)
        seen: set[str] = set()
        for hit in ctx.vector_hits:
            key = f"{hit.repo}:{hit.file_path}"
            if key in seen:
                continue
            seen.add(key)

            root = self._repos.get(hit.repo)
            if not root:
                continue
            full = root / hit.file_path
            if not full.exists():
                continue

            file_issues = _check_syntax(full)
            if file_issues:
                all_issues.extend(f"[{hit.file_path}] {i}" for i in file_issues)

        ctx.files_checked     = len(seen)
        ctx.syntax_issues     = all_issues
        ctx.validation_passed = all(e.validated for e in ctx.proposed_edits)

        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────

    def _validate_edit(self, edit: ProposedEdit) -> list[str]:
        issues: list[str] = []
        ext = Path(edit.file_path).suffix.lower()

        # ① Syntax of the proposed replacement
        if ext in {".json", ".jsonc"}:
            try:
                json.loads(edit.after)
            except json.JSONDecodeError as e:
                issues.append(f"JSON syntax error in proposed 'after': {e}")
        elif ext == ".py":
            try:
                ast.parse(edit.after)
            except SyntaxError as e:
                issues.append(f"Python syntax error in proposed 'after' line {e.lineno}: {e.msg}")

        # ② ADF pipeline regression: required keys must not be removed
        if ext in {".json", ".jsonc"} and edit.before.strip().startswith("{"):
            try:
                before_data = json.loads(edit.before)
                after_data  = json.loads(edit.after)
                if isinstance(before_data, dict) and isinstance(after_data, dict):
                    for key in ("name", "properties"):
                        if key in before_data and key not in after_data:
                            issues.append(f"Regression: required ADF field '{key}' removed")
            except Exception:
                pass

        return issues


# ── File-level syntax checker ─────────────────────────────────────────────────

def _check_syntax(path: Path) -> list[str]:
    ext = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return [f"Cannot read file"]

    if ext in {".json", ".jsonc"}:
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return [f"JSON syntax error at line {e.lineno}: {e.msg}"]

    elif ext == ".py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            return [f"Python syntax error at line {e.lineno}: {e.msg}"]

    return []
