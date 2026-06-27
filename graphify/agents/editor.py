"""
editor.py — Agent 3: generate file edits from Reasoner output.

Only runs in "edit" mode AND when an LLM is configured.
Parses the model's output into ProposedEdit objects (file, before, after).
"""
from __future__ import annotations

import re
from typing import Optional

from graphify.agents.base import BaseAgent, ProposedEdit, RunContext

_SYSTEM_PROMPT = """\
You are a code-editor agent. Based on the task and analysis, propose precise file edits.

For EACH file that needs changing output EXACTLY this block (repeat for multiple files):

FILE: <repo-relative file path>
DESCRIPTION: <one-line description of the change>
BEFORE:
```
<exact original text to replace — must match the file verbatim>
```
AFTER:
```
<exact replacement text>
```
---

Rules:
- Only propose changes you are confident about.
- BEFORE blocks must be verbatim substrings of the real file content.
- If no changes are needed, output the single word: NO_EDITS_NEEDED"""


class EditorAgent(BaseAgent):
    name = "editor"

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, ctx: RunContext) -> RunContext:
        if ctx.mode != "edit":
            return ctx   # silently skip in analyze mode

        if self._llm is None:
            ctx.errors.append(
                "Editor: no LLM configured — pass --llm <model> to enable edits"
            )
            return ctx

        if not ctx.reasoning:
            ctx.errors.append("Editor: Reasoner produced no output to work from")
            return ctx

        # Build a compact prompt from the analysis + top file content
        file_list = "\n".join(
            f"- {h.file_path} (score: {h.score:.3f})"
            for h in ctx.vector_hits[:5]
        )
        prompt = (
            f"Task: {ctx.task}\n\n"
            f"Analysis from Reasoner:\n{ctx.reasoning[:2_000]}\n\n"
            f"Top relevant files:\n{file_list}\n\n"
            f"Context (excerpt):\n{ctx.merged_context[:2_000]}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self._llm.ask(messages, timeout=180)
            if "NO_EDITS_NEEDED" in raw.upper():
                return ctx
            ctx.proposed_edits = _parse_edits(raw, ctx)
        except Exception as exc:
            ctx.errors.append(f"Editor LLM error: {exc}")

        return ctx


# ── Edit parser ───────────────────────────────────────────────────────────────

def _parse_edits(raw: str, ctx: RunContext) -> list[ProposedEdit]:
    edits: list[ProposedEdit] = []
    blocks = re.split(r"^-{3,}$", raw, flags=re.MULTILINE)

    for block in blocks:
        file_m = re.search(r"^FILE:\s*(.+)$",        block, re.MULTILINE)
        desc_m = re.search(r"^DESCRIPTION:\s*(.+)$", block, re.MULTILINE)

        # Extract fenced code blocks (BEFORE then AFTER)
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", block, re.DOTALL)

        if not file_m or len(code_blocks) < 2:
            continue

        file_path   = file_m.group(1).strip()
        description = desc_m.group(1).strip() if desc_m else "No description"
        before      = code_blocks[0].strip()
        after       = code_blocks[1].strip()

        # Find which repo owns this file
        repo = ""
        for hit in ctx.vector_hits:
            if hit.file_path == file_path or file_path.endswith(hit.file_path):
                repo = hit.repo
                break

        edits.append(ProposedEdit(
            file_path   = file_path,
            repo        = repo,
            description = description,
            before      = before,
            after       = after,
        ))

    return edits
