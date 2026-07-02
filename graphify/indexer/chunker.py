"""
chunker.py — splits source files into logical chunks for embedding.

Strategy per file type
----------------------
.py       → Python built-in ast: top-level functions & classes as chunks
.md/.mdx  → split on markdown headings (# / ## / ### …)
everything else → sliding window of WINDOW_LINES with OVERLAP_LINES overlap
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Sliding-window settings
WINDOW_LINES = 60
OVERLAP_LINES = 12

# Hard cap on chunk size fed to the embedder (chars)
MAX_CHUNK_CHARS = 4_096


@dataclass
class Chunk:
    """One indexable unit extracted from a source file."""

    repo: str           # name of the root repo directory
    file_path: str      # path relative to repo root (POSIX slash)
    language: str       # e.g. "python", "typescript", "markdown"
    chunk_type: str     # "function" | "class" | "section" | "window"
    name: str           # symbol / heading name, or "" for windows
    content: str        # raw text of the chunk (trimmed to MAX_CHUNK_CHARS)
    start_line: int     # 1-based
    end_line: int       # 1-based, inclusive
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Python chunker — uses built-in ast (no third-party dep)
# ---------------------------------------------------------------------------

def _chunk_python(path: Path, repo_root: Path, repo_name: str) -> List[Chunk]:
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    rel = path.relative_to(repo_root).as_posix()
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _chunk_by_window(path, repo_root, repo_name, language="python")

    chunks: List[Chunk] = []

    def extract(node: ast.AST, chunk_type: str) -> Chunk:
        start = node.lineno - 1  # type: ignore[attr-defined]
        end = node.end_lineno    # type: ignore[attr-defined]
        content = "\n".join(lines[start:end])
        return Chunk(
            repo=repo_name,
            file_path=rel,
            language="python",
            chunk_type=chunk_type,
            name=node.name,  # type: ignore[attr-defined]
            content=content[:MAX_CHUNK_CHARS],
            start_line=start + 1,
            end_line=end,
        )

    # Collect top-level and class-level nodes only
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            chunks.append(extract(node, "function"))
        elif isinstance(node, ast.ClassDef):
            chunks.append(extract(node, "class"))

    if not chunks:
        # File has no functions/classes — treat as single window
        return _chunk_by_window(path, repo_root, repo_name, language="python")

    return chunks


# ---------------------------------------------------------------------------
# Markdown chunker — splits on headings
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")


def _chunk_markdown(path: Path, repo_root: Path, repo_name: str) -> List[Chunk]:
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    rel = path.relative_to(repo_root).as_posix()
    lines = source.splitlines()

    # Collect heading positions
    boundaries: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            boundaries.append((i, m.group(1).strip()))

    if not boundaries:
        return _chunk_by_window(path, repo_root, repo_name, language="markdown")

    boundaries.append((len(lines), "__END__"))  # sentinel
    chunks: List[Chunk] = []

    for idx in range(len(boundaries) - 1):
        start, heading = boundaries[idx]
        end = boundaries[idx + 1][0]
        content = "\n".join(lines[start:end]).strip()
        if not content:
            continue
        chunks.append(Chunk(
            repo=repo_name,
            file_path=rel,
            language="markdown",
            chunk_type="section",
            name=heading,
            content=content[:MAX_CHUNK_CHARS],
            start_line=start + 1,
            end_line=end,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Generic sliding-window chunker (used as fallback for all other languages)
# ---------------------------------------------------------------------------

def _chunk_by_window(
    path: Path,
    repo_root: Path,
    repo_name: str,
    language: str = "",
) -> List[Chunk]:
    rel = path.relative_to(repo_root).as_posix()
    if not language:
        language = path.suffix.lstrip(".").lower() or "text"

    # utf-8-sig strips the BOM that tools like PowerShell/ADF add to JSON/CSV files
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = source.splitlines()
    chunks: List[Chunk] = []

    step = WINDOW_LINES - OVERLAP_LINES
    i = 0
    while i < len(lines):
        window = lines[i: i + WINDOW_LINES]
        content = "\n".join(window).strip()
        if content:
            chunks.append(Chunk(
                repo=repo_name,
                file_path=rel,
                language=language,
                chunk_type="window",
                name="",
                content=content[:MAX_CHUNK_CHARS],
                start_line=i + 1,
                end_line=min(i + WINDOW_LINES, len(lines)),
            ))
        i = max(i + step, i + 1)  # guard against infinite loop on step=0

    return chunks


# ---------------------------------------------------------------------------
# Extension → language name table
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".ps1": "powershell", ".psm1": "powershell",
    ".md": "markdown", ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".json": "json", ".jsonc": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".cfg": "ini", ".ini": "ini",
    ".html": "html",
    ".css": "css", ".scss": "css",
    ".vue": "vue",
    ".svelte": "svelte",
    ".sql": "sql",
    ".tf": "terraform", ".hcl": "hcl",
    ".bicep": "bicep",
    ".env": "env",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_file(path: Path, repo_root: Path, repo_name: str) -> List[Chunk]:
    """Return a list of Chunk objects for *path*.

    Dispatches to the best strategy based on file extension.
    Always returns at least one chunk (empty files return an empty list).
    """
    ext = path.suffix.lower()
    lang = _EXT_TO_LANG.get(ext, ext.lstrip(".") or "text")

    if ext == ".py":
        return _chunk_python(path, repo_root, repo_name)
    if ext in {".md", ".mdx"}:
        return _chunk_markdown(path, repo_root, repo_name)
    return _chunk_by_window(path, repo_root, repo_name, language=lang)
