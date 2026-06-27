"""
walker.py — discovers indexable source files across repo paths.

Skips common noise directories (.git, node_modules, __pycache__, etc.)
and respects .gitignore patterns via pathspec.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pathspec

# Directories that are never worth indexing
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "target", "bin", "obj",
    ".idea", ".vs", "graphify-out", ".terraform", "vendor",
    "eggs", ".eggs", "site-packages", "htmlcov", "wheels",
})

# File extensions we can chunk and embed
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    # Source code
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".cs", ".cpp", ".c", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    # Shell / config
    ".sh", ".bash", ".zsh", ".ps1", ".psm1",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env",
    # Docs / data
    ".md", ".mdx", ".rst", ".txt",
    ".json", ".jsonc",
    # Web
    ".html", ".css", ".scss", ".vue", ".svelte",
    # SQL / infra
    ".sql", ".tf", ".bicep", ".hcl",
})

# Hard cap: skip binary-likely files above this size (bytes)
MAX_FILE_BYTES = 512 * 1024  # 512 KB


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    gi = root / ".gitignore"
    if gi.exists():
        try:
            return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(errors="replace").splitlines())
        except Exception:
            return None
    return None


def walk_repo(repo_path: str | Path) -> Generator[Path, None, None]:
    """Yield all indexable file paths under *repo_path*.

    Files are yielded as absolute resolved Path objects.
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repo path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    gitignore = _load_gitignore(root)

    for dirpath_str, dirnames, filenames in os.walk(root):
        current = Path(dirpath_str)

        # Prune skip dirs in-place so os.walk skips them entirely
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        )

        for filename in filenames:
            filepath = current / filename

            # Extension filter
            if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Size guard — skip huge generated files
            try:
                if filepath.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            # .gitignore filter
            if gitignore:
                try:
                    rel = filepath.relative_to(root).as_posix()
                    if gitignore.match_file(rel):
                        continue
                except ValueError:
                    pass

            yield filepath
