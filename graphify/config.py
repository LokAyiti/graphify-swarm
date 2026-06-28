"""
config.py — manages .graphify.json

The config file lives in the repo root (tracked in git).
It stores repo NAMES and last-known local paths.
Paths are machine-specific; team members update them via `graphify add-repo`.
When cloning fresh, paths can be empty — `graphify rebuild` works without them
because chunk content is preserved in chunks.jsonl.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_FILE = Path(".graphify.json")


@dataclass
class RepoEntry:
    name: str
    path: str       # absolute local path — machine-specific, may be "" on fresh clone


@dataclass
class GraphifyConfig:
    repos:       List[RepoEntry] = field(default_factory=list)
    default_llm: Optional[str]   = None
    ollama_host: str              = "http://localhost:11434"
    qdrant_url:  Optional[str]   = None  # e.g. "http://localhost:6333" for Docker

    @property
    def repo_paths(self) -> Dict[str, Path]:
        """Return {repo_name: Path} for repos that have a valid local path."""
        result: Dict[str, Path] = {}
        for r in self.repos:
            if r.path:
                p = Path(r.path)
                if p.exists():
                    result[r.name] = p
        return result

    def get_repo(self, name: str) -> Optional[RepoEntry]:
        return next((r for r in self.repos if r.name == name), None)


def load_config(path: Path = CONFIG_FILE) -> GraphifyConfig:
    if not path.exists():
        return GraphifyConfig()
    try:
        raw   = json.loads(path.read_text(encoding="utf-8"))
        repos = [RepoEntry(**r) for r in raw.get("repos", [])]
        return GraphifyConfig(
            repos       = repos,
            default_llm = raw.get("default_llm"),
            ollama_host = raw.get("ollama_host", "http://localhost:11434"),
            qdrant_url  = raw.get("qdrant_url"),
        )
    except Exception:
        return GraphifyConfig()


def save_config(cfg: GraphifyConfig, path: Path = CONFIG_FILE) -> None:
    data = {
        "repos":       [asdict(r) for r in cfg.repos],
        "default_llm": cfg.default_llm,
        "ollama_host": cfg.ollama_host,
        "qdrant_url":  cfg.qdrant_url,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_repo(
    cfg:       GraphifyConfig,
    repo_path: Path,
    name:      Optional[str] = None,
) -> RepoEntry:
    """Add a new repo or update the path of an existing one. Returns the entry."""
    repo_name = name or repo_path.name

    for entry in cfg.repos:
        if entry.name == repo_name:
            entry.path = str(repo_path.resolve())
            return entry

    entry = RepoEntry(name=repo_name, path=str(repo_path.resolve()))
    cfg.repos.append(entry)
    return entry
