"""
extractor.py — extract graph nodes and edges from source files.

Node types
----------
  repo      — repository root
  file      — source file
  function  — function or method
  class     — class definition
  import    — imported module / package
  section   — markdown heading section

Edge types
----------
  contains  — repo→file, file→function, file→class, file→section
  imports   — file→import
  inherits  — class→class
  calls     — function→function (intra-file, Python only)
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class GNode:
    """One node in the code knowledge graph."""
    id: str
    type: str        # repo | file | function | class | import | section
    name: str
    repo: str
    file_path: str
    language: str
    start_line: int = 0
    end_line: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class GEdge:
    """One directed edge in the code knowledge graph."""
    source: str
    target: str
    type: str        # contains | imports | inherits | calls
    metadata: dict = field(default_factory=dict)


ExtractionResult = Tuple[List[GNode], List[GEdge]]


# ── ID helpers ─────────────────────────────────────────────────────────────────

def _fid(repo: str, rel: str) -> str:
    return f"file:{repo}:{rel}"


def _func_id(repo: str, rel: str, name: str, line: int) -> str:
    return f"func:{repo}:{rel}:{name}:{line}"


def _class_id(repo: str, rel: str, name: str, line: int) -> str:
    return f"class:{repo}:{rel}:{name}:{line}"


def _import_id(repo: str, module: str) -> str:
    return f"import:{repo}:{module}"


def _section_id(repo: str, rel: str, line: int) -> str:
    return f"section:{repo}:{rel}:{line}"


# ── Python extractor ──────────────────────────────────────────────────────────

class _CallVisitor(ast.NodeVisitor):
    """Collect calls inside a single function, not descending into nested funcs."""

    def __init__(self, target: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._target = target
        self.calls: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if node is self._target:
            self.generic_visit(node)
        # else: don't descend into nested functions

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


def _extract_python(path: Path, repo_root: Path, repo_name: str) -> ExtractionResult:
    source = path.read_text(encoding="utf-8", errors="replace")
    rel    = path.relative_to(repo_root).as_posix()
    fid    = _fid(repo_name, rel)

    nodes: list[GNode] = [GNode(
        id=fid, type="file", name=path.name,
        repo=repo_name, file_path=rel, language="python",
    )]
    edges: list[GEdge] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return nodes, edges

    # name → node-id map for symbol resolution within this file
    sym: dict[str, str] = {}

    # Pass 1 — collect functions and classes
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            nid = _func_id(repo_name, rel, node.name, node.lineno)
            sym[node.name] = nid
            nodes.append(GNode(
                id=nid, type="function", name=node.name,
                repo=repo_name, file_path=rel, language="python",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", 0),
            ))
            edges.append(GEdge(source=fid, target=nid, type="contains"))

        elif isinstance(node, ast.ClassDef):
            nid = _class_id(repo_name, rel, node.name, node.lineno)
            sym[node.name] = nid
            nodes.append(GNode(
                id=nid, type="class", name=node.name,
                repo=repo_name, file_path=rel, language="python",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", 0),
            ))
            edges.append(GEdge(source=fid, target=nid, type="contains"))

    # Pass 2 — imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                iid = _import_id(repo_name, alias.name)
                nodes.append(GNode(
                    id=iid, type="import", name=alias.name,
                    repo=repo_name, file_path=rel, language="python",
                ))
                edges.append(GEdge(source=fid, target=iid, type="imports"))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or "__unknown__"
            iid = _import_id(repo_name, module)
            nodes.append(GNode(
                id=iid, type="import", name=module,
                repo=repo_name, file_path=rel, language="python",
            ))
            edges.append(GEdge(source=fid, target=iid, type="imports"))

    # Pass 3 — class inheritance (intra-file only)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cid = _class_id(repo_name, rel, node.name, node.lineno)
            for base in node.bases:
                base_name: str | None = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name and base_name in sym:
                    edges.append(GEdge(source=cid, target=sym[base_name], type="inherits"))

    # Pass 4 — intra-file calls (uses scoped visitor to avoid nested-function confusion)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            caller_id = _func_id(repo_name, rel, node.name, node.lineno)
            visitor = _CallVisitor(node)
            visitor.visit(node)
            for callee_name in visitor.calls:
                if callee_name in sym and sym[callee_name] != caller_id:
                    edges.append(GEdge(
                        source=caller_id,
                        target=sym[callee_name],
                        type="calls",
                    ))

    return nodes, edges


# ── JavaScript / TypeScript extractor ─────────────────────────────────────────

_JS_IMPORT   = re.compile(r"""(?:import\s+.*?\s+from|from)\s+['"]([^'"]+)['"]""", re.MULTILINE)
_JS_REQUIRE  = re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE)
_JS_FUNCTION = re.compile(r"""(?:export\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*\(""", re.MULTILINE)
_JS_ARROW    = re.compile(r"""(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(""", re.MULTILINE)
_JS_CLASS    = re.compile(r"""(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?""", re.MULTILINE)


def _line_of(source: str, pos: int) -> int:
    return source[:pos].count("\n") + 1


def _extract_js(path: Path, repo_root: Path, repo_name: str) -> ExtractionResult:
    source = path.read_text(encoding="utf-8", errors="replace")
    rel    = path.relative_to(repo_root).as_posix()
    lang   = "typescript" if path.suffix in {".ts", ".tsx"} else "javascript"
    fid    = _fid(repo_name, rel)

    nodes: list[GNode] = [GNode(
        id=fid, type="file", name=path.name,
        repo=repo_name, file_path=rel, language=lang,
    )]
    edges: list[GEdge] = []

    # Imports
    for m in [*_JS_IMPORT.finditer(source), *_JS_REQUIRE.finditer(source)]:
        module = m.group(1)
        iid = _import_id(repo_name, module)
        nodes.append(GNode(id=iid, type="import", name=module,
                           repo=repo_name, file_path=rel, language=lang))
        edges.append(GEdge(source=fid, target=iid, type="imports"))

    # Functions (named function declarations)
    for m in _JS_FUNCTION.finditer(source):
        name = m.group(1)
        line = _line_of(source, m.start())
        nid  = _func_id(repo_name, rel, name, line)
        nodes.append(GNode(id=nid, type="function", name=name,
                           repo=repo_name, file_path=rel, language=lang,
                           start_line=line))
        edges.append(GEdge(source=fid, target=nid, type="contains"))

    # Classes (and inheritance)
    class_names: dict[str, str] = {}
    for m in _JS_CLASS.finditer(source):
        name, parent = m.group(1), m.group(2)
        line = _line_of(source, m.start())
        nid  = _class_id(repo_name, rel, name, line)
        class_names[name] = nid
        nodes.append(GNode(id=nid, type="class", name=name,
                           repo=repo_name, file_path=rel, language=lang,
                           start_line=line))
        edges.append(GEdge(source=fid, target=nid, type="contains"))
        if parent and parent in class_names:
            edges.append(GEdge(source=nid, target=class_names[parent], type="inherits"))

    return nodes, edges


# ── Markdown extractor ────────────────────────────────────────────────────────

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _extract_markdown(path: Path, repo_root: Path, repo_name: str) -> ExtractionResult:
    source = path.read_text(encoding="utf-8", errors="replace")
    rel    = path.relative_to(repo_root).as_posix()
    fid    = _fid(repo_name, rel)

    nodes: list[GNode] = [GNode(
        id=fid, type="file", name=path.name,
        repo=repo_name, file_path=rel, language="markdown",
    )]
    edges: list[GEdge] = []

    for m in _MD_HEADING.finditer(source):
        heading = m.group(2).strip()
        line    = _line_of(source, m.start())
        sid     = _section_id(repo_name, rel, line)
        nodes.append(GNode(
            id=sid, type="section", name=heading,
            repo=repo_name, file_path=rel, language="markdown",
            start_line=line,
        ))
        edges.append(GEdge(source=fid, target=sid, type="contains"))

    return nodes, edges


# ── JSON extractor ────────────────────────────────────────────────────────────

def _extract_json(path: Path, repo_root: Path, repo_name: str) -> ExtractionResult:
    """Extract a file node with ADF-pipeline-aware metadata from JSON files."""
    # Use utf-8-sig to strip the UTF-8 BOM that PowerShell/ADF tooling often adds
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    rel    = path.relative_to(repo_root).as_posix()
    fid    = _fid(repo_name, rel)

    metadata: dict = {}
    try:
        data = json.loads(source)
        if isinstance(data, dict):
            # Azure Data Factory pipeline detection
            props = data.get("properties", {})
            if isinstance(props, dict) and "activities" in props:
                acts = props["activities"]
                metadata["pipeline_name"]    = data.get("name", path.stem)
                metadata["activity_count"]   = len(acts)
                metadata["activity_names"]   = [a.get("name", "") for a in acts[:20]]
                metadata["pipeline_type"]    = data.get("type", "")
                metadata["folder"]           = props.get("folder", {}).get("name", "")
            else:
                metadata["top_keys"] = list(data.keys())[:10]
    except (json.JSONDecodeError, Exception):
        pass

    return [GNode(
        id=fid, type="file", name=path.name,
        repo=repo_name, file_path=rel, language="json",
        metadata=metadata,
    )], []


# ── Dispatcher ────────────────────────────────────────────────────────────────

_EXT_HANDLER = {
    ".py":   _extract_python,
    ".js":   _extract_js,  ".jsx": _extract_js,
    ".ts":   _extract_js,  ".tsx": _extract_js,
    ".md":   _extract_markdown,  ".mdx": _extract_markdown,
    ".json": _extract_json, ".jsonc": _extract_json,
}


def extract_file(path: Path, repo_root: Path, repo_name: str) -> ExtractionResult:
    """Return (nodes, edges) for a single source file.

    Falls back to a bare file node on parse errors or unknown extensions.
    """
    handler = _EXT_HANDLER.get(path.suffix.lower())
    try:
        if handler:
            return handler(path, repo_root, repo_name)
    except Exception:
        pass

    # Generic fallback — just a file node
    rel  = path.relative_to(repo_root).as_posix()
    lang = path.suffix.lstrip(".").lower() or "text"
    return [GNode(
        id=_fid(repo_name, rel), type="file", name=path.name,
        repo=repo_name, file_path=rel, language=lang,
    )], []
