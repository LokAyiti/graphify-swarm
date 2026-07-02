"""
memory_store.py — SQLite-backed persistent memory for Graphify.

Tables
------
  patterns   : reusable retrieval patterns learned from episodic logs
  feedback   : explicit user feedback on answers (good / bad / corrected)
  rules      : promoted patterns with boost scores + scoping

Schema intentionally simple — one file, zero migrations needed at this stage.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional


_DB_PATH = Path("graphify-out") / "memory" / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo        TEXT NOT NULL,
    query_type  TEXT,          -- keyword / semantic / structural
    keywords    TEXT,          -- JSON array of trigger words
    chunk_types TEXT,          -- JSON array: ["function","section",...]
    languages   TEXT,          -- JSON array: ["python","json",...]
    hit_count   INTEGER DEFAULT 1,
    avg_score   REAL    DEFAULT 0.0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    query        TEXT    NOT NULL,
    repos        TEXT,          -- JSON array
    provider     TEXT,
    model        TEXT,
    rating       TEXT    NOT NULL,  -- "good" | "bad" | "corrected"
    correction   TEXT,              -- user's rewrite if rating == "corrected"
    promoted     INTEGER DEFAULT 0  -- 1 = used to update a pattern/rule
);

CREATE TABLE IF NOT EXISTS rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo        TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger     TEXT NOT NULL,  -- JSON: {keywords, min_score, chunk_types}
    action      TEXT NOT NULL,  -- JSON: {boost_repos, penalise_chunk_types}
    trust_score REAL DEFAULT 0.5,
    source      TEXT,           -- "promoted" | "manual"
    created_at  TEXT NOT NULL
);
"""


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Feedback API
# ---------------------------------------------------------------------------

def record_feedback(
    query:      str,
    repos:      list[str],
    provider:   str,
    model:      str,
    rating:     str,            # "good" | "bad" | "corrected"
    correction: Optional[str] = None,
) -> int:
    """Insert a feedback row and return its id."""
    if rating not in ("good", "bad", "corrected"):
        raise ValueError(f"rating must be 'good', 'bad', or 'corrected', got '{rating}'")
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO feedback (ts, query, repos, provider, model, rating, correction)
               VALUES (?,?,?,?,?,?,?)""",
            (_now(), query, json.dumps(repos), provider, model, rating, correction),
        )
        return cur.lastrowid


def get_last_query_feedback() -> Optional[dict]:
    """Return the most recent feedback row, or None."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def feedback_stats() -> dict:
    """Aggregate feedback counts."""
    with _conn() as con:
        rows = con.execute(
            "SELECT rating, COUNT(*) as cnt FROM feedback GROUP BY rating"
        ).fetchall()
        total = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    counts = {r["rating"]: r["cnt"] for r in rows}
    return {
        "total":     total,
        "good":      counts.get("good", 0),
        "bad":       counts.get("bad", 0),
        "corrected": counts.get("corrected", 0),
    }


# ---------------------------------------------------------------------------
# Pattern API
# ---------------------------------------------------------------------------

def record_pattern(
    repo:        str,
    keywords:    list[str],
    chunk_types: list[str],
    languages:   list[str],
    avg_score:   float,
    query_type:  str = "semantic",
) -> None:
    """Upsert a pattern (merge by repo + keyword fingerprint)."""
    fingerprint = json.dumps(sorted(keywords))
    with _conn() as con:
        existing = con.execute(
            "SELECT id, hit_count, avg_score FROM patterns WHERE repo=? AND keywords=?",
            (repo, fingerprint),
        ).fetchone()
        if existing:
            new_count = existing["hit_count"] + 1
            new_avg   = (existing["avg_score"] * existing["hit_count"] + avg_score) / new_count
            con.execute(
                "UPDATE patterns SET hit_count=?, avg_score=?, updated_at=? WHERE id=?",
                (new_count, round(new_avg, 4), _now(), existing["id"]),
            )
        else:
            con.execute(
                """INSERT INTO patterns
                   (repo, query_type, keywords, chunk_types, languages, avg_score, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    repo, query_type, fingerprint,
                    json.dumps(chunk_types), json.dumps(languages),
                    round(avg_score, 4), _now(), _now(),
                ),
            )


def get_patterns(repo: Optional[str] = None, min_hits: int = 2) -> list[dict]:
    """Return patterns sorted by hit_count desc."""
    with _conn() as con:
        if repo:
            rows = con.execute(
                "SELECT * FROM patterns WHERE repo=? AND hit_count>=? ORDER BY hit_count DESC",
                (repo, min_hits),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM patterns WHERE hit_count>=? ORDER BY hit_count DESC",
                (min_hits,),
            ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["keywords"]    = json.loads(d["keywords"] or "[]")
        d["chunk_types"] = json.loads(d["chunk_types"] or "[]")
        d["languages"]   = json.loads(d["languages"] or "[]")
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Memory health summary
# ---------------------------------------------------------------------------

def memory_summary() -> dict:
    """Return a dict describing the current state of memory."""
    with _conn() as con:
        pattern_count = con.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        rule_count    = con.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        fb_total      = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        good          = con.execute("SELECT COUNT(*) FROM feedback WHERE rating='good'").fetchone()[0]
        bad           = con.execute("SELECT COUNT(*) FROM feedback WHERE rating='bad'").fetchone()[0]
    return {
        "patterns":      pattern_count,
        "rules":         rule_count,
        "feedback_total": fb_total,
        "feedback_good":  good,
        "feedback_bad":   bad,
        "db_path":        str(_DB_PATH),
    }
