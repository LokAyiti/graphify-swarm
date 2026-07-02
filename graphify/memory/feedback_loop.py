"""
feedback_loop.py — 3-state feedback with boost/decay and pattern promotion.

Rating states
-------------
  good       → boosts the pattern that produced this answer
  bad        → decays the pattern; marks it for review
  corrected  → same as bad + stores the user's correction as training signal

Promotion rule: a pattern with hit_count >= 5 AND avg_score >= 0.80
AND at least 2 "good" feedback rows for the same repo/keywords cluster
gets promoted to the rules table with trust_score = 0.7.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from graphify.memory.memory_store import (
    _conn,
    _now,
    feedback_stats,
    get_last_query_feedback,
    get_patterns,
    record_feedback,
)


_PROMOTE_MIN_HITS  = 5
_PROMOTE_MIN_SCORE = 0.80
_PROMOTE_MIN_GOOD  = 2


def submit_feedback(
    query:      str,
    repos:      list[str],
    provider:   str,
    model:      str,
    rating:     str,
    correction: Optional[str] = None,
) -> dict:
    """
    Record feedback and immediately run the boost/decay step.

    Returns a summary dict with the feedback id and any promotions triggered.
    """
    fid = record_feedback(
        query=query, repos=repos, provider=provider,
        model=model, rating=rating, correction=correction,
    )

    promotions: list[str] = []

    if rating == "good":
        # Boost: increment hit_count so the pattern ranks higher
        _boost_matching_patterns(repos, query)
        promotions = _maybe_promote(repos)

    elif rating in ("bad", "corrected"):
        # Decay: reduce trust on matching patterns
        _decay_matching_patterns(repos, query)

    return {
        "feedback_id": fid,
        "rating":      rating,
        "promotions":  promotions,
    }


def _boost_matching_patterns(repos: list[str], query: str) -> None:
    """Increase hit_count on patterns whose keywords appear in the query."""
    words = set(query.lower().split())
    with _conn() as con:
        for repo in repos:
            rows = con.execute(
                "SELECT id, keywords, hit_count, avg_score FROM patterns WHERE repo=?",
                (repo,),
            ).fetchall()
            for row in rows:
                kws = set(json.loads(row["keywords"] or "[]"))
                if kws & words:   # at least one keyword overlap
                    new_count = row["hit_count"] + 1
                    con.execute(
                        "UPDATE patterns SET hit_count=?, updated_at=? WHERE id=?",
                        (new_count, _now(), row["id"]),
                    )


def _decay_matching_patterns(repos: list[str], query: str) -> None:
    """Reduce avg_score on patterns whose keywords appear in the query."""
    words = set(query.lower().split())
    with _conn() as con:
        for repo in repos:
            rows = con.execute(
                "SELECT id, keywords, avg_score FROM patterns WHERE repo=?",
                (repo,),
            ).fetchall()
            for row in rows:
                kws = set(json.loads(row["keywords"] or "[]"))
                if kws & words:
                    decayed = max(0.0, round(row["avg_score"] * 0.9, 4))
                    con.execute(
                        "UPDATE patterns SET avg_score=?, updated_at=? WHERE id=?",
                        (decayed, _now(), row["id"]),
                    )


def _maybe_promote(repos: list[str]) -> list[str]:
    """Promote high-confidence patterns to the rules table."""
    promoted: list[str] = []
    with _conn() as con:
        for repo in repos:
            candidates = con.execute(
                """SELECT p.id, p.keywords, p.chunk_types, p.languages, p.hit_count, p.avg_score
                   FROM patterns p
                   WHERE p.repo=?
                     AND p.hit_count >= ?
                     AND p.avg_score >= ?
                     AND NOT EXISTS (
                         SELECT 1 FROM rules r WHERE r.repo=? AND r.trigger LIKE '%' || p.keywords || '%'
                     )""",
                (repo, _PROMOTE_MIN_HITS, _PROMOTE_MIN_SCORE, repo),
            ).fetchall()

            for c in candidates:
                # Check good-feedback count for this repo
                good_count = con.execute(
                    """SELECT COUNT(*) FROM feedback
                       WHERE rating='good'
                         AND repos LIKE ?""",
                    (f"%{repo}%",),
                ).fetchone()[0]

                if good_count >= _PROMOTE_MIN_GOOD:
                    trigger = json.dumps({
                        "keywords":    json.loads(c["keywords"] or "[]"),
                        "min_score":   _PROMOTE_MIN_SCORE,
                        "chunk_types": json.loads(c["chunk_types"] or "[]"),
                    })
                    action = json.dumps({
                        "boost_repos":          [repo],
                        "preferred_chunk_types": json.loads(c["chunk_types"] or "[]"),
                    })
                    desc = f"Auto-promoted from pattern #{c['id']} (hits={c['hit_count']}, score={c['avg_score']})"
                    con.execute(
                        """INSERT INTO rules (repo, description, trigger, action, trust_score, source, created_at)
                           VALUES (?,?,?,?,?,?,?)""",
                        (repo, desc, trigger, action, 0.7, "promoted", _now()),
                    )
                    promoted.append(f"Promoted pattern for repo '{repo}': {json.loads(c['keywords'] or '[]')}")

    return promoted
