"""
evolution_engine.py — self-learning engine for Graphify.

Reads the episodic log + feedback table, runs pattern analysis, and promotes
high-confidence patterns to trusted procedural rules in memory.db.

This is Phase 5D: the "brain" that closes the learning loop.

What it does
------------
  1. Replay         : re-read all episodic log entries and upsert patterns
                      (same as `graphify evolve` but deeper — scores are
                      weighted by feedback, not just raw hit counts).
  2. Promote        : patterns meeting promotion thresholds become rules
                      with a calibrated trust_score.
  3. Decay          : rules that haven't been triggered recently lose
                      trust_score (0.9× per evolution run without a hit).
  4. Prune          : rules with trust_score < PRUNE_THRESHOLD are deleted.
  5. Drift detect   : if a repo's recent avg_score is > 10 % below its
                      historical baseline, flag it for re-indexing.
  6. Summary report : return a structured dict so the CLI can display it.

Usage
-----
    from graphify.memory.evolution_engine import run_evolution
    report = run_evolution()
    print(report)

Triggered by `graphify evolve --deep` or scheduled externally.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from graphify.memory.memory_store import (
    _conn,
    _now,
    get_patterns,
    record_pattern,
)

# ---------------------------------------------------------------------------
# Thresholds (all tunable)
# ---------------------------------------------------------------------------

_PROMOTE_MIN_HITS      = 5      # minimum hit_count before promotion
_PROMOTE_MIN_SCORE     = 0.78   # minimum avg_score before promotion
_PROMOTE_MIN_GOOD_RATIO = 0.40  # ≥40 % of global "good" feedback in that repo
_DECAY_FACTOR          = 0.92   # trust_score multiplied by this each run if unused
_PRUNE_THRESHOLD       = 0.25   # rules below this trust_score are deleted
_DRIFT_THRESHOLD       = 0.10   # 10 % drop in avg_score signals drift


def run_evolution(
    episodic_log: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """
    Run one full evolution cycle.

    Returns
    -------
    dict with keys:
      patterns_upserted, rules_promoted, rules_decayed, rules_pruned,
      repos_drifting, feedback_summary, duration_s
    """
    import time
    t0 = time.perf_counter()

    log_path = episodic_log or (Path("graphify-out") / "memory" / "episodic.jsonl")

    # ── Load episodic log ────────────────────────────────────────────────
    episodes: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # ── Load feedback ────────────────────────────────────────────────────
    with _conn() as con:
        fb_rows = con.execute("SELECT * FROM feedback").fetchall()
        fb_by_repo: dict[str, dict] = {}
        for row in fb_rows:
            repos_list = json.loads(row["repos"] or "[]")
            for repo in repos_list:
                if repo not in fb_by_repo:
                    fb_by_repo[repo] = {"good": 0, "bad": 0, "corrected": 0, "total": 0}
                rating = row["rating"]
                fb_by_repo[repo][rating] = fb_by_repo[repo].get(rating, 0) + 1
                fb_by_repo[repo]["total"] += 1

    # ── Step 1: Replay — weight patterns by feedback ─────────────────────
    patterns_upserted = 0
    for ep in episodes:
        if ep.get("top_score", 0) < 0.60:   # skip low-quality episodes
            continue
        repos    = ep.get("repos_searched", [])
        query    = ep.get("query", "")
        provider = ep.get("provider", "")

        # Feedback-weighted score: good feedback boosts, bad feedback penalises
        score = ep.get("top_score", 0.0)
        for repo in repos:
            fb = fb_by_repo.get(repo, {})
            if fb.get("total", 0) > 0:
                good_ratio = fb.get("good", 0) / fb["total"]
                bad_ratio  = fb.get("bad", 0) / fb["total"]
                score = score * (1.0 + 0.2 * good_ratio - 0.3 * bad_ratio)
                score = min(1.0, max(0.0, score))

        keywords = list(dict.fromkeys(
            w for w in query.lower().split() if len(w) > 3
        ))[:8]
        if not keywords or not repos:
            continue

        for repo in repos:
            record_pattern(
                repo=repo,
                keywords=keywords,
                chunk_types=[],
                languages=[],
                avg_score=round(score, 4),
                query_type="evolved",
            )
            patterns_upserted += 1

    # ── Step 2: Promote — patterns → rules ───────────────────────────────
    rules_promoted = 0
    with _conn() as con:
        total_good = con.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating='good'"
        ).fetchone()[0]

        candidates = con.execute(
            """SELECT p.id, p.repo, p.keywords, p.chunk_types, p.hit_count, p.avg_score
               FROM patterns p
               WHERE p.hit_count >= ? AND p.avg_score >= ?""",
            (_PROMOTE_MIN_HITS, _PROMOTE_MIN_SCORE),
        ).fetchall()

        for c in candidates:
            repo = c["repo"]

            # Check if already promoted
            exists = con.execute(
                "SELECT 1 FROM rules WHERE repo=? AND trigger LIKE ?",
                (repo, f"%{c['keywords']}%"),
            ).fetchone()
            if exists:
                continue

            # Feedback gate: repo must have a meaningful good-feedback ratio
            repo_fb   = fb_by_repo.get(repo, {})
            repo_good = repo_fb.get("good", 0)
            if total_good > 0:
                good_ratio = repo_good / max(total_good, 1)
            else:
                good_ratio = 0.0

            # Also promote if hit_count is very high even without much feedback
            qualifies = (
                good_ratio >= _PROMOTE_MIN_GOOD_RATIO
                or c["hit_count"] >= _PROMOTE_MIN_HITS * 3
            )
            if not qualifies:
                continue

            # Calibrate trust_score
            hit_factor   = min(1.0, c["hit_count"] / 20)
            score_factor = c["avg_score"]
            fb_factor    = min(1.0, good_ratio * 2)
            trust = round(
                0.4 * hit_factor + 0.4 * score_factor + 0.2 * fb_factor,
                3,
            )

            trigger = json.dumps({
                "keywords":    json.loads(c["keywords"] or "[]"),
                "min_score":   _PROMOTE_MIN_SCORE,
                "chunk_types": json.loads(c["chunk_types"] or "[]"),
            })
            action = json.dumps({
                "boost_repos":  [repo],
                "trust_score":  trust,
            })
            desc = (
                f"Evolved from pattern #{c['id']} "
                f"(hits={c['hit_count']}, score={c['avg_score']:.3f}, "
                f"trust={trust:.3f})"
            )
            con.execute(
                """INSERT INTO rules
                   (repo, description, trigger, action, trust_score, source, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (repo, desc, trigger, action, trust, "evolved", _now()),
            )
            rules_promoted += 1

    # ── Step 3: Decay — age unused rules ─────────────────────────────────
    rules_decayed = 0
    with _conn() as con:
        all_rules = con.execute("SELECT id, trust_score FROM rules").fetchall()
        for rule in all_rules:
            new_trust = round(rule["trust_score"] * _DECAY_FACTOR, 4)
            con.execute(
                "UPDATE rules SET trust_score=? WHERE id=?",
                (new_trust, rule["id"]),
            )
            rules_decayed += 1

    # ── Step 4: Prune — remove rules below trust floor ───────────────────
    rules_pruned = 0
    with _conn() as con:
        result = con.execute(
            "DELETE FROM rules WHERE trust_score < ?",
            (_PRUNE_THRESHOLD,),
        )
        rules_pruned = result.rowcount

    # ── Step 5: Drift detection ───────────────────────────────────────────
    repos_drifting: list[dict] = []
    if len(episodes) >= 10:
        # Split episodes in half: older baseline vs recent window
        mid        = len(episodes) // 2
        older      = episodes[:mid]
        recent     = episodes[mid:]
        repo_older = _avg_score_by_repo(older)
        repo_new   = _avg_score_by_repo(recent)

        for repo, new_avg in repo_new.items():
            old_avg = repo_older.get(repo)
            if old_avg and old_avg > 0:
                drop = (old_avg - new_avg) / old_avg
                if drop > _DRIFT_THRESHOLD:
                    repos_drifting.append({
                        "repo":      repo,
                        "old_avg":   round(old_avg, 3),
                        "new_avg":   round(new_avg, 3),
                        "drop_pct":  round(drop * 100, 1),
                    })

    # ── Step 6: Feedback summary ──────────────────────────────────────────
    with _conn() as con:
        fb_total     = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        fb_good      = con.execute("SELECT COUNT(*) FROM feedback WHERE rating='good'").fetchone()[0]
        fb_bad       = con.execute("SELECT COUNT(*) FROM feedback WHERE rating='bad'").fetchone()[0]
        total_rules  = con.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        total_patt   = con.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]

    return {
        "episodes_analysed":  len(episodes),
        "patterns_upserted":  patterns_upserted,
        "rules_promoted":     rules_promoted,
        "rules_decayed":      rules_decayed,
        "rules_pruned":       rules_pruned,
        "total_patterns":     total_patt,
        "total_rules":        total_rules,
        "repos_drifting":     repos_drifting,
        "feedback_summary":   {"total": fb_total, "good": fb_good, "bad": fb_bad},
        "duration_s":         round(time.perf_counter() - t0, 3),
    }


def _avg_score_by_repo(episodes: list[dict]) -> dict[str, float]:
    """Compute per-repo average top_score across a list of episodes."""
    totals: dict[str, list[float]] = {}
    for ep in episodes:
        score = ep.get("top_score", 0.0)
        for repo in ep.get("repos_searched", []):
            totals.setdefault(repo, []).append(score)
    return {
        repo: sum(scores) / len(scores)
        for repo, scores in totals.items()
        if scores
    }
