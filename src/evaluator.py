"""
Offline evaluation harness for the Music Recommender.

Runs 4 predefined user profiles through the existing scoring pipeline,
checks that the top result meets genre and confidence expectations, and
prints a pass/fail report.

No Gemini, no RAG, no network calls — works fully offline.

Run with:
    python -m src.evaluator
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.logger_config import get_logger
from src.recommender import load_songs, recommend_songs

_logger = get_logger("evaluator")
_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "songs.csv"


# ── Evaluation case definition ────────────────────────────────────────────────

@dataclass
class EvalCase:
    name: str
    user_prefs: dict          # {genre, mood, energy}
    expected_genre: str       # genre the top result must belong to
    min_confidence: float     # minimum confidence % the top result must achieve


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    top_title: str
    top_genre: str
    confidence: float
    failure_reason: str       # empty string when passed


# ── Test suite ────────────────────────────────────────────────────────────────

EVAL_CASES: List[EvalCase] = [
    EvalCase(
        name="upbeat workout",
        user_prefs={"genre": "pop", "mood": "intense", "energy": 0.9},
        expected_genre="pop",
        min_confidence=75.0,
    ),
    EvalCase(
        name="chill study",
        user_prefs={"genre": "lofi", "mood": "chill", "energy": 0.35},
        expected_genre="lofi",
        min_confidence=80.0,
    ),
    EvalCase(
        name="intense rock",
        user_prefs={"genre": "rock", "mood": "intense", "energy": 0.9},
        expected_genre="rock",
        min_confidence=80.0,
    ),
    EvalCase(
        name="relaxed acoustic",
        user_prefs={"genre": "jazz", "mood": "relaxed", "energy": 0.35},
        expected_genre="jazz",
        min_confidence=80.0,
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(songs: list) -> List[EvalResult]:
    """Run every EvalCase against the given song catalog.

    Parameters
    ----------
    songs : list of song dicts as returned by load_songs()

    Returns
    -------
    List[EvalResult] — one result per eval case, in the same order as EVAL_CASES.
    """
    results: List[EvalResult] = []

    for case in EVAL_CASES:
        _logger.debug(f"Evaluating case: {case.name!r}")

        recs = recommend_songs(case.user_prefs, songs, k=1)

        if not recs:
            results.append(EvalResult(
                case=case,
                passed=False,
                top_title="N/A",
                top_genre="N/A",
                confidence=0.0,
                failure_reason="recommend_songs() returned no results",
            ))
            continue

        top_song, _score, confidence_pct, _explanation = recs[0]

        genre_ok      = top_song["genre"] == case.expected_genre
        confidence_ok = confidence_pct >= case.min_confidence
        passed        = genre_ok and confidence_ok

        reasons: List[str] = []
        if not genre_ok:
            reasons.append(
                f"genre '{top_song['genre']}' ≠ expected '{case.expected_genre}'"
            )
        if not confidence_ok:
            reasons.append(
                f"confidence {confidence_pct}% < threshold {case.min_confidence}%"
            )

        results.append(EvalResult(
            case=case,
            passed=passed,
            top_title=top_song["title"],
            top_genre=top_song["genre"],
            confidence=confidence_pct,
            failure_reason="; ".join(reasons),
        ))

    return results


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(results: List[EvalResult]) -> None:
    """Print a formatted pass/fail summary to stdout."""
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    avg    = sum(r.confidence for r in results) / total if total else 0.0

    print()
    print("EVALUATION REPORT")
    print("-" * 55)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"[{status}] {r.case.name:<20} -> "
            f"{r.top_title} ({r.confidence}%)"
        )
        if not r.passed:
            print(f"       ! {r.failure_reason}")

    print("-" * 55)
    print(f"Result:             {passed}/{total} cases passed")
    print(f"Average confidence: {avg:.1f}%")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    """Load catalog, run evaluation, print report. Returns exit code."""
    _logger.info(f"Loading catalog from {_DATA_PATH}")
    songs = load_songs(str(_DATA_PATH))
    _logger.info(f"Loaded {len(songs)} songs — running {len(EVAL_CASES)} eval cases")

    results = run_evaluation(songs)
    print_report(results)

    all_passed = all(r.passed for r in results)
    if not all_passed:
        _logger.warning("One or more evaluation cases FAILED.")
    else:
        _logger.info("All evaluation cases passed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
