"""
Style comparison utility for the Music Recommender.

Runs both explanation styles (standard and music_coach) in fallback mode
(no Gemini API key required) and prints a side-by-side comparison so the
difference in tone and format is immediately visible.

Run with:
    python -m src.style_evaluator
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.explainer import explain_recommendations
from src.logger_config import get_logger
from src.recommender import load_songs, recommend_songs

_logger = get_logger("style_evaluator")
_DATA_PATH = _ROOT / "data" / "songs.csv"

_DEMO_PREFS: Dict = {"genre": "pop", "mood": "happy", "energy": 0.85}
_DEMO_QUERY: str = ""


# ── Core comparison function ──────────────────────────────────────────────────

def compare_explanation_styles(
    recommendations: List[Tuple],
    user_prefs: Dict,
    user_query: str = "",
) -> Tuple[List[str], List[str]]:
    """Return (standard_explanations, coach_explanations) using fallback mode.

    Both styles run without a Gemini API key so the comparison is reproducible
    and works in any environment.
    """
    standard = explain_recommendations(
        user_query=user_query,
        user_prefs=user_prefs,
        recommendations=recommendations,
        use_gemini=False,
        explanation_style="standard",
    )
    coach = explain_recommendations(
        user_query=user_query,
        user_prefs=user_prefs,
        recommendations=recommendations,
        use_gemini=False,
        explanation_style="music_coach",
    )
    return standard, coach


# ── Report printer ────────────────────────────────────────────────────────────

def print_comparison(
    recommendations: List[Tuple],
    user_prefs: Dict,
    user_query: str = "",
) -> None:
    """Print a side-by-side style comparison to stdout."""
    standard, coach = compare_explanation_styles(recommendations, user_prefs, user_query)

    print()
    print("STYLE COMPARISON")
    print("-" * 65)

    for i, (std, cch) in enumerate(zip(standard, coach)):
        song = recommendations[i][0]
        print(f"Song {i + 1}: \"{song['title']}\" by {song['artist']}")
        print(f"  Standard:    {std}")
        print(f"  Music Coach: {cch}")
        print()

    print("-" * 65)
    print("Difference: Music Coach style is shorter and vibe-focused,")
    print("  using casual language vs. the factual rule-based breakdown.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    songs = load_songs(str(_DATA_PATH))
    recommendations = recommend_songs(_DEMO_PREFS, songs, k=3)
    print_comparison(recommendations, _DEMO_PREFS, _DEMO_QUERY)


if __name__ == "__main__":
    main()
