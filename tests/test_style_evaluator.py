"""
Tests for src/style_evaluator.py and the explanation_style parameter of explainer.py

All tests run offline with use_gemini=False — no API key required.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainer import explain_recommendations
from src.style_evaluator import compare_explanation_styles

# ── Shared fixtures ───────────────────────────────────────────────────────────

_USER_PREFS = {"genre": "pop", "mood": "happy", "energy": 0.85}

_RECS = [
    (
        {
            "id": 1, "title": "Sunrise City", "artist": "Neon Echo",
            "genre": "pop", "mood": "happy", "energy": 0.82,
            "tempo_bpm": 118, "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
        },
        3.97, 99.2,
        "Your preferred genre (pop) matches exactly; mood (happy) aligns; "
        "energy (0.82) is nearly identical to your target (0.85).",
    ),
    (
        {
            "id": 5, "title": "Gym Hero", "artist": "Max Pulse",
            "genre": "pop", "mood": "intense", "energy": 0.93,
            "tempo_bpm": 132, "valence": 0.77, "danceability": 0.88, "acousticness": 0.05,
        },
        2.92, 73.0,
        "Your preferred genre (pop) matches exactly; mood is intense "
        "(you prefer happy); energy (0.93) is close to your target (0.85).",
    ),
]


# ── Style output shape ────────────────────────────────────────────────────────

def test_both_styles_return_correct_count():
    standard, coach = compare_explanation_styles(_RECS, _USER_PREFS)
    assert len(standard) == len(_RECS)
    assert len(coach)    == len(_RECS)


def test_both_styles_return_non_empty_strings():
    standard, coach = compare_explanation_styles(_RECS, _USER_PREFS)
    for s, c in zip(standard, coach):
        assert isinstance(s, str) and s.strip() != ""
        assert isinstance(c, str) and c.strip() != ""


# ── Style differentiation ─────────────────────────────────────────────────────

def test_styles_produce_different_output():
    standard, coach = compare_explanation_styles(_RECS, _USER_PREFS)
    assert standard != coach, (
        "Standard and Music Coach fallbacks should produce distinct output."
    )


def test_standard_fallback_equals_rule_explanation():
    standard, _ = compare_explanation_styles(_RECS, _USER_PREFS)
    for i, (_, _, _, rule_expl) in enumerate(_RECS):
        assert standard[i] == rule_expl


def test_music_coach_fallback_contains_vibe_language():
    _, coach = compare_explanation_styles(_RECS, _USER_PREFS)
    for c in coach:
        lowered = c.lower()
        assert "vibe" in lowered or "match" in lowered, (
            f"Music Coach explanation should contain vibe language. Got: {c!r}"
        )


def test_music_coach_fallback_is_shorter_than_standard():
    standard, coach = compare_explanation_styles(_RECS, _USER_PREFS)
    avg_standard = sum(len(s) for s in standard) / len(standard)
    avg_coach    = sum(len(c) for c in coach)    / len(coach)
    assert avg_coach < avg_standard, (
        "Music Coach style should produce shorter explanations than Standard."
    )


# ── Direct explain_recommendations style param ────────────────────────────────

def test_explain_standard_unchanged_from_baseline():
    result = explain_recommendations(
        user_query="",
        user_prefs=_USER_PREFS,
        recommendations=_RECS,
        use_gemini=False,
        explanation_style="standard",
    )
    for i, (_, _, _, rule_expl) in enumerate(_RECS):
        assert result[i] == rule_expl


def test_explain_music_coach_differs_from_standard():
    standard = explain_recommendations(
        user_query="", user_prefs=_USER_PREFS,
        recommendations=_RECS, use_gemini=False, explanation_style="standard",
    )
    coach = explain_recommendations(
        user_query="", user_prefs=_USER_PREFS,
        recommendations=_RECS, use_gemini=False, explanation_style="music_coach",
    )
    assert standard != coach


def test_unknown_style_falls_back_to_standard():
    result = explain_recommendations(
        user_query="",
        user_prefs=_USER_PREFS,
        recommendations=_RECS,
        use_gemini=False,
        explanation_style="nonexistent_style",
    )
    # Unrecognised style should behave like standard (rule_expl unchanged)
    for i, (_, _, _, rule_expl) in enumerate(_RECS):
        assert result[i] == rule_expl
