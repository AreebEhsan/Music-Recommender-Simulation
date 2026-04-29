"""
Tests for src/explainer.py

All tests run without a GEMINI_API_KEY by using use_gemini=False or by
ensuring the environment variable is absent. Gemini is never called here.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainer import explain_recommendations, _build_prompt, _parse_numbered_response


# ── Shared fixtures ───────────────────────────────────────────────────────────

_USER_PREFS = {"genre": "pop", "mood": "happy", "energy": 0.85}

_RECOMMENDATIONS = [
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
        "Your preferred genre (pop) matches exactly; mood is intense (you prefer happy); "
        "energy (0.93) is close to your target (0.85).",
    ),
    (
        {
            "id": 10, "title": "Rooftop Lights", "artist": "Indigo Parade",
            "genre": "indie pop", "mood": "happy", "energy": 0.76,
            "tempo_bpm": 124, "valence": 0.81, "danceability": 0.82, "acousticness": 0.35,
        },
        1.91, 47.8,
        "Genre is indie pop (you prefer pop); mood (happy) aligns; "
        "energy (0.76) is close to your target (0.85).",
    ),
]


# ── explain_recommendations ───────────────────────────────────────────────────

def test_fallback_returns_correct_count():
    result = explain_recommendations(
        user_query="upbeat pop",
        user_prefs=_USER_PREFS,
        recommendations=_RECOMMENDATIONS,
        use_gemini=False,
    )
    assert len(result) == len(_RECOMMENDATIONS)


def test_fallback_returns_strings():
    result = explain_recommendations(
        user_query="",
        user_prefs=_USER_PREFS,
        recommendations=_RECOMMENDATIONS,
        use_gemini=False,
    )
    for item in result:
        assert isinstance(item, str)
        assert item.strip() != ""


def test_fallback_content_matches_rule_explanation():
    result = explain_recommendations(
        user_query="",
        user_prefs=_USER_PREFS,
        recommendations=_RECOMMENDATIONS,
        use_gemini=False,
    )
    # Fallback must echo the rule-based explanation from the recommendation tuple
    for i, (_, _, _, rule_expl) in enumerate(_RECOMMENDATIONS):
        assert result[i] == rule_expl, (
            f"Fallback explanation at index {i} does not match rule-based explanation."
        )


def test_empty_recommendations_returns_empty_list():
    result = explain_recommendations(
        user_query="anything",
        user_prefs=_USER_PREFS,
        recommendations=[],
        use_gemini=False,
    )
    assert result == []


def test_no_api_key_falls_back_automatically(monkeypatch):
    # Remove key from environment so the function must fall back
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = explain_recommendations(
        user_query="morning run",
        user_prefs=_USER_PREFS,
        recommendations=_RECOMMENDATIONS,
        use_gemini=True,   # True but key is absent → should fall back
    )
    assert len(result) == len(_RECOMMENDATIONS)
    for item in result:
        assert isinstance(item, str)
        assert item.strip() != ""


def test_single_recommendation_fallback():
    single = [_RECOMMENDATIONS[0]]
    result = explain_recommendations(
        user_query="",
        user_prefs=_USER_PREFS,
        recommendations=single,
        use_gemini=False,
    )
    assert len(result) == 1
    assert result[0] == single[0][3]  # tuple index 3 is the rule explanation


# ── _build_prompt ─────────────────────────────────────────────────────────────

def test_build_prompt_contains_song_titles():
    prompt = _build_prompt("upbeat pop", _USER_PREFS, _RECOMMENDATIONS)
    assert "Sunrise City" in prompt
    assert "Gym Hero" in prompt
    assert "Rooftop Lights" in prompt


def test_build_prompt_contains_user_prefs():
    prompt = _build_prompt("", _USER_PREFS, _RECOMMENDATIONS)
    assert "pop" in prompt
    assert "happy" in prompt
    assert "0.85" in prompt


def test_build_prompt_contains_anti_hallucination_rules():
    prompt = _build_prompt("", _USER_PREFS, _RECOMMENDATIONS)
    # Prompt must explicitly forbid inventing facts
    assert "Spotify" in prompt or "NOT" in prompt or "not" in prompt


# ── _parse_numbered_response ──────────────────────────────────────────────────

def test_parse_clean_numbered_list():
    text = "1. First explanation here.\n2. Second explanation here.\n3. Third explanation."
    result = _parse_numbered_response(text, expected=3)
    assert len(result) == 3
    assert result[0] == "First explanation here."
    assert result[1] == "Second explanation here."
    assert result[2] == "Third explanation."


def test_parse_pads_short_response():
    # Gemini returns only 2 items but we expect 3
    text = "1. Only one.\n2. Only two."
    result = _parse_numbered_response(text, expected=3)
    assert len(result) == 3


def test_parse_trims_extra_items():
    # Gemini returns 4 items but we only expect 3
    text = "1. A.\n2. B.\n3. C.\n4. D."
    result = _parse_numbered_response(text, expected=3)
    assert len(result) == 3
