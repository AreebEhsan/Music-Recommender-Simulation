"""
Tests for src/agent.py

All tests run without a Gemini API key (use_gemini=False).
RAG tests that use SongRetriever require sentence-transformers, which is
already a project dependency — the model is downloaded once and cached.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import run_recommendation_agent

# ── Shared test catalog ───────────────────────────────────────────────────────

_SONGS = [
    {
        "id": 1, "title": "Sunrise City", "artist": "Neon Echo",
        "genre": "pop", "mood": "happy", "energy": 0.82,
        "tempo_bpm": 118, "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
    },
    {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "chill", "energy": 0.42,
        "tempo_bpm": 78, "valence": 0.56, "danceability": 0.62, "acousticness": 0.71,
    },
    {
        "id": 3, "title": "Storm Runner", "artist": "Voltline",
        "genre": "rock", "mood": "intense", "energy": 0.91,
        "tempo_bpm": 152, "valence": 0.48, "danceability": 0.66, "acousticness": 0.10,
    },
    {
        "id": 5, "title": "Gym Hero", "artist": "Max Pulse",
        "genre": "pop", "mood": "intense", "energy": 0.93,
        "tempo_bpm": 132, "valence": 0.77, "danceability": 0.88, "acousticness": 0.05,
    },
    {
        "id": 6, "title": "Spacewalk Thoughts", "artist": "Orbit Bloom",
        "genre": "ambient", "mood": "chill", "energy": 0.28,
        "tempo_bpm": 60, "valence": 0.65, "danceability": 0.41, "acousticness": 0.92,
    },
]

_USER_PREFS = {"genre": "pop", "mood": "happy", "energy": 0.8}


# ── Step structure ────────────────────────────────────────────────────────────

def test_agent_returns_exactly_five_steps():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    assert len(result["steps"]) == 5


def test_agent_step_names_are_correct_and_ordered():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    names = [s["step"] for s in result["steps"]]
    assert names == ["PLAN", "RETRIEVE", "SCORE", "EXPLAIN", "REFLECT"]


def test_each_step_has_message_string():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    for step in result["steps"]:
        assert isinstance(step["message"], str)
        assert step["message"].strip() != ""


# ── Recommendations ───────────────────────────────────────────────────────────

def test_agent_returns_recommendations():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    assert len(result["recommendations"]) > 0


def test_agent_respects_top_k():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=2, use_gemini=False
    )
    assert len(result["recommendations"]) <= 2


def test_explanations_match_recommendation_count():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    assert len(result["explanations"]) == len(result["recommendations"])


# ── Query / RAG behaviour ─────────────────────────────────────────────────────

def test_empty_query_plan_mentions_no_rag():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    plan_msg = result["steps"][0]["message"].lower()
    assert "no natural language query" in plan_msg or "scoring all" in plan_msg


def test_empty_query_produces_no_rag_candidates():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    assert result["rag_candidates"] == []
    assert result["sim_lookup"] == {}


def test_rag_query_plan_mentions_rag():
    result = run_recommendation_agent(
        "upbeat pop workout", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    plan_msg = result["steps"][0]["message"].lower()
    assert "rag" in plan_msg


def test_rag_query_populates_sim_lookup():
    result = run_recommendation_agent(
        "upbeat pop workout", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    assert len(result["sim_lookup"]) > 0
    for song_id, pct in result["sim_lookup"].items():
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0


# ── Warnings ──────────────────────────────────────────────────────────────────

def test_low_confidence_triggers_warning():
    # Single song that completely mismatches the requested profile
    mismatch_songs = [
        {
            "id": 99, "title": "Obscure Track", "artist": "Nobody",
            "genre": "classical", "mood": "relaxed", "energy": 0.10,
            "tempo_bpm": 55, "valence": 0.30, "danceability": 0.20, "acousticness": 0.95,
        }
    ]
    prefs = {"genre": "hip-hop", "mood": "intense", "energy": 0.95}
    result = run_recommendation_agent(
        "", prefs, mismatch_songs, top_k=1, use_gemini=False
    )
    # Confidence will be ~3.8% — well below the 50% threshold
    assert len(result["warnings"]) > 0
    assert any("confidence" in w.lower() for w in result["warnings"])


def test_no_warnings_for_strong_match():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=1, use_gemini=False
    )
    # pop/happy/0.8 against a catalog with matching pop/happy songs → high confidence
    low_conf_warnings = [
        w for w in result["warnings"]
        if "confidence" in w.lower() and "low" in w.lower()
    ]
    assert len(low_conf_warnings) == 0


# ── use_gemini=False compatibility ────────────────────────────────────────────

def test_agent_works_with_gemini_disabled():
    result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=3, use_gemini=False
    )
    # Should complete without exception and return explanations
    assert isinstance(result["explanations"], list)
    assert all(isinstance(e, str) for e in result["explanations"])


def test_agent_music_coach_style_differs_from_standard():
    standard_result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=2, use_gemini=False,
        explanation_style="standard",
    )
    coach_result = run_recommendation_agent(
        "", _USER_PREFS, _SONGS, top_k=2, use_gemini=False,
        explanation_style="music_coach",
    )
    assert standard_result["explanations"] != coach_result["explanations"]
