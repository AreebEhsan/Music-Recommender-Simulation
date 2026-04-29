"""
Gemini explanation layer for the Music Recommender.

Two explanation styles are supported:
  "standard"     — factual 1-2 sentence breakdown grounded in scoring data
  "music_coach"  — short, conversational, vibe-focused; talks like a friend

If GEMINI_API_KEY is absent or use_gemini=False, a deterministic fallback is
used for each style so the system always returns something useful.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from src.logger_config import get_logger

load_dotenv()

_logger = get_logger("explainer")
_GEMINI_MODEL = "gemini-1.5-flash"

# ── Style instructions ────────────────────────────────────────────────────────

_STANDARD_INSTRUCTION = (
    "You are a music recommendation assistant. "
    "Write a short, conversational 1-2 sentence explanation for each recommended song "
    "that tells the listener *why* it fits their preferences.\n\n"
    "Rules you must follow:\n"
    "- Base every sentence ONLY on the metadata provided below "
    "(genre, mood, energy, tempo, valence, danceability, acousticness, "
    "confidence score, and scoring breakdown).\n"
    "- Do NOT mention Spotify, streaming counts, chart positions, album names, "
    "lyrics, or any fact not present in the data below.\n"
    "- Be specific: name the attributes that make each song a good fit.\n"
    "- Return EXACTLY a numbered list — one entry per song, nothing else."
)

_MUSIC_COACH_INSTRUCTION = (
    "You are a friendly music coach giving casual advice to a listener. "
    "For each recommended song, write EXACTLY ONE punchy sentence that captures "
    "why this song matches their vibe — like a friend recommending music. "
    "Use casual, encouraging language. Focus on how the song feels "
    "(energetic, chill, intense, etc.) and connect it to what the listener asked for. "
    "Still base your explanation ONLY on the metadata provided — "
    "do NOT invent Spotify stats, chart positions, albums, or lyrics. "
    "Return EXACTLY a numbered list — one entry per song, nothing else."
)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    user_query: str,
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, float, str]],
    explanation_style: str = "standard",
) -> str:
    """Compose the Gemini prompt for the requested style."""

    instruction = (
        _MUSIC_COACH_INSTRUCTION
        if explanation_style == "music_coach"
        else _STANDARD_INSTRUCTION
    )

    query_line = (
        f'User\'s natural language query: "{user_query}"'
        if user_query.strip()
        else "No natural language query provided — use structured preferences only."
    )

    song_lines = []
    for i, (song, score, confidence_pct, rule_explanation) in enumerate(
        recommendations, start=1
    ):
        song_lines.append(
            f"{i}. \"{song['title']}\" by {song['artist']}\n"
            f"   Genre: {song['genre']} | Mood: {song['mood']} | "
            f"Energy: {song['energy']:.2f} | Tempo: {int(song['tempo_bpm'])} BPM\n"
            f"   Valence: {song['valence']:.2f} | Danceability: {song['danceability']:.2f} | "
            f"Acousticness: {song['acousticness']:.2f}\n"
            f"   Match confidence: {confidence_pct}%\n"
            f"   Scoring breakdown: {rule_explanation}"
        )

    songs_block = "\n\n".join(song_lines)

    return (
        f"{instruction}\n\n"
        f"{query_line}\n"
        f"User preferences: genre={user_prefs['genre']}, "
        f"mood={user_prefs['mood']}, "
        f"energy={float(user_prefs['energy']):.2f}\n\n"
        f"Recommended songs:\n\n{songs_block}\n\n"
        f"Respond with exactly {len(recommendations)} numbered explanations."
    )


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_numbered_response(text: str, expected: int) -> List[str]:
    """Split Gemini's numbered list into individual explanation strings."""
    parts = re.split(r"(?m)^\s*\d+\.\s+", text.strip())
    parts = [p.strip().replace("\n", " ") for p in parts if p.strip()]

    if len(parts) == expected:
        return parts

    _logger.warning(
        f"Numbered parse yielded {len(parts)} parts, expected {expected}. "
        "Trying blank-line split."
    )
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if len(paragraphs) == expected:
        return paragraphs

    _logger.warning("Could not reliably parse Gemini response; using partial results.")
    result = parts[:expected] if len(parts) >= expected else parts
    while len(result) < expected:
        result.append("")
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def explain_recommendations(
    user_query: str,
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, float, str]],
    use_gemini: bool = True,
    explanation_style: str = "standard",
) -> List[str]:
    """Return one explanation string per recommendation.

    Parameters
    ----------
    user_query        : Free-text query the user typed (may be empty).
    user_prefs        : Cleaned preference dict {genre, mood, energy}.
    recommendations   : Output of recommend_songs() —
                        list of (song, score, confidence_pct, rule_explanation).
    use_gemini        : When False the fallback is used regardless of key presence.
    explanation_style : "standard" (default) or "music_coach".

    Returns
    -------
    List[str] of length == len(recommendations).
    Always returns something — Gemini failure never propagates to the caller.
    """
    if not recommendations:
        return []

    # ── Fallback builder — deterministic, no API needed ───────────────────────
    def _fallback() -> List[str]:
        if explanation_style == "music_coach":
            result = []
            for _, _, confidence_pct, rule_expl in recommendations:
                first_clause = rule_expl.rstrip(".").split(";")[0].strip()
                result.append(
                    f"Vibe check ({confidence_pct}% match) — {first_clause}."
                )
            return result
        # standard: return rule-based explanation unchanged
        return [rule_expl for _, _, _, rule_expl in recommendations]

    # ── Decide whether to attempt Gemini ─────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not use_gemini:
        _logger.info(
            f"Gemini disabled by caller — using {explanation_style} fallback."
        )
        return _fallback()

    if not api_key:
        _logger.warning(
            "GEMINI_API_KEY not set — using fallback explanations. "
            "Add your key to a .env file or set it as an environment variable."
        )
        return _fallback()

    # ── Gemini call ───────────────────────────────────────────────────────────
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_GEMINI_MODEL)
        prompt = _build_prompt(
            user_query, user_prefs, recommendations, explanation_style
        )

        _logger.debug(
            f"Sending {explanation_style!r} prompt to Gemini "
            f"({len(recommendations)} songs)."
        )
        response = model.generate_content(prompt)
        raw_text = response.text

        explanations = _parse_numbered_response(raw_text, len(recommendations))
        _logger.info(
            f"Gemini ({explanation_style}) explanations received — "
            f"{len(explanations)} strings parsed."
        )
        return explanations

    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            f"Gemini call failed ({type(exc).__name__}: {exc}). "
            f"Using {explanation_style} fallback."
        )
        return _fallback()
