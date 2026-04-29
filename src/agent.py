"""
Lightweight observable recommendation agent.

run_recommendation_agent() wraps the full pipeline in five labelled steps
(PLAN → RETRIEVE → SCORE → EXPLAIN → REFLECT) and returns a structured dict
that Streamlit can render as a trace, making the system's reasoning visible.

Design principles:
- Each step is deterministic and produces a human-readable message.
- No step raises an exception to the caller; failures degrade gracefully.
- The agent adds no new logic beyond what the existing modules already do.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from src.explainer import explain_recommendations
from src.logger_config import get_logger
from src.recommender import recommend_songs

_logger = get_logger("agent")

_LOW_CONFIDENCE_THRESHOLD = 50.0  # % below which we flag weak recommendations


def run_recommendation_agent(
    user_query: str,
    user_prefs: Dict,
    songs: List[Dict],
    top_k: int = 5,
    use_gemini: bool = True,
    explanation_style: str = "standard",
) -> Dict:
    """Run the full recommendation pipeline with visible step-by-step tracing.

    Parameters
    ----------
    user_query        : Free-text query (may be empty — disables RAG).
    user_prefs        : Cleaned preference dict {genre, mood, energy}.
    songs             : Full song catalog as returned by load_songs().
    top_k             : Number of final recommendations to return.
    use_gemini        : Forward to explain_recommendations().
    explanation_style : "standard" or "music_coach".

    Returns
    -------
    {
      "steps":         list of {"step": str, "message": str},
      "recommendations": list of (song, score, confidence_pct, rule_explanation),
      "explanations":  list of str,
      "rag_candidates": list of (song_dict, similarity_float),  # empty if no RAG
      "sim_lookup":    dict[song_id -> similarity_pct],          # empty if no RAG
      "warnings":      list of str,
    }
    """
    steps: List[Dict[str, str]] = []
    warnings: List[str] = []
    rag_candidates: List[Tuple[Dict, float]] = []
    sim_lookup: Dict[int, float] = {}

    # ── STEP 1: PLAN ─────────────────────────────────────────────────────────
    use_rag = bool(user_query.strip())
    retrieve_k = min(len(songs), max(top_k + 3, 8))

    if use_rag:
        plan_msg = (
            f"RAG enabled: will retrieve {retrieve_k} candidates from the FAISS "
            f"semantic index, then re-rank by scoring engine."
        )
    else:
        plan_msg = (
            f"No natural language query provided: scoring all {len(songs)} songs "
            "directly with the rule-based engine."
        )

    steps.append({"step": "PLAN", "message": plan_msg})
    _logger.info(f"Agent PLAN: {plan_msg}")

    # ── STEP 2: RETRIEVE ─────────────────────────────────────────────────────
    if use_rag:
        try:
            from src.retriever import SongRetriever

            retriever = SongRetriever(songs)
            rag_candidates = retriever.retrieve(user_query.strip(), top_k=retrieve_k)
            candidates = [s for s, _ in rag_candidates]
            sim_lookup = {s["id"]: round(sim * 100, 1) for s, sim in rag_candidates}
            best_sim = max(sim_lookup.values()) if sim_lookup else 0.0
            retrieve_msg = (
                f"Retrieved {len(candidates)} candidates via FAISS "
                f"(best similarity: {best_sim:.1f}%)."
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"RAG retrieval failed: {exc}. Falling back to full catalog.")
            candidates = songs
            retrieve_msg = (
                f"RAG retrieval failed ({type(exc).__name__}) — "
                "using full catalog as fallback."
            )
            warnings.append("RAG retrieval encountered an error; full catalog was used.")
    else:
        candidates = songs
        retrieve_msg = f"Using full catalog: {len(candidates)} songs as candidates."

    steps.append({"step": "RETRIEVE", "message": retrieve_msg})
    _logger.info(f"Agent RETRIEVE: {retrieve_msg}")

    # ── STEP 3: SCORE ─────────────────────────────────────────────────────────
    recommendations = recommend_songs(user_prefs, candidates, k=top_k)

    if recommendations:
        top_song, _score, top_conf, _ = recommendations[0]
        score_msg = (
            f"Scored {len(candidates)} candidates. "
            f"Top result: \"{top_song['title']}\" ({top_conf}% confidence). "
            f"Returning {len(recommendations)} recommendation(s)."
        )
    else:
        score_msg = "Scoring complete — no results found for these preferences."

    steps.append({"step": "SCORE", "message": score_msg})
    _logger.info(f"Agent SCORE: {score_msg}")

    # ── STEP 4: EXPLAIN ───────────────────────────────────────────────────────
    explanations = explain_recommendations(
        user_query=user_query,
        user_prefs=user_prefs,
        recommendations=recommendations,
        use_gemini=use_gemini,
        explanation_style=explanation_style,
    )

    api_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if use_gemini and api_key_present:
        explain_msg = (
            f"Gemini ({explanation_style} style) generated "
            f"{len(explanations)} explanation(s)."
        )
    elif use_gemini and not api_key_present:
        explain_msg = (
            f"GEMINI_API_KEY not set — {explanation_style} fallback "
            "explanation used."
        )
    else:
        explain_msg = (
            f"Gemini disabled — {explanation_style} fallback "
            "explanation used."
        )

    steps.append({"step": "EXPLAIN", "message": explain_msg})
    _logger.info(f"Agent EXPLAIN: {explain_msg}")

    # ── STEP 5: REFLECT ───────────────────────────────────────────────────────
    if not recommendations:
        reflect_msg = (
            "No results to evaluate. "
            "Recommendation quality: unavailable."
        )
        warnings.append(
            "No recommendations were found. "
            "Try selecting a different genre or mood."
        )
    else:
        top_conf = recommendations[0][2]
        if top_conf < _LOW_CONFIDENCE_THRESHOLD:
            reflect_msg = (
                f"Top confidence is {top_conf}% — below {_LOW_CONFIDENCE_THRESHOLD}%. "
                "Recommendations may not closely match preferences."
            )
            warnings.append(
                f"Low confidence alert: top result is only {top_conf}% match. "
                "Consider adjusting genre, mood, or energy for better results."
            )
        else:
            reflect_msg = (
                f"Top confidence is {top_conf}% — results look strong. "
                f"All {len(recommendations)} recommendation(s) scored above "
                f"{_LOW_CONFIDENCE_THRESHOLD}%."
                if all(r[2] >= _LOW_CONFIDENCE_THRESHOLD for r in recommendations)
                else f"Top confidence is {top_conf}% — strong lead result. "
                "Some lower-ranked results have weaker matches."
            )

    steps.append({"step": "REFLECT", "message": reflect_msg})
    _logger.info(f"Agent REFLECT: {reflect_msg}")

    return {
        "steps": steps,
        "recommendations": recommendations,
        "explanations": explanations,
        "rag_candidates": rag_candidates,
        "sim_lookup": sim_lookup,
        "warnings": warnings,
    }
