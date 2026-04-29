import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.agent import run_recommendation_agent
from src.guardrails import validate_profile
from src.logger_config import get_logger
from src.recommender import load_songs
from src.retriever import preview_index_documents

_DATA_PATH = str(_ROOT / "data" / "songs.csv")
_logger = get_logger("app")

_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz",
    "synthwave", "indie pop", "hip-hop", "classical", "electronic",
]
_MOODS  = ["happy", "chill", "intense", "relaxed", "moody", "focused"]
_STYLE_MAP = {"Standard": "standard", "Music Coach": "music_coach"}

# Step-label colours for the agent trace
_STEP_ICONS = {
    "PLAN":     "🗺️",
    "RETRIEVE": "🔍",
    "SCORE":    "📊",
    "EXPLAIN":  "💬",
    "REFLECT":  "🪞",
}

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("System Settings")
    st.caption("AI Music Recommender v2.0 · CodePath AI110 Project 4")
    st.divider()

    style_label = st.selectbox(
        "Explanation Style",
        list(_STYLE_MAP.keys()),
        help=(
            "**Standard** — factual, attribute-by-attribute breakdown.\n\n"
            "**Music Coach** — short, casual, vibe-focused sentence."
        ),
    )
    explanation_style = _STYLE_MAP[style_label]

    top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    use_gemini = st.checkbox(
        "Use Gemini Explanations",
        value=True,
        help="Requires GEMINI_API_KEY in .env. Falls back to rule-based text if unavailable.",
    )

    st.divider()
    st.markdown("**About this system**")
    st.caption(
        "Hybrid recommender combining rule-based scoring, "
        "FAISS semantic retrieval, and Gemini-powered explanations. "
        "Runs 23 automated tests and a 4-case offline evaluator."
    )
    st.caption("Run evaluator: `python -m src.evaluator`")
    st.caption("Run style compare: `python -m src.style_evaluator`")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("AI Music Recommender System")
st.write(
    "Describe what you want in plain English (optional) to enable **RAG retrieval**, "
    "or use the structured controls below. Change **Explanation Style** in the sidebar."
)
st.divider()

# Natural language query (full width — primary new input)
nl_query = st.text_input(
    "Natural Language Query *(optional — enables RAG retrieval)*",
    placeholder="e.g. something upbeat and energetic for a morning workout",
)

# Structured preferences
col_left, col_right = st.columns(2)
with col_left:
    genre = st.selectbox("Preferred Genre", _GENRES)
    mood  = st.selectbox("Preferred Mood",  _MOODS)
with col_right:
    energy = st.slider(
        "Target Energy", min_value=0.0, max_value=1.0, value=0.7, step=0.01,
        help="0 = very calm, 1 = very high energy",
    )

st.divider()

# ── Generate button ───────────────────────────────────────────────────────────
if st.button("Generate Recommendations", type="primary", use_container_width=True):

    # Validate inputs before handing off to agent
    raw_prefs = {"genre": genre, "mood": mood, "energy": energy}
    try:
        user_prefs = validate_profile(raw_prefs)
        _logger.info(f"Input validated: {user_prefs}")
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()

    # Load catalog
    songs = load_songs(_DATA_PATH)
    _logger.info(f"Loaded {len(songs)} songs")

    # Run agent
    with st.spinner("Running recommendation agent…"):
        result = run_recommendation_agent(
            user_query=nl_query.strip(),
            user_prefs=user_prefs,
            songs=songs,
            top_k=top_k,
            use_gemini=use_gemini,
            explanation_style=explanation_style,
        )

    recommendations  = result["recommendations"]
    explanations     = result["explanations"]
    sim_lookup       = result["sim_lookup"]
    rag_candidates   = result["rag_candidates"]
    agent_steps      = result["steps"]
    warnings         = result["warnings"]
    rag_active       = bool(nl_query.strip()) and bool(rag_candidates)

    # Surface any warnings prominently
    for w in warnings:
        st.warning(w)

    # ── Recommendation cards ──────────────────────────────────────────────────
    st.subheader("Top Recommendations")

    if not recommendations:
        st.info("No recommendations found. Try adjusting genre, mood, or energy.")
    else:
        for rank, (song, score, confidence_pct, rule_explanation) in enumerate(
            recommendations, start=1
        ):
            ai_text  = explanations[rank - 1] if rank - 1 < len(explanations) else ""
            sim_pct  = sim_lookup.get(song["id"])

            with st.container():
                st.markdown(
                    f"**#{rank} — {song['title']}** &nbsp;·&nbsp; *{song['artist']}*"
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Genre",       song["genre"])
                c2.metric("Mood",        song["mood"])
                c3.metric("Energy",      f"{song['energy']:.2f}")
                c4.metric("Confidence",  f"{confidence_pct}%")
                c5.metric("Similarity",  f"{sim_pct}%" if sim_pct is not None else "N/A")

                if ai_text:
                    st.write(ai_text)

                with st.expander("Scoring breakdown"):
                    sim_str = (
                        f" | Similarity: {sim_pct}%"
                        if sim_pct is not None else ""
                    )
                    st.caption(
                        f"Score: {score:.2f} / 4.00{sim_str} — {rule_explanation}"
                    )

                st.divider()

    # ── Agent Trace ───────────────────────────────────────────────────────────
    with st.expander("Agent Trace"):
        for step in agent_steps:
            icon = _STEP_ICONS.get(step["step"], "•")
            st.markdown(f"**{icon} {step['step']}** — {step['message']}")

    # ── RAG Retrieved Candidates ──────────────────────────────────────────────
    if rag_active:
        with st.expander(f"RAG Retrieved Candidates ({len(rag_candidates)} songs)"):
            st.caption(
                "These are the songs FAISS retrieved before scoring. "
                "The scoring engine re-ranked them to produce the final list."
            )
            for cand_song, cand_sim in rag_candidates:
                st.write(
                    f"**{cand_song['title']}** by {cand_song['artist']} "
                    f"— {cand_song['genre']} / {cand_song['mood']} "
                    f"| Similarity: {round(cand_sim * 100, 1)}%"
                )

    # ── Indexed Document Example ──────────────────────────────────────────────
    with st.expander("View Indexed Song Document Example"):
        st.caption(
            "This is how each song is represented as text before embedding. "
            "Descriptive labels (e.g. 'high energy', 'very danceable') help "
            "match vague natural language queries to the right songs."
        )
        sample_docs = preview_index_documents(songs, limit=1)
        if sample_docs:
            st.code(sample_docs[0], language=None)
