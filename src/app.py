import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve correctly
# regardless of where `streamlit run src/app.py` is invoked from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.guardrails import validate_profile
from src.logger_config import get_logger
from src.recommender import load_songs, recommend_songs

_DATA_PATH = str(_ROOT / "data" / "songs.csv")
_logger = get_logger("app")

_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz",
    "synthwave", "indie pop", "hip-hop", "classical", "electronic",
]
_MOODS = ["happy", "chill", "intense", "relaxed", "moody", "focused"]

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Music Recommender", page_icon="🎵", layout="centered")
st.title("AI Music Recommender System")
st.write(
    "Choose your preferences below. "
    "Optionally describe what you want in plain English to enable **RAG retrieval**."
)
st.divider()

# ── Structured inputs ─────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    genre = st.selectbox("Preferred Genre", _GENRES)
    mood  = st.selectbox("Preferred Mood",  _MOODS)

with col_right:
    energy = st.slider("Target Energy", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    top_k  = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# ── Natural language query (RAG) ──────────────────────────────────────────────
nl_query = st.text_input(
    "Natural Language Query *(optional — enables RAG retrieval)*",
    placeholder="e.g. something upbeat and energetic for a morning workout",
)

st.divider()

# ── Action ────────────────────────────────────────────────────────────────────
if st.button("Generate Recommendations", type="primary", use_container_width=True):

    trace: list[str] = []

    # ── Step 1: Guardrails ────────────────────────────────────────────────────
    raw_prefs = {"genre": genre, "mood": mood, "energy": energy}
    try:
        user_prefs = validate_profile(raw_prefs)
        _logger.info(f"Input validated: {user_prefs}")
        trace.append(
            f"✅ Input validated — "
            f"genre={user_prefs['genre']}, mood={user_prefs['mood']}, "
            f"energy={user_prefs['energy']:.2f}"
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()

    # ── Step 2: Load catalog ──────────────────────────────────────────────────
    songs = load_songs(_DATA_PATH)
    _logger.info(f"Loaded {len(songs)} songs from catalog")
    trace.append(f"✅ Songs loaded — {len(songs)} tracks in catalog")

    # ── Step 3: RAG retrieval (if query provided) or full catalog ─────────────
    # sim_lookup maps song id → similarity % for display alongside results.
    sim_lookup: dict[int, float] = {}

    if nl_query.strip():
        from src.retriever import SongRetriever

        query_clean = nl_query.strip()
        _logger.info(f"RAG query received: '{query_clean}'")
        trace.append(f"🔍 Natural language query received: \"{query_clean}\"")

        # Always retrieve a pool larger than top_k so re-ranking has choices.
        retrieve_k = min(len(songs), max(top_k + 3, 8))

        with st.spinner("Building retrieval index…"):
            retriever = SongRetriever(songs)
            rag_results = retriever.retrieve(query_clean, top_k=retrieve_k)

        candidates = [s for s, _ in rag_results]
        sim_lookup  = {s["id"]: round(sim * 100, 1) for s, sim in rag_results}

        _logger.info(f"RAG: retrieved {len(candidates)} candidates")
        trace.append(
            f"🔍 RAG: {len(candidates)} candidates retrieved from FAISS "
            f"(pool size={retrieve_k})"
        )

    else:
        candidates = songs
        trace.append("ℹ️ No natural language query — full catalog used as candidates")

    # ── Step 4: Score and rank ────────────────────────────────────────────────
    recommendations = recommend_songs(user_prefs, candidates, k=top_k)
    _logger.info(
        f"Scored {len(candidates)} candidates → "
        f"returning top {len(recommendations)}"
    )
    trace.append(
        f"✅ Candidates re-ranked by scoring engine — "
        f"top {len(recommendations)} returned"
    )

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("Top Recommendations")

    if not recommendations:
        st.warning("No recommendations found. Try adjusting your preferences.")
    else:
        rag_active = bool(nl_query.strip())

        for rank, (song, score, confidence_pct, explanation) in enumerate(
            recommendations, start=1
        ):
            with st.container():
                st.markdown(
                    f"**#{rank} — {song['title']}** &nbsp;·&nbsp; *{song['artist']}*"
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Genre",      song["genre"])
                c2.metric("Mood",       song["mood"])
                c3.metric("Energy",     f"{song['energy']:.2f}")
                c4.metric("Confidence", f"{confidence_pct}%")

                sim_str = (
                    f" | Similarity: {sim_lookup[song['id']]}%"
                    if rag_active and song["id"] in sim_lookup
                    else ""
                )
                st.caption(f"Score: {score:.2f} / 4.00{sim_str} — {explanation}")
                st.divider()

    # ── System Trace ──────────────────────────────────────────────────────────
    with st.expander("System Trace"):
        for event in trace:
            st.write(event)
