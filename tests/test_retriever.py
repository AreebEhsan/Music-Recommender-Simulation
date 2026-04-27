"""
Tests for src/retriever.py

These tests use a small in-memory song list so they run without touching the
CSV file and without needing any API keys. The sentence-transformer model is
downloaded automatically on first run (~90 MB, cached locally afterward).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import SongRetriever, build_song_document

# ── Fixture ───────────────────────────────────────────────────────────────────

_TEST_SONGS = [
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


# ── build_song_document ───────────────────────────────────────────────────────

def test_build_song_document_returns_string():
    doc = build_song_document(_TEST_SONGS[0])
    assert isinstance(doc, str)
    assert doc.strip() != ""


def test_build_song_document_contains_key_fields():
    doc = build_song_document(_TEST_SONGS[0])
    assert "Sunrise City" in doc
    assert "Neon Echo" in doc
    assert "pop" in doc
    assert "happy" in doc


def test_build_song_document_contains_descriptive_labels():
    # High-energy song should include the word "high"
    doc = build_song_document(_TEST_SONGS[0])   # energy 0.82
    assert "high" in doc

    # Low-energy ambient song should include the word "low"
    doc_low = build_song_document(_TEST_SONGS[4])  # energy 0.28
    assert "low" in doc_low


# ── SongRetriever ─────────────────────────────────────────────────────────────

def test_retriever_builds_index():
    retriever = SongRetriever(_TEST_SONGS)
    assert retriever._index.ntotal == len(_TEST_SONGS)


def test_retrieve_returns_correct_count():
    retriever = SongRetriever(_TEST_SONGS)
    results = retriever.retrieve("upbeat pop workout", top_k=3)
    assert len(results) == 3


def test_retrieve_result_structure():
    retriever = SongRetriever(_TEST_SONGS)
    results = retriever.retrieve("upbeat pop workout", top_k=3)
    for song, score in results:
        assert isinstance(song, dict), "each result must contain a song dict"
        assert "title" in song
        assert "genre" in song
        assert isinstance(score, float), "similarity score must be a float"
        assert 0.0 <= score <= 1.0, f"score {score} out of [0, 1] range"


def test_retrieve_top_k_capped_at_catalog_size():
    retriever = SongRetriever(_TEST_SONGS)
    # Requesting more than catalog size should not raise and should return all songs
    results = retriever.retrieve("music", top_k=100)
    assert len(results) == len(_TEST_SONGS)


def test_retrieve_upbeat_query_favors_high_energy():
    retriever = SongRetriever(_TEST_SONGS)
    results = retriever.retrieve("upbeat energetic workout pop high energy", top_k=2)
    top_song, _ = results[0]
    # The top result should be high-energy (energy >= 0.7)
    assert top_song["energy"] >= 0.7, (
        f"Expected high-energy top result, got '{top_song['title']}' "
        f"with energy={top_song['energy']}"
    )


def test_retrieve_chill_query_favors_low_energy():
    retriever = SongRetriever(_TEST_SONGS)
    results = retriever.retrieve("calm relaxing ambient chill low energy study", top_k=2)
    top_song, _ = results[0]
    # The top result should be lower-energy than the average
    assert top_song["energy"] <= 0.6, (
        f"Expected low-energy top result, got '{top_song['title']}' "
        f"with energy={top_song['energy']}"
    )
