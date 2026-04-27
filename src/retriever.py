"""
RAG retrieval layer for the Music Recommender.

Each song is converted into a natural language document, embedded with
sentence-transformers (all-MiniLM-L6-v2), and stored in a FAISS index.
At query time the user's free-text description is embedded with the same
model and the nearest-neighbour songs are returned as retrieval candidates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.logger_config import get_logger

_logger = get_logger("retriever")
_MODEL_NAME = "all-MiniLM-L6-v2"


# ── Document builder ──────────────────────────────────────────────────────────

def build_song_document(song: Dict) -> str:
    """Convert a song dict into a descriptive natural language string for embedding.

    Using words like 'high energy', 'very danceable', 'acoustic' alongside the
    raw numbers helps the embedding model match vague user queries such as
    'something upbeat for the gym' to the right songs.
    """

    def _energy_label(v: float) -> str:
        return "high" if v >= 0.7 else ("moderate" if v >= 0.4 else "low")

    def _valence_label(v: float) -> str:
        if v >= 0.7:
            return "very positive"
        if v >= 0.5:
            return "positive"
        if v >= 0.3:
            return "neutral"
        return "melancholic"

    def _dance_label(v: float) -> str:
        return "very danceable" if v >= 0.7 else ("moderately danceable" if v >= 0.4 else "not very danceable")

    def _acoustic_label(v: float) -> str:
        return "acoustic" if v >= 0.7 else ("mixed" if v >= 0.3 else "electronic")

    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. "
        f"Mood: {song['mood']}. "
        f"Energy: {song['energy']:.2f} ({_energy_label(song['energy'])}). "
        f"Tempo: {int(song['tempo_bpm'])} BPM. "
        f"Valence: {song['valence']:.2f} ({_valence_label(song['valence'])}). "
        f"Danceability: {song['danceability']:.2f} ({_dance_label(song['danceability'])}). "
        f"Acousticness: {song['acousticness']:.2f} ({_acoustic_label(song['acousticness'])})."
    )


# ── Retriever ─────────────────────────────────────────────────────────────────

class SongRetriever:
    """FAISS-backed semantic retriever for song dicts.

    Usage::
        retriever = SongRetriever(songs)          # build index once
        results   = retriever.retrieve(query, 8)  # list of (song, score)
    """

    def __init__(self, songs: List[Dict]) -> None:
        self.songs = songs

        _logger.debug(f"Loading sentence-transformer model: {_MODEL_NAME}")
        self._model = SentenceTransformer(_MODEL_NAME)

        documents = [build_song_document(s) for s in songs]

        embeddings: np.ndarray = self._model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Normalize so that IndexFlatIP == cosine similarity (range 0 – 1)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        _logger.info(f"SongRetriever: indexed {len(songs)} songs (embedding dim={dim})")

    def retrieve(self, query: str, top_k: int = 8) -> List[Tuple[Dict, float]]:
        """Return the top_k most semantically similar songs for *query*.

        Each result is a (song_dict, similarity_score) tuple where
        similarity_score is a cosine similarity in [0.0, 1.0].
        """
        k = min(top_k, len(self.songs))

        query_vec: np.ndarray = self._model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, k)

        results: List[Tuple[Dict, float]] = [
            (self.songs[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
        ]

        _logger.debug(
            f"retrieve: query='{query[:60]}' top_k={k} "
            f"best_score={results[0][1]:.3f}"
        )
        return results
