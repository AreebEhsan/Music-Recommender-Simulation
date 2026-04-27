from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        score = 0.0
        if song.genre == user.favorite_genre:
            score += 2.0
        if song.mood == user.favorite_mood:
            score += 1.0
        score += max(0.0, 1.0 - abs(song.energy - user.target_energy))
        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        ranked = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        parts = []
        if song.genre == user.favorite_genre:
            parts.append("genre match (+2.0)")
        if song.mood == user.favorite_mood:
            parts.append("mood match (+1.0)")
        energy_score = max(0.0, 1.0 - abs(song.energy - user.target_energy))
        parts.append(f"energy close (+{energy_score:.2f})")
        return ", ".join(parts)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, float, str]:
    """
    Scores a single song against a user preference dict.
    Returns (total_score, confidence_pct, explanation_string).

    Scoring breakdown (max 4.0):
      +2.0  genre match
      +1.0  mood match
      +0–1  energy closeness: max(0, 1 - |song_energy - target_energy|)

    confidence_pct = round(score / 4.0 * 100, 1)
    """
    score = 0.0
    parts = []

    pref_genre = user_prefs.get("genre", "")
    pref_mood = user_prefs.get("mood", "")
    target_energy = float(user_prefs.get("energy", 0.0))

    if song["genre"] == pref_genre:
        score += 2.0
        parts.append(f"Your preferred genre ({pref_genre}) matches exactly")
    else:
        parts.append(f"Genre is {song['genre']} (you prefer {pref_genre})")

    if song["mood"] == pref_mood:
        score += 1.0
        parts.append(f"mood ({pref_mood}) aligns")
    else:
        parts.append(f"mood is {song['mood']} (you prefer {pref_mood})")

    energy_diff = abs(song["energy"] - target_energy)
    energy_score = max(0.0, 1.0 - energy_diff)
    score += energy_score

    if energy_diff <= 0.05:
        parts.append(
            f"energy ({song['energy']}) is nearly identical to your target ({target_energy})"
        )
    elif energy_diff <= 0.20:
        parts.append(
            f"energy ({song['energy']}) is close to your target ({target_energy})"
        )
    else:
        parts.append(
            f"energy ({song['energy']}) differs from your target ({target_energy})"
        )

    explanation = "; ".join(parts) + "."
    confidence_pct = round((score / 4.0) * 100, 1)
    return score, confidence_pct, explanation


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Returns a list of (song, score, confidence_pct, explanation) tuples, sorted by score.
    """
    scored = []
    for song in songs:
        score, confidence_pct, explanation = score_song(user_prefs, song)
        scored.append((song, score, confidence_pct, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
