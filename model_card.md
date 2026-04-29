# Model Card — AI Music Recommender System
### VibeMatch 2.0 (CodePath AI110 — Project 4)

---

## 1. Model Overview

**Name:** VibeMatch 2.0
**Type:** Hybrid rule-based + RAG + LLM explainer
**Purpose:** Suggest songs from a small catalog that match a user's stated genre, mood, and energy preferences. Extended from Project 3 (rule-based only) to include semantic retrieval and AI-generated explanations.
**Intended use:** Educational demo. Not intended for production deployment or real user personalization.

---

## 2. Data

**Source:** `data/songs.csv` — a manually curated catalog of 18 songs.

**Attributes per song (9 total):**

| Attribute | Type | Description |
|---|---|---|
| `title` | string | Song name |
| `artist` | string | Artist name |
| `genre` | string | Musical genre (10 genres represented) |
| `mood` | string | Dominant mood (6 moods represented) |
| `energy` | float 0–1 | Overall energy level |
| `tempo_bpm` | float | Beats per minute |
| `valence` | float 0–1 | Musical positivity (1 = very positive) |
| `danceability` | float 0–1 | How suitable for dancing |
| `acousticness` | float 0–1 | Degree of acoustic vs. electronic sound |

**Genres represented:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, classical, electronic

**Moods represented:** happy, chill, intense, relaxed, moody, focused

**What is NOT in the data:** lyrics, streaming counts, release year, popularity, user history, cultural context.

---

## 3. How It Works

The system runs a three-stage pipeline on every request.

### Stage 1 — Input Validation (Guardrails)

`src/guardrails.py` checks that genre and mood are non-empty strings and that energy is a float in `[0.0, 1.0]`. Out-of-range energy values are clamped and a warning is logged. Invalid genre or mood raises a `ValueError` that the UI surfaces to the user.

### Stage 2 — Candidate Retrieval

**With a natural language query (RAG path):**
Each song is converted to a natural language document (e.g., *"Gym Hero by Max Pulse. Genre: pop. Mood: intense. Energy: 0.93 (high)..."*). These documents are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS `IndexFlatIP` index. The user query is embedded with the same model and the top-k most semantically similar songs are retrieved via cosine similarity. Only the retrieved candidates proceed to scoring.

**Without a natural language query (direct path):**
All 18 songs are passed directly to the scoring engine. This is identical to the Project 3 behavior.

### Stage 3 — Scoring and Ranking

`src/recommender.py` scores every candidate using the deterministic formula:

```
score = (song.genre == pref.genre) * 2.0
      + (song.mood  == pref.mood)  * 1.0
      + max(0.0, 1.0 - |song.energy - pref.energy|)

confidence_pct = round(score / 4.0 * 100, 1)
```

Maximum possible score: **4.0** (100% confidence). Songs are sorted by score descending and the top-k are returned.

### Stage 4 — Explanation Generation

`src/explainer.py` sends the ranked results to Gemini (`gemini-1.5-flash`) with a prompt that:
- Lists each song's full metadata and scoring breakdown
- Instructs Gemini to write 1–2 sentences per song
- Explicitly forbids inventing facts not present in the data (no Spotify stats, albums, or chart positions)

If `GEMINI_API_KEY` is absent or the API call fails for any reason, the system falls back to the rule-based explanation string generated during scoring.

---

## 4. Evaluation

### Automated tests

| Test file | Tests | What is covered |
|---|---|---|
| `test_recommender.py` | 2 | Scoring sort order, explanation non-empty |
| `test_retriever.py` | 9 | Index build, result count, result structure, semantic correctness |
| `test_explainer.py` | 12 | Fallback correctness, prompt content, response parsing |
| **Total** | **23** | **All passing, no API key required** |

### Offline evaluation harness

`src/evaluator.py` runs 4 predefined profiles through the scoring engine and checks two conditions per case: (1) the top result's genre matches the expected genre, and (2) the confidence % meets the minimum threshold.

```
EVALUATION REPORT
-------------------------------------------------------
[PASS] upbeat workout       -> Gym Hero (99.2%)
[PASS] chill study          -> Library Rain (100.0%)
[PASS] intense rock         -> Storm Runner (99.8%)
[PASS] relaxed acoustic     -> Coffee Shop Stories (99.5%)
-------------------------------------------------------
Result:             4/4 cases passed
Average confidence: 99.6%
```

---

## 5. Limitations

**Small dataset:** 18 songs is far too few for a real recommender. A single song can dominate an entire genre or mood category, and any profile with a less common genre will get low-confidence fallback results.

**Exact-match genre and mood scoring:** The `+2.0` genre bonus and `+1.0` mood bonus require exact string equality. A user who prefers "indie" will not match songs tagged "indie pop" even though they are related. The RAG retrieval layer partially compensates for this, but the scoring engine still uses exact matching.

**No user history or personalization:** Every session starts fresh. The system has no memory of past preferences, play counts, or feedback signals.

**Gemini grounding is prompt-only:** The anti-hallucination rules are enforced through the prompt, not through a technical constraint. Gemini can still occasionally produce text that goes slightly beyond the provided metadata. The fallback explanation is always available and is guaranteed to be grounded in the scoring logic.

**Energy is the only continuous attribute used for scoring:** Tempo, valence, danceability, and acousticness are embedded in the RAG document representation and affect retrieval, but they do not contribute to the rule-based score. This means two songs with identical genre, mood, and energy rank equally regardless of how different they sound in other dimensions.

---

## 6. Ethical Considerations

**Filter bubble risk:** Because the scoring formula heavily rewards exact genre and mood matches (+2.0 and +1.0 out of a maximum of 4.0), users who always specify the same genre will receive recommendations that never leave that genre. Over time this reinforces narrow listening habits. A diversity control (e.g., requiring at least one result from a different genre) would partially mitigate this.

**Limited representation:** The 18-song catalog was curated manually for demonstration purposes. It does not proportionally represent the global diversity of music genres, languages, or cultural traditions. Any bias introduced by the curation choice is directly reflected in what the system can recommend.

**No sensitive attribute handling:** The system does not collect or store any personal data. User preferences are entered per session and never persisted.

---

## 7. Future Improvements

- **Larger, diverse catalog:** Even a few hundred songs would substantially reduce the winner-takes-all behavior and allow meaningful diversity controls.
- **Extend the scoring formula:** Including valence, danceability, or tempo as weighted scoring terms would make the rule-based engine more nuanced and reduce reliance on exact genre/mood matches.
- **Collaborative filtering:** Tracking which songs users actually select (not just request) would allow personalized recommendations based on behavior, not just stated preferences.
- **Fuzzy genre matching:** Embedding genres in a small taxonomy (e.g., "indie pop" is a subtype of both "pop" and "indie") would allow partial genre credit in scoring.
- **Persistent FAISS index:** Caching the embedding index to disk would remove the model-load delay on subsequent runs, which currently takes a few seconds while `sentence-transformers` loads.
- **Evaluation with human judges:** The current evaluator checks genre and confidence but not subjective recommendation quality. Having listeners rate the suggestions would provide a richer quality signal.
