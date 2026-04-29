# AI Music Recommender System
### CodePath AI110 — Project 4: Applied AI System

A rule-based music recommender extended with RAG retrieval, Gemini-powered explanations, a Streamlit frontend, and an offline evaluation harness. Built incrementally from the Project 3 foundation.

---

## Project 3 Foundation

The original system (Project 3) was a pure rule-based CLI recommender. Given a user's preferred genre, mood, and target energy level, it scored every song in a 10-song catalog and returned the top-k matches with a short explanation.

**What Project 3 included:**
- `data/songs.csv` — 10 songs with 9 attributes each (genre, mood, energy, tempo_bpm, valence, danceability, acousticness, title, artist)
- Scoring formula: `genre match (+2.0) + mood match (+1.0) + max(0, 1 - |song_energy - target_energy|)`
- Ranked top-k output with rule-based explanations
- Three hardcoded user profiles run via `python -m src.main`
- Two passing unit tests and a model card

---

## Project 4 Extensions

Project 4 added four significant capabilities on top of the working Project 3 base without changing the core scoring formula.

### 1. RAG Retrieval (FAISS + sentence-transformers)

The system can now accept a **natural language query** such as *"something upbeat and energetic for a morning workout"* instead of — or alongside — the structured genre/mood/energy inputs.

When a query is provided:
1. Each song in the catalog is converted into a natural language description (e.g., *"Gym Hero by Max Pulse. Genre: pop. Mood: intense. Energy: 0.93 (high)..."*).
2. All descriptions are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index.
3. The user query is embedded with the same model and the top semantically similar songs are retrieved.
4. Only the retrieved candidates are passed to the scoring engine for re-ranking.

If no query is given, all 18 songs are scored directly (same behavior as Project 3).

### 2. Gemini Explanation Layer

After scoring, the top recommendations are sent to the Gemini API (`gemini-1.5-flash`) along with a structured prompt grounding Gemini strictly in the song metadata. Gemini returns 1–2 sentences per song explaining why it fits the user's preferences in plain English.

If `GEMINI_API_KEY` is not set, the system falls back to the rule-based explanation string automatically — the app never breaks.

### 3. Streamlit Frontend

A browser-based UI (`src/app.py`) replaces the terminal output. Users select genre, mood, energy, and the number of recommendations using dropdowns and sliders, optionally add a natural language query, and toggle Gemini explanations on or off. Results show each song's confidence %, Gemini explanation, and a collapsible scoring breakdown.

### 4. Evaluation Harness

`src/evaluator.py` defines four predefined user profiles and checks that the top result for each meets a genre match and minimum confidence threshold. It runs fully offline with no API calls and exits with a non-zero code if any case fails, making it CI-compatible.

---

## How the System Works

```
User Input (Streamlit or CLI)
         |
         v
   [Guardrails]
   Validate genre, mood (non-empty)
   Clamp energy to [0.0, 1.0]
   Log warnings if corrected
         |
         v
  [Catalog Loader]
  Read data/songs.csv → 18 song dicts
         |
    NL query?
    Yes |      | No
        v      v
  [FAISS     [Full
  Retriever]  Catalog]
    Embed query
    Search index
    Return top candidates
        |      |
        +------+
         |
         v
   [Scoring Engine]
   score = genre_match(+2.0)
         + mood_match(+1.0)
         + energy_closeness(0–1.0)
   confidence_pct = score / 4.0 * 100
         |
         v
  [Gemini Explainer]
  Build prompt from song metadata
  Call gemini-1.5-flash
  Parse numbered response
  (fallback to rule-based text if key absent)
         |
         v
     [Output]
  Title · Artist
  Genre | Mood | Energy | Confidence %
  Gemini explanation
  ▸ Scoring breakdown (expander)
         |
         v
     [Logger]
  logs/recommender.log (DEBUG)
  Console (INFO)
```

---

## Setup Instructions

### 1. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB) on first use. This is cached locally and only happens once.

### 3. Set up your Gemini API key (optional)

Copy the example file and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

The system works without this key — it falls back to rule-based explanations automatically.

### 4. Run the Streamlit app

```bash
streamlit run src/app.py
```

### 5. Run the CLI (no browser required)

```bash
python -m src.main
```

### 6. Run the evaluation harness

```bash
python -m src.evaluator
```

### 7. Run all tests

```bash
pytest
```

---

## Example Usage

### CLI output (one profile shown)

```
============================================================
Profile: High-Energy Pop
Prefs: genre=pop, mood=happy, energy=0.85

Top recommendations:

Sunrise City - Score: 3.97  (Confidence: 99.2%)
Because: Your preferred genre (pop) matches exactly; mood (happy) aligns;
         energy (0.82) is nearly identical to your target (0.85).

Gym Hero - Score: 2.92  (Confidence: 73.0%)
Because: Your preferred genre (pop) matches exactly; mood is intense
         (you prefer happy); energy (0.93) is close to your target (0.85).
```

### Streamlit with RAG + Gemini (example)

**Query:** *"something upbeat and energetic for a morning workout"*
**Genre:** pop | **Mood:** intense | **Energy:** 0.90

**Result #1 — Gym Hero · Max Pulse**
Genre: pop | Mood: intense | Energy: 0.93 | Confidence: 99.2% | Similarity: 87.4%

*Gemini explanation:* "Gym Hero is a near-perfect match — its pop genre and intense mood align exactly with your preferences, and its 0.93 energy is nearly identical to your 0.90 target, making it ideal for a high-energy workout session."

---

## Evaluation Results

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

All 23 automated tests also pass (`pytest`).

---

## Design Decisions

### Why FAISS instead of a persistent vector database?

FAISS runs entirely in memory and requires no database process, configuration file, or migration. With 18 songs the index builds in under a second on every run. A persistent store (e.g., Chroma) would add setup complexity with no meaningful benefit at this scale.

### Why `all-MiniLM-L6-v2` for embeddings?

It is fast, free, runs offline after the first download, and produces strong semantic embeddings for short English text. Using it for both document indexing and query embedding means no second API key is needed for the retrieval step — Gemini is reserved for the single task it does best: generating readable prose.

### Why use Gemini only for explanations and not for retrieval or scoring?

The rule-based scoring formula is transparent and deterministic — you can read exactly why each song ranked where it did. Replacing scoring with an LLM would make the system a black box and harder to debug. Gemini is only used at the final step to translate the structured scoring breakdown into a sentence a human would naturally say.

---

## File Structure

```
Music-Recommender-Simulation/
├── data/
│   └── songs.csv              # 18-song catalog
├── src/
│   ├── main.py                # CLI runner (3 hardcoded profiles)
│   ├── app.py                 # Streamlit frontend
│   ├── recommender.py         # Scoring engine + load_songs()
│   ├── retriever.py           # FAISS RAG retriever
│   ├── explainer.py           # Gemini explanation layer
│   ├── guardrails.py          # Input validation
│   ├── logger_config.py       # Logging setup
│   └── evaluator.py           # Offline evaluation harness
├── tests/
│   ├── test_recommender.py    # 2 tests (scoring + OOP path)
│   ├── test_retriever.py      # 9 tests (FAISS + embeddings)
│   └── test_explainer.py      # 12 tests (fallback + prompt + parser)
├── logs/
│   └── recommender.log        # Generated at runtime
├── .env.example               # API key template
├── requirements.txt
├── README.md
├── model_card.md
└── architecture.md
```
