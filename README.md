# AI Music Recommender System
### CodePath AI110 — Project 4: Applied AI System

A rule-based music recommender extended with an **observable agentic workflow**, RAG retrieval, Gemini-powered explanations, **specialized explanation styles**, a Streamlit frontend, and an offline evaluation harness. Built incrementally from the Project 3 foundation.

## 🎥 Demo Video

[Watch Demo](https://drive.google.com/file/d/1iG6v9oQAHJRnlcda7hwoKIHzmpNSBpkP/view?usp=sharing)

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

Project 4 added six significant capabilities on top of the working Project 3 base without changing the core scoring formula.

### 1. Observable Agentic Workflow

`src/agent.py` wraps the full pipeline in five labelled, human-readable steps that are visible in the UI as an **Agent Trace**:

| Step | What it does |
|---|---|
| **PLAN** | Decides whether to use RAG (natural language query present) or score the full catalog directly |
| **RETRIEVE** | Runs FAISS semantic search and returns candidates with similarity scores, or passes the full catalog |
| **SCORE** | Calls the rule-based engine and selects the top-k results |
| **EXPLAIN** | Generates explanations via Gemini (or deterministic fallback) in the selected style |
| **REFLECT** | Checks top confidence against a 50% threshold and emits warnings for weak matches |

The agent returns a structured dict — `steps`, `recommendations`, `explanations`, `rag_candidates`, `sim_lookup`, `warnings` — that the Streamlit UI renders directly.

### 2. Specialized Explanation Styles

`src/explainer.py` supports two **explanation styles** selectable from the sidebar:

- **Standard** — factual, attribute-by-attribute breakdown (*"Your preferred genre (pop) matches exactly; energy (0.82) is nearly identical to your target (0.85)."*)
- **Music Coach** — short, casual, vibe-focused sentence (*"Vibe check (99% match) — Your preferred genre (pop) matches exactly."*)

Both styles work with and without a Gemini API key. The style is forwarded through `run_recommendation_agent()` → `explain_recommendations()` so the full pipeline respects the user's choice.

`src/style_evaluator.py` provides a standalone comparison script:

```bash
python -m src.style_evaluator
```

### 3. RAG Retrieval (FAISS + sentence-transformers)

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

### 4. Streamlit Frontend

A browser-based UI (`src/app.py`) with sidebar controls and main area results. Every recommendation card shows five metrics (Genre, Mood, Energy, Confidence %, Similarity %) and the selected explanation style. The app includes three expanders:

- **Agent Trace** — shows all five PLAN→RETRIEVE→SCORE→EXPLAIN→REFLECT steps with icons
- **RAG Retrieved Candidates** — lists the FAISS-retrieved songs before scoring (only shown when RAG is active)
- **View Indexed Song Document Example** — shows how songs are represented as text before embedding

### 6. Evaluation Harness

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
   ┌─────────────────────────────────┐
   │   AGENT (src/agent.py)          │
   │                                 │
   │  PLAN                           │
   │  Decide RAG vs. full catalog    │
   │         |                       │
   │  RETRIEVE                       │
   │    NL query?                    │
   │    Yes |      | No              │
   │        v      v                 │
   │  [FAISS     [Full               │
   │  Retriever]  Catalog]           │
   │    Embed → search → candidates  │
   │        |      |                 │
   │        +------+                 │
   │         |                       │
   │  SCORE                          │
   │  score = genre_match(+2.0)      │
   │        + mood_match(+1.0)       │
   │        + energy_closeness(0–1)  │
   │  confidence_pct = score/4.0*100 │
   │         |                       │
   │  EXPLAIN                        │
   │  Gemini (standard/music_coach)  │
   │  or deterministic fallback      │
   │         |                       │
   │  REFLECT                        │
   │  Check top confidence vs 50%    │
   │  Emit warnings if weak match    │
   └─────────────────────────────────┘
         |
         v
     [Output]
  Title · Artist
  Genre | Mood | Energy | Confidence % | Similarity %
  Explanation (selected style)
  ▸ Agent Trace expander
  ▸ RAG Candidates expander (if RAG active)
  ▸ Index Document Example expander
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

### 7. Run the style comparison script

```bash
python -m src.style_evaluator
```

### 8. Run all tests

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

All 46 automated tests pass (`pytest`).

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
│   └── songs.csv                  # 18-song catalog
├── src/
│   ├── main.py                    # CLI runner (3 hardcoded profiles)
│   ├── app.py                     # Streamlit frontend (sidebar + agent integration)
│   ├── agent.py                   # Observable 5-step agentic workflow
│   ├── recommender.py             # Scoring engine + load_songs()
│   ├── retriever.py               # FAISS RAG retriever + preview_index_documents()
│   ├── explainer.py               # Gemini explainer with style support
│   ├── style_evaluator.py         # Side-by-side style comparison script
│   ├── guardrails.py              # Input validation
│   ├── logger_config.py           # Logging setup
│   └── evaluator.py               # Offline evaluation harness
├── tests/
│   ├── test_recommender.py        # 2 tests (scoring + OOP path)
│   ├── test_retriever.py          # 9 tests (FAISS + embeddings)
│   ├── test_explainer.py          # 12 tests (fallback + prompt + parser)
│   ├── test_agent.py              # 14 tests (agentic workflow)
│   └── test_style_evaluator.py    # 9 tests (explanation styles)
├── logs/
│   └── recommender.log            # Generated at runtime
├── .env.example                   # API key template
├── requirements.txt
├── README.md
├── model_card.md
└── architecture.md
```
