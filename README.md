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

FAISS runs entirely in memory and requires no database process, configuration file, or migration. The index is a `faiss.IndexFlatIP` storing 18 × 384 = 6,912 float32 values — roughly 27 KB — and rebuilds in milliseconds on any hardware.

More importantly, `IndexFlatIP` performs **exact** brute-force inner-product search: it checks all 18 vectors on every query with no approximation error. Approximate indexes (HNSW, IVF) exist to trade recall for speed at scales of hundreds of thousands or millions of vectors; at 18 vectors the brute-force baseline is already instantaneous, and adding approximation would only reduce recall. L2-normalizing the vectors before insertion converts the inner product to cosine similarity in [0, 1], which makes the similarity scores directly interpretable as a percentage without a separate normalization step.

A persistent store (Chroma, Qdrant, Pinecone) would add a serialization layer, a background database process, index version management, and the risk of stale vectors if `songs.csv` is updated without re-running ingestion. Those tradeoffs are worthwhile when the index must survive across processes or grow too large for RAM. At this scale, rebuilding from scratch on every run is strictly cheaper and eliminates an entire class of cache-invalidation bugs.

### Why `all-MiniLM-L6-v2` for embeddings?

`all-MiniLM-L6-v2` is a 6-layer knowledge-distilled BERT variant that produces 384-dimensional vectors — half the width of the full `all-mpnet-base-v2` model. Smaller vectors mean a smaller FAISS index and faster dot-product math, with only a marginal recall loss on standard STS benchmarks.

The more important property for this use case: the model was fine-tuned on NLI and Semantic Textual Similarity (STS) tasks to pull paraphrases and conceptually related sentences close together in embedding space. That matters here because the query vocabulary (*"energetic workout track"*) and the document vocabulary (*"Genre: pop. Mood: intense. Energy: 0.93 (high)."*) are systematically different — a bag-of-words retriever would miss the connection, but a sentence transformer bridges it.

Using the same model for both document indexing and query embedding is a hard requirement: switching to a different encoder at query time would project the query into a different vector space, making dot-product scores meaningless. By using one model end-to-end, we guarantee the document and query embeddings share the same coordinate system. This design also avoids a second API dependency — Gemini is reserved for the one task it does best (natural-language generation), while all retrieval runs offline after the initial model download.

### Why use Gemini only for explanations and not for retrieval or scoring?

The rule-based formula has three components with bounded, auditable outputs: `genre_match ∈ {0, 2}`, `mood_match ∈ {0, 1}`, `energy_closeness ∈ [0, 1]`. Confidence is `score / 4.0 × 100`. Every number traces directly to a row in `songs.csv` and can be verified by hand. The evaluation harness (`src/evaluator.py`) runs entirely offline and exits with a deterministic pass/fail because the scoring path never touches an external API.

Using Gemini for scoring would introduce three compounding problems. First, LLM outputs are probabilistic — the same query could produce different rankings on different runs, breaking reproducibility. Second, scoring 18 candidates would require either 18 serial API calls or a single batch prompt that returns 18 scores, both of which are slower and harder to parse reliably. Third, LLM-generated scores carry no natural scale: a Gemini score of "7/10" has no interpretable relationship to specific attribute matches the way the formula does.

Gemini's actual role is narrow and grounded: the structured scoring breakdown — produced deterministically by the rule engine — is passed to Gemini as explicit context, and Gemini is asked only to rephrase that breakdown into a fluent sentence. This sharply reduces the risk of hallucination because Gemini is not deciding *what* to say about the song, only *how* to phrase what the rules already established. The LLM stays entirely out of the critical ranking path.

### Why convert songs to natural language descriptions before embedding, rather than embedding raw attribute vectors?

The 18 songs could be represented as raw numeric feature vectors (e.g., genre as a one-hot, energy as a float) and compared with Euclidean distance. That approach breaks down the moment the query is a sentence: there is no meaningful geometric distance between *"something upbeat for a morning workout"* and a feature vector of `[1, 0, 0, 0.93, 118, ...]`.

By converting each song to a sentence like *"Gym Hero by Max Pulse. Genre: pop. Mood: intense. Energy: 0.93 (high). Tempo: 118 BPM. Very danceable. Electronic sound."*, both the query and the catalog exist in the same modality — natural language — and the sentence transformer handles the vocabulary alignment. The descriptive labels (`"high"`, `"very danceable"`, `"melancholic"`) are chosen to match words a user would naturally use in a free-text query. For example, a user query mentioning "acoustic" maps directly to the `"acoustic sound"` label in the document, whereas a raw `acousticness=0.83` float would require the model to learn that numerical proximity implies semantic similarity — something it was not trained to do for arbitrary domain-specific scales.

### Why rebuild the FAISS index on every run rather than persisting it to disk?

FAISS supports `faiss.write_index()` / `faiss.read_index()` for index persistence. The reason not to use it: at 18 songs with a ~27 KB index, the disk I/O to deserialize a saved index is slower than rebuilding from raw embeddings. More critically, a persisted index can silently go stale — if `songs.csv` is updated and the index file is not regenerated, the retriever returns results from the old catalog with no error. Rebuilding every run is a near-zero cost that removes the need to manage index freshness entirely.

At larger catalog sizes (thousands of songs), the calculation inverts: embedding generation becomes the bottleneck (not disk I/O), and a persistent index with an explicit invalidation strategy would be the right design. The current choice is correct for this scale and does not need to be defended as universally correct.

### Why five steps in the agent pipeline, and what does the REFLECT step add?

The five-step structure (PLAN → RETRIEVE → SCORE → EXPLAIN → REFLECT) maps one observable decision per concern: routing, retrieval, ranking, generation, and quality assurance. Each step produces a human-readable message stored in the `steps` list, so the Agent Trace panel in the UI shows exactly what path was taken and why — without requiring a user to read source code or logs.

REFLECT is the step most easily omitted but most valuable for user trust. It checks whether the top result's confidence exceeds 50%, corresponding to a score of at least 2.0 / 4.0 — meaning the song matches at least one major attribute (genre or mood) plus some energy proximity. If the threshold is not met, the system emits an explicit warning before presenting results. This surfaces a real failure mode: a user querying for a niche genre/mood/energy combination that no song in the catalog satisfies well would otherwise receive a low-confidence result presented with the same visual weight as a 99% match. REFLECT converts that silent degradation into an actionable signal.

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
