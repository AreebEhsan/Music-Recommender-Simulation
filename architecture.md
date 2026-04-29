# System Architecture — AI Music Recommender

---

## Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Input                                   │
│   Streamlit app (src/app.py)  /  CLI (src/main.py)             │
│                                                                 │
│   genre (selectbox)   mood (selectbox)   energy (slider)        │
│   top_k (slider)      natural language query (text, optional)   │
│   "Use Gemini" checkbox                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Guardrails                                   │
│                  src/guardrails.py                              │
│                                                                 │
│   • genre non-empty?  if not → raise ValueError                 │
│   • mood non-empty?   if not → raise ValueError                 │
│   • energy in [0, 1]? if not → clamp + log WARNING             │
│   • returns cleaned copy of user_prefs dict                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ clean user_prefs
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Catalog Loader                               │
│                 src/recommender.py → load_songs()               │
│                                                                 │
│   data/songs.csv → 18 song dicts                               │
│   fields: id, title, artist, genre, mood, energy,              │
│           tempo_bpm, valence, danceability, acousticness        │
└──────────────┬───────────────────────────────┬──────────────────┘
               │                               │
     NL query provided?                  No NL query
               │ Yes                           │
               ▼                               │
┌──────────────────────────┐                   │
│     RAG Retriever        │                   │
│   src/retriever.py       │                   │
│                          │                   │
│  build_song_document()   │                   │
│  → natural language text │                   │
│    per song              │                   │
│                          │                   │
│  SentenceTransformer     │                   │
│  (all-MiniLM-L6-v2)      │                   │
│  → embed all documents   │                   │
│                          │                   │
│  faiss.IndexFlatIP       │                   │
│  → cosine similarity     │                   │
│    index (in-memory)     │                   │
│                          │                   │
│  retrieve(query, top_k)  │                   │
│  → embed query           │                   │
│  → search index          │                   │
│  → List[(song, score)]   │                   │
└──────────────┬───────────┘                   │
               │ candidate songs               │ all 18 songs
               └───────────────┬───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Scoring Engine                               │
│              src/recommender.py → score_song()                  │
│                                                                 │
│   For each candidate song:                                      │
│                                                                 │
│   score  =  (song.genre == pref.genre)  * 2.0                  │
│           + (song.mood  == pref.mood)   * 1.0                  │
│           + max(0, 1 - |song.energy - pref.energy|)            │
│                                                                 │
│   max possible score: 4.0                                       │
│                                                                 │
│   confidence_pct = round(score / 4.0 * 100, 1)                 │
│                                                                 │
│   explanation = human-readable string describing                │
│                 which factors matched and by how much           │
│                                                                 │
│   → sorted descending by score, top-k returned                 │
│   → List[(song, score, confidence_pct, explanation)]           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ranked recommendations
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gemini Explainer                             │
│                  src/explainer.py                               │
│                                                                 │
│   If GEMINI_API_KEY set AND "Use Gemini" checked:               │
│     • build prompt from user query + song metadata + scores     │
│     • call gemini-1.5-flash                                     │
│     • parse numbered response → List[str]                       │
│                                                                 │
│   Otherwise (fallback):                                         │
│     • return rule-based explanation strings from scoring step   │
│     • log WARNING (key absent) or INFO (disabled by user)       │
│                                                                 │
│   Caller never receives an exception — fallback is always ready │
└───────────────────────────┬─────────────────────────────────────┘
                            │ List[str] explanations
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Display                               │
│                                                                 │
│   For each recommendation:                                      │
│     #rank — Title · Artist                                      │
│     Genre | Mood | Energy | Confidence %                        │
│     [Gemini or fallback explanation]                            │
│     ▸ Scoring breakdown (expander)                              │
│       score / 4.00 | similarity % (if RAG) | rule explanation   │
│                                                                 │
│   System Trace expander shows each pipeline step               │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   Logger            │   ← cross-cutting concern
                    │ src/logger_config.py│     runs at every step
                    │                     │
                    │ console → INFO      │
                    │ logs/               │
                    │  recommender.log    │
                    │  → DEBUG            │
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │   Evaluator         │   ← separate offline script
                    │ src/evaluator.py    │
                    │                     │
                    │ 4 predefined cases  │
                    │ uses scoring only   │
                    │ (no RAG, no Gemini) │
                    │                     │
                    │ python -m           │
                    │   src.evaluator     │
                    └─────────────────────┘
```

---

## Component Reference

### `src/guardrails.py`

Validates the user preference dict before anything else runs. It is the system's first line of defense against bad input. Raises `ValueError` for missing genre or mood so the UI can surface a clear error message to the user. Clamps energy silently and records a warning in the log so unexpected values are traceable without crashing the app.

### `src/recommender.py`

The core of the system, unchanged from Project 3 in terms of scoring logic. Contains two parallel implementations: a functional path (`load_songs`, `score_song`, `recommend_songs`) used by the Streamlit app and the CLI, and an object-oriented path (`Song`, `UserProfile`, `Recommender`) kept for the existing unit tests. Both implement the same formula.

### `src/retriever.py`

Implements the RAG (Retrieval-Augmented Generation) layer. `build_song_document()` converts each song dict into a descriptive English sentence that includes qualitative labels (e.g., *"Energy: 0.93 (high). Danceability: 0.88 (very danceable)."*) alongside the raw numbers. This enriched representation improves semantic matching for vague queries like *"something upbeat"*. The `SongRetriever` class builds an in-memory FAISS `IndexFlatIP` index using L2-normalized embeddings, meaning every similarity score returned is a true cosine similarity in `[0, 1]`.

### `src/explainer.py`

A thin wrapper around the Gemini API that is designed to never propagate exceptions to the caller. The decision tree is: `use_gemini=False` → fallback. API key missing → fallback + log warning. API key present → call Gemini → if any exception → fallback + log warning. The prompt explicitly instructs the model to base every sentence only on the provided metadata and forbids inventing popularity or streaming data.

### `src/logger_config.py`

Sets up a named Python logger with two handlers: a `StreamHandler` at `INFO` level for clean terminal output, and a `FileHandler` at `DEBUG` level that writes everything to `logs/recommender.log`. The `if logger.handlers: return logger` guard prevents duplicate log lines when the module is imported more than once within the same Streamlit session.

### `src/evaluator.py`

A standalone script that bypasses the Streamlit UI, the RAG retriever, and the Gemini explainer. It calls `recommend_songs()` directly with four pre-specified profiles and checks two conditions per case: genre of top result and minimum confidence percentage. Returns exit code `0` (all pass) or `1` (any fail), making it suitable as a step in a CI/CD pipeline.

### `src/app.py`

The Streamlit frontend. Adds the project root to `sys.path` at the top of the file so all `src.*` imports resolve correctly regardless of the working directory from which `streamlit run` is invoked. The natural language query input is optional: when left empty, the RAG path is skipped and the system behaves identically to the CLI. A spinner is shown during FAISS index construction and during the Gemini API call.

### `src/main.py`

The original CLI runner from Project 3, preserved unchanged. Runs three hardcoded profiles (High-Energy Pop, Chill Lofi, Intense Rock) and prints results to the terminal. Useful for quick offline demos without opening a browser.

---

## Technology Choices

| Component | Technology | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, offline after first download, strong semantic quality for short text |
| Vector index | FAISS `IndexFlatIP` | In-memory, zero config, cosine similarity via L2 normalization |
| LLM explanations | Gemini `gemini-1.5-flash` | Fast, inexpensive, good at following structured prompts |
| Frontend | Streamlit | Minimal boilerplate for interactive Python apps |
| Logging | Python `logging` module | Built-in, no extra dependency, dual handler pattern is simple to extend |
| Testing | pytest | Standard Python testing framework, works with `monkeypatch` for env var isolation |
