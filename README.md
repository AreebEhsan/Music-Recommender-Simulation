# Music Recommender Simulation

## Project Summary
An explainable, rule-based recommender that ranks a 10-song catalog (`data/songs.csv`) by how well each track matches a user’s stated genre, mood, and target energy. The CLI runner (`python -m src.main`) shows three example profiles—High-Energy Pop, Chill Lofi, and Intense Rock—and prints top recommendations with reasons. Final scoring uses genre +2.0, mood +1.0, and a single energy closeness term.

## How The System Works
- Song features used: genre, mood, energy (0–1), tempo_bpm, valence, danceability, acousticness.
- User profile: favorite genre, favorite mood, target energy.
- Scoring (baseline): +2.0 if genre matches, +1.0 if mood matches, plus `max(0, 1 - |song_energy - target_energy|)` for energy similarity.
- Ranking: score every song, sort descending, take top k (default 5).
- Explainability: each printed recommendation lists which factors contributed.

## Getting Started
1) Create a virtual environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Run the app:
```bash
python -m src.main
```
4) Run tests:
```bash
pytest
```

## Experiments You Tried
- Weight shift (temporary): lowered genre weight to +1.0 and doubled the energy term. Result: more cross-genre recommendations (high-energy non-pop/non-rock songs rose), increasing diversity but reducing genre fidelity. Chill Lofi barely changed. We reverted to the baseline weights for final code and kept this experiment for discussion.

## Limitations and Risks
- Tiny 10-song catalog; a single track can dominate results.
- No lyrics or cultural context—only simple audio-style features are considered.
- Ranking is sensitive to the chosen weights; small tweaks shift the balance between strict-genre and vibe-driven results.
- Profiles are single-point tastes; no history, diversity controls, or fairness checks.

## Reflection
Tuning the weights showed how quickly “best” shifts: boosting energy made lists feel more varied but less loyal to the requested genre. The printed explanations made these tradeoffs obvious and easy to reason about. With such a small dataset, one song can jump several spots, reminding me that real recommenders need broader data and carefully chosen weights to avoid accidental bias.
