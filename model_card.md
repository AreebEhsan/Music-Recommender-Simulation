# Model Card: Music Recommender Simulation

## 1. Model Name
VibeMatch 1.0 (classroom simulation)

## 2. Intended Use
Suggests 3–5 songs from a small catalog based on a user’s preferred genre, mood, and target energy. Intended for learning/demo purposes only, not for production or real users.

## 3. How It Works
- Inputs: song features (genre, mood, energy, tempo_bpm, valence, danceability, acousticness) and user prefs (favorite genre, favorite mood, target energy).
- Scoring (baseline/final): +2.0 if genre matches, +1.0 if mood matches, plus `max(0, 1 - |song_energy - target_energy|)` added once.
- Ranks songs by total score and returns top k. Explanations list which factors contributed for each recommendation.

## 4. Data
- Source: `data/songs.csv` (10 songs).
- Genres/moods: pop/happy, lofi/chill, rock/intense, plus a few adjacent vibes.
- No lyrics or user history; only simple audio-style attributes. The catalog is too small to represent broad musical taste.

## 5. Strengths
- Very transparent: every score is a simple sum, and explanations are printed.
- Easy to tune and reason about for classroom experiments.
- Handles multiple user profiles in one run for quick comparisons.

## 6. Limitations and Bias
- Tiny dataset makes rankings unstable; single songs can dominate.
- Feature set is narrow (no lyrics, culture, or popularity signals).
- Weight choices swing behavior between genre fidelity and vibe diversity; bias can appear if weights don’t match user intent.
- No fairness or diversity controls; profiles assume one fixed taste point.

## 7. Evaluation
- Tested three profiles: High-Energy Pop, Chill Lofi, Intense Rock.
- Ran a weight-shift experiment (genre +1.0, energy doubled) to compare against the baseline. It increased cross-genre diversity but reduced genre alignment, especially for Pop and Rock.
- Baseline scoring was reinstated for final code; observations from the experiment are retained in docs.

## 8. Future Work
- Add diversity controls or minimum genre coverage.
- Support multi-constraint preferences (e.g., mood + tempo ranges) and user history.
- Expand the catalog and include richer features (lyrics sentiment, era, popularity).
- Toggleable weight presets for “strict genre” vs. “vibe first.”

## 9. Personal Reflection
Small, explainable rules felt “smart” when they matched the vibe, but tiny tweaks flipped rankings—reminding me how sensitive recommenders are to weighting. With only 10 songs, bias and instability are obvious, which makes it a good learning example but not a fair system for real users. Clear explanations helped me spot tradeoffs quickly and appreciate why production systems need both better data and careful evaluation.
