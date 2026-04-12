"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    profiles = [
        ("High-Energy Pop", {"genre": "pop", "mood": "happy", "energy": 0.85}),
        ("Chill Lofi", {"genre": "lofi", "mood": "chill", "energy": 0.35}),
        ("Intense Rock", {"genre": "rock", "mood": "intense", "energy": 0.9}),
    ]

    print("Music Recommender Simulation")
    print()

    for name, user_prefs in profiles:
        print("=" * 60)
        print(f"Profile: {name}")
        print(
            "Prefs: "
            f"genre={user_prefs['genre']}, "
            f"mood={user_prefs['mood']}, "
            f"energy={user_prefs['energy']}"
        )

        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\nTop recommendations:\n")
        for song, score, explanation in recommendations:
            print(f"{song['title']} - Score: {score:.2f}")
            print(f"Because: {explanation}")
            print()

        print()


if __name__ == "__main__":
    main()
