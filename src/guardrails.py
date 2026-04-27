from typing import Dict

from src.logger_config import get_logger

_logger = get_logger("guardrails")


def validate_profile(user_prefs: Dict) -> Dict:
    """Validate and clean a user preference dict.

    Rules:
    - genre must be a non-empty string
    - mood must be a non-empty string
    - energy must be a float in [0.0, 1.0]; clamped with a warning if out of range

    Returns a new cleaned dict (original is not mutated).
    Raises ValueError if genre or mood are missing/empty.
    """
    cleaned = dict(user_prefs)

    # --- genre ---
    genre = str(cleaned.get("genre", "")).strip()
    if not genre:
        raise ValueError("genre must not be empty.")
    cleaned["genre"] = genre

    # --- mood ---
    mood = str(cleaned.get("mood", "")).strip()
    if not mood:
        raise ValueError("mood must not be empty.")
    cleaned["mood"] = mood

    # --- energy ---
    try:
        energy = float(cleaned.get("energy", 0.5))
    except (ValueError, TypeError):
        _logger.warning("energy value could not be parsed — defaulting to 0.5")
        energy = 0.5

    if energy < 0.0:
        _logger.warning(f"energy {energy:.2f} is below 0.0; clamping to 0.0")
        energy = 0.0
    elif energy > 1.0:
        _logger.warning(f"energy {energy:.2f} is above 1.0; clamping to 1.0")
        energy = 1.0

    cleaned["energy"] = energy
    return cleaned
