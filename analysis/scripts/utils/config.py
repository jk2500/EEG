"""Configuration constants for EEG analysis scripts."""

from __future__ import annotations


def format_epoch_path(epoch_length: float) -> str:
    """
    Format epoch length for use in file paths.

    Parameters
    ----------
    epoch_length : float
        Epoch length in seconds (e.g., 5.0).

    Returns
    -------
    str
        Formatted string for paths (e.g., '5p00s').
    """
    return f"{epoch_length:.2f}s".replace(".", "p")


# Standard frequency bands
BANDS: list[str] = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]

# DS005620 dataset conditions
DS005620_CONDITIONS: list[str] = ["awake_eyes_open", "awake_eyes_closed", "sedation_1"]

# Sedation resting state dataset conditions
SEDATION_CONDITIONS: list[str] = ["baseline", "light_sedation", "deep_sedation", "recovery"]

# Standard pairwise comparisons for sedation dataset
SEDATION_CONDITION_PAIRS: list[tuple[str, str]] = [
    ("baseline", "light_sedation"),
    ("baseline", "deep_sedation"),
    ("baseline", "recovery"),
    ("light_sedation", "deep_sedation"),
    ("light_sedation", "recovery"),
    ("deep_sedation", "recovery"),
]

# Composite band definitions for aggregated metrics
COMPOSITE_BANDS: dict[str, list[str]] = {
    "high_abc": ["alpha", "beta", "gamma"],
    "high_bg": ["beta", "gamma"],
    "mid_alpha_beta": ["alpha", "beta"],
    "gamma_only": ["gamma"],
    "alpha_only": ["alpha"],
    "low_dt": ["delta", "theta"],
}
