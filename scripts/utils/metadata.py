"""Metadata extraction utilities for EEG file paths."""

import re
from pathlib import Path
from typing import Optional


# Pattern for BrainVision files: sub-XXXX_task-YYY_acq-ZZZ_eeg.vhdr
BRAINVISION_PATTERN = re.compile(
    r"sub-(\d+)_task-([^_]+)_acq-([^_]+)(?:_run-(\d+))?_eeg\.vhdr"
)

# Pattern for EEGLAB files in sedation dataset
SEDATION_PATTERN = re.compile(
    r"sub-(\d+)[_/]?(\w+)?.*\.(?:set|fdt)"
)


def infer_subject_from_path(path: str) -> Optional[str]:
    """
    Extract subject ID from a file path.

    Handles various path formats:
    - sub-1010/...
    - sub-1010_task-...
    - .../sub-1010/eeg/...

    Parameters
    ----------
    path : str
        File path to parse.

    Returns
    -------
    Optional[str]
        Subject ID (e.g., "sub-1010") or None if not found.
    """
    path_str = str(path)

    # Try to find sub-XXXX pattern anywhere in path
    match = re.search(r"(sub-\d+)", path_str)
    if match:
        return match.group(1)

    # Try BrainVision filename pattern
    filename = Path(path_str).name
    bv_match = BRAINVISION_PATTERN.match(filename)
    if bv_match:
        return f"sub-{bv_match.group(1)}"

    return None


def infer_condition_from_path(path: str) -> Optional[str]:
    """
    Extract condition from a file path.

    Handles various path formats:
    - sub-1010_task-awake_acq-EO_eeg.vhdr -> "awake_EO"
    - sub-1010_task-sed_acq-rest_run-1_eeg.vhdr -> "sedation_1"
    - .../baseline/... -> "baseline"
    - .../light_sedation/... -> "light_sedation"

    Parameters
    ----------
    path : str
        File path to parse.

    Returns
    -------
    Optional[str]
        Condition name or None if not found.
    """
    path_str = str(path)
    path_obj = Path(path_str)

    # Check for sedation dataset conditions in path
    sedation_conditions = ["baseline", "light_sedation", "deep_sedation", "recovery"]
    for cond in sedation_conditions:
        if f"/{cond}/" in path_str or f"\\{cond}\\" in path_str:
            return cond

    # Try BrainVision filename pattern
    filename = path_obj.name
    bv_match = BRAINVISION_PATTERN.match(filename)
    if bv_match:
        task = bv_match.group(2)
        acq = bv_match.group(3)
        run = bv_match.group(4)

        # Map common task/acq combinations
        if task == "awake":
            return f"awake_eyes_{acq.lower()}" if acq in ("EO", "EC") else f"awake_{acq}"
        elif task == "sed":
            return f"sedation_{run}" if run else "sedation_1"
        else:
            return f"{task}_{acq}"

    # Check for DS005620 conditions in path components
    parts = path_obj.parts
    ds005620_conditions = ["awake_eyes_open", "awake_eyes_closed", "sedation_1"]
    for part in parts:
        if part in ds005620_conditions:
            return part

    return None


def parse_brainvision_filename(filename: str) -> Optional[dict]:
    """
    Parse a BrainVision filename to extract metadata.

    Parameters
    ----------
    filename : str
        Filename like "sub-1010_task-awake_acq-EO_eeg.vhdr"

    Returns
    -------
    Optional[dict]
        Dictionary with keys: subject, task, acq, run, condition, run_id
        or None if parsing fails.
    """
    match = BRAINVISION_PATTERN.match(filename)
    if not match:
        return None

    subject = f"sub-{match.group(1)}"
    task = match.group(2)
    acq = match.group(3)
    run = match.group(4)

    condition = f"{task}_{acq}"
    run_id = condition if run is None else f"{condition}_run-{run}"

    return {
        "subject": subject,
        "task": task,
        "acq": acq,
        "run": run,
        "condition": condition,
        "run_id": run_id,
    }
