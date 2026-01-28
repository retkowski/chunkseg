"""Core utilities for time-chunked segmentation: target sequence building and flag conversions."""

import math

import numpy as np


def build_target_sequence(
    duration: float,
    boundaries: list[float],
    chunk_size_sec: float,
) -> np.ndarray:
    """Build a binary 0/1 target vector from boundary timestamps.

    Each element corresponds to a time chunk of ``chunk_size_sec`` seconds.
    A chunk is marked 1 if any boundary falls within it, 0 otherwise.

    Args:
        duration: Total duration in seconds.
        boundaries: List of boundary timestamps in seconds.
        chunk_size_sec: Length of each chunk in seconds.

    Returns:
        1-D numpy array of 0s and 1s with dtype float32.
    """
    if duration <= 0.0:
        return np.zeros(0, dtype=np.float32)

    duration = math.floor(duration / chunk_size_sec) * chunk_size_sec
    num_segments = int(duration / chunk_size_sec)
    targets = np.zeros(num_segments, dtype=np.float32)

    for b in boundaries:
        if b < 0.0 or b > duration:
            continue
        seg_idx = int(b / chunk_size_sec)
        if seg_idx >= num_segments:
            seg_idx = num_segments - 1
        targets[seg_idx] = 1.0

    return targets


def flags_array_to_str(x: np.ndarray) -> str:
    """Convert a binary numpy array to a string of '0' and '1' characters."""
    return "".join(str(int(v)) for v in x.tolist())


def flags_str_to_array(s: str) -> np.ndarray:
    """Convert a binary flag string (e.g. ``'010010'``) to a numpy float32 array."""
    return np.array([float(c) for c in s], dtype=np.float32)


def boundary_flags_to_timestamps(
    boundary_flags: str,
    annotated_sentences: list[dict],
) -> list[float]:
    """Derive boundary timestamps from a binary flag string and sentence metadata.

    For each position *i* where ``boundary_flags[i] == '1'``, the start
    timestamp of sentence *i* is used as the boundary timestamp.
    """
    ts: list[float] = []
    for i, f in enumerate(boundary_flags):
        if f == "1":
            ts.append(float(annotated_sentences[i]["start"]))
    return ts
