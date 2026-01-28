"""Alignment wrapper for deriving timestamps from transcript text via alqalign."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback


def _get_alq_align():
    try:
        from alqalign.app import align as alq_align
        return alq_align
    except ImportError:
        raise ImportError(
            "alqalign is required for transcript alignment but is not installed. "
            "Install it with: pip install chunkseg[align]"
        )


def _is_punct_only(text: str) -> bool:
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    return re.search(r"\w", stripped) is None


def write_temp_text(lines: list[str]) -> str:
    """Write a list of text lines to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="chunkseg_", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for line in lines:
            f.write((line or "").strip() + "\n")
    return path


def update_timestamps_from_alignment(
    sentences: list[dict],
    segments: list[dict],
) -> list[dict]:
    """Merge aligned segment timestamps back onto sentence dicts."""
    updated = []
    for s, seg in zip(sentences, segments):
        s2 = dict(s)
        s2["start"] = float(seg.get("start", s.get("start", 0.0)))
        s2["end"] = float(seg.get("end", s.get("end", s2["start"])))
        updated.append(s2)
    return updated


def align_sentences(
    audio_path: str,
    sentences: list[dict],
    lang: str,
) -> list[dict] | None:
    """Align sentence timestamps against an audio file using alqalign.

    Args:
        audio_path: Path to WAV audio file.
        sentences: List of dicts, each with at least a ``"text"`` key.
        lang: alqalign language code (e.g. ``"eng"``).

    Returns:
        Updated sentence dicts with ``start``/``end`` timestamps, or
        ``None`` if alignment could not be completed.
    """
    alq_align = _get_alq_align()

    if not sentences:
        return sentences

    remaining = list(sentences)
    used_trim_last = False
    exp_step = 1
    max_step = 128

    while True:
        tmp_txt_path = None
        try:
            text_lines = [s.get("text", "").replace("\n", " ") for s in remaining]
            tmp_txt_path = write_temp_text(text_lines)
            segments = alq_align(audio_path, tmp_txt_path, lang)

            if not isinstance(segments, list) or not all(
                isinstance(x, dict) for x in segments
            ):
                sys.stderr.write(
                    "[WARN] Unexpected alqalign output; keeping original timestamps.\n"
                )
                return None

            if len(segments) != len(remaining):
                # Try to explain mismatch via punctuation-only sentences
                punct_positions = [
                    i
                    for i, s in enumerate(remaining)
                    if _is_punct_only(str(s.get("text", "")))
                ]

                if (
                    len(segments) + len(punct_positions) == len(remaining)
                    and len(segments) > 0
                ):
                    keep_indices = [
                        i for i in range(len(remaining)) if i not in punct_positions
                    ]
                    kept_sentences = [remaining[i] for i in keep_indices]
                    aligned_kept = update_timestamps_from_alignment(
                        kept_sentences, segments
                    )
                    aligned_map = {
                        idx: aligned_kept[j] for j, idx in enumerate(keep_indices)
                    }

                    aligned_full = []
                    prev_end = (
                        float(aligned_kept[0].get("start", 0.0))
                        if aligned_kept
                        else 0.0
                    )
                    for i, s in enumerate(remaining):
                        if i in aligned_map:
                            s2 = aligned_map[i]
                            aligned_full.append(s2)
                            prev_end = float(s2.get("end", prev_end))
                        else:
                            s2 = dict(s)
                            s2["start"] = prev_end
                            s2["end"] = prev_end
                            aligned_full.append(s2)
                    return aligned_full

                sys.stderr.write(
                    f"[WARN] Segment count mismatch for {os.path.basename(audio_path)}: "
                    f"{len(segments)} (aligned) vs {len(remaining)} (sentences).\n"
                )
                return None

            return update_timestamps_from_alignment(remaining, segments)

        except FileNotFoundError:
            if len(remaining) <= 1:
                return None

            if not used_trim_last:
                n = len(remaining)
                remaining = remaining[:-1]
                if remaining:
                    pivot_text = remaining[-1].get("text", "")
                    remaining = [
                        s
                        for s in remaining[:-1]
                        if s.get("text", "") != pivot_text
                    ] + [remaining[-1]]
                used_trim_last = True
                if not remaining:
                    return None
            else:
                step = min(exp_step, max_step)
                trim_n = min(step, len(remaining) - 1)
                remaining = remaining[:-trim_n]
                exp_step = min(step * 2, max_step)
                if len(remaining) <= 1:
                    pass  # one more attempt
            continue

        except Exception as e:
            sys.stderr.write(
                f"[WARN] Alignment failed for {audio_path}: {e}\n"
                f"{traceback.format_exc()}\n"
            )
            return None

        finally:
            if tmp_txt_path and os.path.exists(tmp_txt_path):
                try:
                    os.remove(tmp_txt_path)
                except Exception:
                    pass


def sections_to_boundary_timestamps(
    aligned_sentences: list[dict],
    sections: list[list[str]],
) -> list[float]:
    """Derive boundary timestamps from aligned sentences and section structure.

    Each section consists of a contiguous slice of sentences.  The boundary
    timestamp for each section (except the first) is the ``start`` time of
    the section's first sentence.

    Args:
        aligned_sentences: Flat list of sentence dicts with ``start``/``end``
            timestamps (as returned by :func:`align_sentences`).
        sections: List of sentence lists (as returned by the parsers).

    Returns:
        List of boundary timestamps in seconds.
    """
    boundaries: list[float] = []
    idx = 0
    for sec_i, section in enumerate(sections):
        if sec_i > 0 and idx < len(aligned_sentences):
            boundaries.append(float(aligned_sentences[idx]["start"]))
        idx += len(section)
    return boundaries
