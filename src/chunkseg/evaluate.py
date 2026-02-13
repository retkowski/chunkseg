"""Main evaluation API: ``evaluate()`` and ``evaluate_batch()``."""

from __future__ import annotations

import numpy as np

from .core import build_target_sequence
from .metrics import aggregate_metrics, compute_metrics
from .parsers import parse_transcript


def evaluate(
    hypothesis: list[float] | str,
    reference: list[float],
    duration: float,
    chunk_size: float = 6.0,
    *,
    audio: str | None = None,
    format: str | None = None,
    custom_pattern: str | None = None,
    timestamp_format: str | None = None,
    lang: str = "eng",
    force_alignment: bool = False,
) -> dict:
    """Evaluate a single segmentation hypothesis against a reference.

    The *hypothesis* parameter type determines the evaluation mode:

    * ``list[float]`` — **Timestamps mode**.  Both *hypothesis* and
      *reference* are boundary timestamps in seconds.
    * ``str`` — **Transcript mode**.  The hypothesis is raw text that will
      be parsed (according to *format*) and optionally aligned to audio to
      derive boundary timestamps.  Requires *format*.

    Args:
        hypothesis: Predicted boundaries (timestamps) or raw transcript text.
        reference: Reference boundary timestamps in seconds.
        duration: Total duration in seconds.
        chunk_size: Chunk size in seconds (default 6.0).
        audio: Path to audio file (transcript mode only, required when
            timestamps are not embedded in the text).
        format: Parser preset. Required for transcript mode. Options:

            - ``"cstart"`` — ``[CSTART] Title [CEND]`` format
            - ``"cstart_ts"`` — ``[CSTART] HH:MM:SS - Title [CEND]`` format
            - ``"newline"`` — sections separated by blank lines
            - ``"markdown"`` — markdown headers (``# Title``)
            - ``"markdown_ts"`` — markdown with timestamps (``# 00:15:30 - Title``)
            - ``"custom"`` — user-provided regex pattern
            - ``"custom_ts"`` — user-provided regex with timestamp capture

        custom_pattern: Regex pattern when *format* is ``"custom"`` or ``"custom_ts"``.
        timestamp_format: Timestamp format for ``"custom_ts"`` (default: ``"H:MM:SS"``).
            Supported: ``"HH:MM:SS"``, ``"H:MM:SS"``, ``"MM:SS"``, ``"M:SS"``, or
            a custom regex with named groups ``h``/``m``/``s``.
        lang: ISO 639-3 language code for forced alignment (default ``"eng"``).
        force_alignment: If True, ignore embedded timestamps and derive them from
            audio alignment instead. Requires audio path. (default False).

    Returns:
        Dict mapping metric names to float values.
    """
    if isinstance(hypothesis, str):
        hyp_timestamps = _timestamps_from_transcript(
            hypothesis, audio=audio, format=format,
            custom_pattern=custom_pattern, timestamp_format=timestamp_format,
            lang=lang, force_alignment=force_alignment,
        )
    else:
        hyp_timestamps = list(hypothesis)

    ref_seq = build_target_sequence(duration, reference, chunk_size)
    hyp_seq = build_target_sequence(duration, hyp_timestamps, chunk_size)

    # Ensure same length (should match given same duration/chunk_size)
    min_len = min(len(ref_seq), len(hyp_seq))
    if min_len == 0:
        return {}
    ref_seq = ref_seq[:min_len]
    hyp_seq = hyp_seq[:min_len]

    return compute_metrics(hyp_seq, ref_seq)


def evaluate_batch(
    samples: list[dict],
    chunk_size: float = 6.0,
    *,
    format: str | None = None,
    custom_pattern: str | None = None,
    timestamp_format: str | None = None,
    lang: str = "eng",
    num_bootstrap: int = 100,
    force_alignment: bool = False,
) -> dict:
    """Evaluate a batch of samples and return aggregated metrics.

    Each sample is a dict with keys matching :func:`evaluate` parameters:
    ``hypothesis``, ``reference``, ``duration``, and optionally ``audio``.

    For transcript mode, each sample must include an ``audio`` field with
    the full path to the audio file (required when timestamps are not
    embedded in the transcript text).

    Args:
        samples: List of sample dicts. Required keys: ``hypothesis``,
            ``reference``, ``duration``. Optional: ``audio``, ``format``.
        chunk_size: Chunk size in seconds (default 6.0).
        format: Parser preset override (applied to all transcript samples).
        custom_pattern: Regex pattern when *format* is ``"custom"`` or ``"custom_ts"``.
        timestamp_format: Timestamp format for ``"custom_ts"`` (default: ``"H:MM:SS"``).
        lang: ISO 639-3 language code for forced alignment (default ``"eng"``).
        num_bootstrap: Number of bootstrap iterations for CIs.
        force_alignment: If True, ignore embedded timestamps and derive them from
            audio alignment instead. Requires audio paths in samples. (default False).

    Returns:
        Dict mapping metric names to dicts of
        ``{mean, std, ci_lower, ci_upper}``.  Includes a derived ``f1``
        entry computed from aggregated precision/recall.
    """
    all_metrics: list[dict] = []

    for sample in samples:
        sample_format = format or sample.get("format")

        result = evaluate(
            hypothesis=sample["hypothesis"],
            reference=sample["reference"],
            duration=sample["duration"],
            chunk_size=chunk_size,
            audio=sample.get("audio"),
            format=sample_format,
            custom_pattern=custom_pattern,
            timestamp_format=timestamp_format,
            lang=lang,
            force_alignment=force_alignment,
        )
        if result:
            all_metrics.append(result)

    return aggregate_metrics(all_metrics, num_iterations=num_bootstrap)


def _timestamps_from_transcript(
    text: str,
    *,
    audio: str | None,
    format: str | None,
    custom_pattern: str | None,
    timestamp_format: str | None,
    lang: str,
    force_alignment: bool = False,
) -> list[float]:
    """Parse transcript text and derive boundary timestamps.

    If the format includes timestamps (``"cstart_ts"``, ``"markdown_ts"``,
    ``"custom_ts"``) and force_alignment is False, those timestamps are used
    directly without alignment.

    If force_alignment is True or the format has no embedded timestamps,
    the text is parsed into sections, flattened into sentences, aligned to
    audio via alqalign, and boundary timestamps are derived from the aligned
    section boundaries.
    """
    if format is None:
        raise ValueError(
            "format is required when hypothesis is a string (transcript mode)"
        )

    # Parse the transcript
    result = parse_transcript(
        text, format,
        custom_pattern=custom_pattern,
        timestamp_format=timestamp_format,
    )

    # If timestamps are embedded in the format, use them directly
    # (unless force_alignment is True)
    if result.timestamps and not force_alignment:
        return result.timestamps

    sections = result.sections
    if not sections:
        return []

    if audio is None:
        raise ValueError(
            "audio path is required for transcript mode when timestamps "
            "are not embedded in the text"
        )

    # Build sentence dicts for alignment
    from .align import align_sentences, sections_to_boundary_timestamps

    sentence_dicts: list[dict] = []
    for section in sections:
        for sent in section:
            sentence_dicts.append({"text": sent})

    aligned = align_sentences(audio, sentence_dicts, lang)
    if aligned is None:
        return []

    return sections_to_boundary_timestamps(aligned, sections)
