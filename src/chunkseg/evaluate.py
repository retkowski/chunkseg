"""Main evaluation API: ``evaluate()`` and ``evaluate_batch()``."""

from __future__ import annotations

import numpy as np

from .core import build_target_sequence
from .metrics import aggregate_metrics, collar_boundary_f1, compute_metrics, compute_wer
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
    fa_backend: str = "mms_fa",
    force_alignment: bool = False,
    reference_transcript: str | None = None,
    reference_titles: list[tuple[str, float]] | None = None,
    hyp_titles: list[tuple[str, float]] | None = None,
    tolerance: float = 5.0,
    collar: float = 3.0,
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
        fa_backend: Forced alignment backend: ``"mms_fa"`` (default, requires
            ``chunkseg[align]``) or ``"alqalign"`` (see
            https://github.com/xinjli/alqalign; includes exponential trimming
            for long transcripts).
        force_alignment: If True, ignore embedded timestamps and derive them from
            audio alignment instead. Requires audio path. (default False).
        reference_transcript: Reference transcript text for WER computation.
            When provided with a string hypothesis, WER is automatically computed.
        reference_titles: Reference titles as ``(title, start_seconds)`` pairs
            for title evaluation (BERTScore + ROUGE-L). When provided with a
            string hypothesis whose format includes titles and timestamps, title
            metrics are computed automatically.
        hyp_titles: Hypothesis titles as ``(title, start_seconds)`` pairs.
            If ``None`` and *hypothesis* is a string with titles, they are
            extracted from the parsed transcript automatically.
        tolerance: Time tolerance in seconds for temporally-matched title
            scoring (TM-BS / TM-RL, default 5.0).
        collar: Collar size in seconds for collar-based boundary F1 (default 3.0).

    Returns:
        Dict mapping metric names to float values.
    """
    if isinstance(hypothesis, str):
        hyp_timestamps = _timestamps_from_transcript(
            hypothesis, audio=audio, format=format,
            custom_pattern=custom_pattern, timestamp_format=timestamp_format,
            lang=lang, fa_backend=fa_backend, force_alignment=force_alignment,
        )
    else:
        hyp_timestamps = list(hypothesis)

    # 0.0 is never a valid inter-section boundary
    reference = [t for t in reference if t > 0]
    hyp_timestamps = [t for t in hyp_timestamps if t > 0]

    ref_seq = build_target_sequence(duration, reference, chunk_size)
    hyp_seq = build_target_sequence(duration, hyp_timestamps, chunk_size)

    # Ensure same length (should match given same duration/chunk_size)
    min_len = min(len(ref_seq), len(hyp_seq))
    if min_len == 0:
        return {}
    ref_seq = ref_seq[:min_len]
    hyp_seq = hyp_seq[:min_len]

    metrics = compute_metrics(hyp_seq, ref_seq)

    # Collar-based boundary F1 (always computed from raw timestamps)
    metrics.update(collar_boundary_f1(reference, hyp_timestamps, collar))

    # Compute WER if reference transcript is provided
    if reference_transcript is not None and isinstance(hypothesis, str):
        hyp_text = _extract_hypothesis_text(
            hypothesis, format=format,
            custom_pattern=custom_pattern,
            timestamp_format=timestamp_format,
        )
        if hyp_text:
            metrics.update(compute_wer(hyp_text, reference_transcript))

    # Compute BERTScore title metrics if reference titles are provided
    if reference_titles is not None:
        _hyp_titles = hyp_titles
        if _hyp_titles is None and isinstance(hypothesis, str):
            _hyp_titles = _extract_hypothesis_titles(
                hypothesis, format=format,
                custom_pattern=custom_pattern,
                timestamp_format=timestamp_format,
                hyp_timestamps=hyp_timestamps,
            )
        if _hyp_titles:
            from .titles import compute_title_scores
            metrics.update(compute_title_scores(
                _hyp_titles, reference_titles, tolerance=tolerance,
            ))

    return metrics


def evaluate_batch(
    samples: list[dict],
    chunk_size: float = 6.0,
    *,
    format: str | None = None,
    custom_pattern: str | None = None,
    timestamp_format: str | None = None,
    lang: str = "eng",
    fa_backend: str = "mms_fa",
    num_bootstrap: int = 100,
    force_alignment: bool = False,
    wer: bool = False,
    titles: bool = False,
    tolerance: float = 5.0,
    collar: float = 3.0,
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
        fa_backend: Forced alignment backend: ``"mms_fa"`` (default) or
            ``"alqalign"`` (see https://github.com/xinjli/alqalign; includes trimming).
        num_bootstrap: Number of bootstrap iterations for CIs.
        force_alignment: If True, ignore embedded timestamps and derive them from
            audio alignment instead. Requires audio paths in samples. (default False).
        wer: If True, compute Word Error Rate. Each sample must include a
            ``reference_transcript`` field. (default False).
        titles: If True, compute title metrics: BERTScore (TM-BS, GC-BS) and
            ROUGE-L (TM-RL, GC-RL). Each sample must include a
            ``reference_titles`` field with ``[[title, start_seconds], ...]``
            entries. (default False).
        tolerance: Time tolerance in seconds for TM-BS matching (default 5.0).

    Returns:
        Dict mapping metric names to dicts of
        ``{mean, std, ci_lower, ci_upper}``.  Includes a derived ``f1``
        entry computed from aggregated precision/recall.
    """
    all_metrics: list[dict] = []

    for sample in samples:
        sample_format = format or sample.get("format")

        ref_transcript = sample.get("reference_transcript") if wer else None

        ref_titles = None
        sample_hyp_titles = None
        if titles:
            raw = sample.get("reference_titles")
            if raw is not None:
                ref_titles = [(t, s) for t, s in raw]
            raw_hyp = sample.get("hyp_titles")
            if raw_hyp is not None:
                sample_hyp_titles = [(t, s) for t, s in raw_hyp]

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
            fa_backend=fa_backend,
            force_alignment=force_alignment,
            reference_transcript=ref_transcript,
            reference_titles=ref_titles,
            hyp_titles=sample_hyp_titles,
            tolerance=tolerance,
            collar=collar,
        )
        if result:
            all_metrics.append(result)

    return aggregate_metrics(all_metrics, num_iterations=num_bootstrap)


def _extract_hypothesis_text(
    text: str,
    *,
    format: str | None,
    custom_pattern: str | None,
    timestamp_format: str | None,
) -> str:
    """Parse structured transcript and return flattened plain text."""
    if format is None:
        return text
    result = parse_transcript(
        text, format,
        custom_pattern=custom_pattern,
        timestamp_format=timestamp_format,
    )
    return " ".join(
        sent for section in result.sections for sent in section
    )


def _extract_hypothesis_titles(
    text: str,
    *,
    format: str | None,
    custom_pattern: str | None,
    timestamp_format: str | None,
    hyp_timestamps: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Parse structured transcript and return (title, start_time) pairs."""
    if format is None:
        return []
    result = parse_transcript(
        text, format,
        custom_pattern=custom_pattern,
        timestamp_format=timestamp_format,
    )
    if not result.titles:
        return []
    # Build section start times using the same timestamps as segmentation.
    if hyp_timestamps is not None:
        section_starts = [0.0] + list(hyp_timestamps)
    elif result.timestamps:
        section_starts = list(result.timestamps)
    else:
        return []
    n = min(len(result.titles), len(section_starts))
    return list(zip(result.titles[:n], section_starts[:n]))


def _timestamps_from_transcript(
    text: str,
    *,
    audio: str | None,
    format: str | None,
    custom_pattern: str | None,
    timestamp_format: str | None,
    lang: str,
    fa_backend: str = "mms_fa",
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

    aligned = align_sentences(audio, sentence_dicts, lang, backend=fa_backend)
    if aligned is None:
        return []

    return sections_to_boundary_timestamps(aligned, sections)
