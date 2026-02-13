"""Alignment wrapper for deriving timestamps from transcript text via torchaudio MMS_FA."""

from __future__ import annotations

import re
import sys
import traceback


# Module-level cache for torchaudio components
_model = None
_tokenizer = None
_aligner = None
_device = None
_sample_rate = None
_dictionary = None


def _get_torchaudio_components():
    """Lazy-load and cache torchaudio MMS_FA model, tokenizer, and aligner."""
    global _model, _tokenizer, _aligner, _device, _sample_rate, _dictionary
    if _model is not None:
        return _model, _tokenizer, _aligner, _device, _sample_rate, _dictionary

    try:
        import torch
        import torchaudio  # noqa: F401
        from torchaudio.pipelines import MMS_FA as bundle
    except ImportError:
        raise ImportError(
            "torch and torchaudio are required for transcript alignment but "
            "are not installed. Install them with: pip install torch torchaudio"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _sample_rate = bundle.sample_rate  # 16000
    _model = bundle.get_model().to(_device)
    _model.eval()
    _tokenizer = bundle.get_tokenizer()
    _aligner = bundle.get_aligner()
    _dictionary = bundle.get_dict()

    return _model, _tokenizer, _aligner, _device, _sample_rate, _dictionary


def _normalize_text_for_alignment(text: str) -> str:
    """Normalize text for MMS forced alignment."""
    text = text.lower()
    text = text.replace("\u2019", "'")  # right single quote -> apostrophe
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def _load_audio(audio_path: str, target_sample_rate: int, device):
    """Load audio file, resample to target rate, and convert to mono."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )

    return waveform.to(device)


def _compute_emissions(model, waveform, device):
    """Compute emission log-probabilities from the acoustic model."""
    import torch

    try:
        with torch.inference_mode():
            emission, _ = model(waveform)
        return emission
    except torch.cuda.OutOfMemoryError:
        sys.stderr.write(
            "[WARN] CUDA out of memory for emission computation, "
            "falling back to CPU.\n"
        )
        model_cpu = model.cpu()
        waveform_cpu = waveform.cpu()
        with torch.inference_mode():
            emission, _ = model_cpu(waveform_cpu)
        model.to(device)
        return emission


def _torchaudio_align(
    audio_path: str,
    sentences: list[str],
    lang: str,
) -> list[dict]:
    """Perform forced alignment using torchaudio MMS_FA.

    Args:
        audio_path: Path to audio file.
        sentences: List of sentence text strings.
        lang: ISO 639-3 language code.

    Returns:
        List of dicts with ``"start"`` and ``"end"`` keys (seconds),
        one per input sentence.
    """
    import torch  # noqa: F401

    model, tokenizer, aligner, device, sample_rate, dictionary = (
        _get_torchaudio_components()
    )

    waveform = _load_audio(audio_path, sample_rate, device)
    emission = _compute_emissions(model, waveform, device)
    num_frames = emission.shape[1]

    # Frame-to-time conversion factor
    ratio = waveform.shape[1] / num_frames

    # Normalize and tokenize all sentences as a flat word list, tracking sentence boundaries
    all_words: list[str] = []
    sentence_word_counts: list[int] = []

    for sent_text in sentences:
        normalized = _normalize_text_for_alignment(sent_text)
        words = normalized.split()

        # Filter out words containing characters not in the dictionary
        valid_words = [
            w for w in words if all(c in dictionary for c in w)
        ]

        all_words.extend(valid_words)
        sentence_word_counts.append(len(valid_words))

    if not all_words:
        return [{"start": 0.0, "end": 0.0} for _ in sentences]

    tokens = tokenizer(all_words)
    token_spans = aligner(emission[0], tokens)

    # Aggregate token-level spans to sentence-level timestamps
    results: list[dict] = []
    word_idx = 0
    for word_count in sentence_word_counts:
        if word_count == 0:
            prev_end = results[-1]["end"] if results else 0.0
            results.append({"start": prev_end, "end": prev_end})
        else:
            sent_spans = token_spans[word_idx : word_idx + word_count]
            first_span = sent_spans[0][0]
            last_span = sent_spans[-1][-1]

            start_time = float(first_span.start * ratio / sample_rate)
            end_time = float(last_span.end * ratio / sample_rate)

            results.append({"start": start_time, "end": end_time})
        word_idx += word_count

    return results


def _is_punct_only(text: str) -> bool:
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    return re.search(r"\w", stripped) is None


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
    """Align sentence timestamps against an audio file using torchaudio MMS_FA.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.).
        sentences: List of dicts, each with at least a ``"text"`` key.
        lang: ISO 639-3 language code (e.g. ``"eng"``).

    Returns:
        Updated sentence dicts with ``start``/``end`` timestamps, or
        ``None`` if alignment could not be completed.
    """
    if not sentences:
        return sentences

    text_lines = [s.get("text", "").replace("\n", " ") for s in sentences]

    try:
        segments = _torchaudio_align(audio_path, text_lines, lang)
    except Exception as e:
        sys.stderr.write(
            f"[WARN] Alignment failed for {audio_path}: {e}\n"
            f"{traceback.format_exc()}\n"
        )
        return None

    return update_timestamps_from_alignment(sentences, segments)


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
