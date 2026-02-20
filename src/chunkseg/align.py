"""Alignment wrapper for deriving timestamps from transcript text.

Supports two backends:
- ``"mms_fa"`` (default): torchaudio MMS_FA — no extra install required beyond
  ``chunkseg[align]``. Includes exponential trimming: the acoustic model runs
  once; only the cheap CTC alignment step is retried with fewer sentences.
- ``"alqalign"``: alqalign — see https://github.com/xinjli/alqalign for
  installation. Includes exponential trimming via re-running full alignment.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback


# ── MMS_FA (torchaudio) ───────────────────────────────────────────────────────

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
            "[WARN] CUDA out of memory for emission computation — "
            "audio too long for GPU. Skipping alignment.\n"
        )
        raise


def _align_text_to_emission(
    emission,
    waveform,
    text_lines: list[str],
    sample_rate: int,
    dictionary,
    tokenizer,
    aligner,
) -> list[dict]:
    """Align text lines against a precomputed emission."""
    num_frames = emission.shape[1]
    ratio = waveform.shape[1] / num_frames

    all_words: list[str] = []
    sentence_word_counts: list[int] = []

    for sent_text in text_lines:
        normalized = _normalize_text_for_alignment(sent_text)
        words = normalized.split()
        valid_words = [w for w in words if all(c in dictionary for c in w)]
        all_words.extend(valid_words)
        sentence_word_counts.append(len(valid_words))

    if not all_words:
        return [{"start": 0.0, "end": 0.0} for _ in text_lines]

    tokens = tokenizer(all_words)
    token_spans = aligner(emission[0], tokens)  # raises RuntimeError on CTC failure

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


def _align_sentences_mms_fa(
    audio_path: str,
    sentences: list[dict],
    lang: str,
) -> list[dict] | None:
    """Align using MMS_FA with exponential trimming for long transcripts.

    The acoustic model (audio → emission) runs only once. On CTC failure the
    cheap alignment step is retried with sentences trimmed from the tail.
    Trimmed sentences receive the last successfully aligned timestamp.
    """
    model, tokenizer, aligner, device, sample_rate, dictionary = (
        _get_torchaudio_components()
    )

    try:
        waveform = _load_audio(audio_path, sample_rate, device)
        emission = _compute_emissions(model, waveform, device)
    except Exception as e:
        sys.stderr.write(
            f"[WARN] Alignment failed for {audio_path}: {e}\n"
            f"{traceback.format_exc()}\n"
        )
        return None

    n = len(sentences)
    exp_step = 1
    max_step = 128
    first_trim = True

    while n > 0:
        text_lines = [s.get("text", "").replace("\n", " ") for s in sentences[:n]]
        try:
            segments = _align_text_to_emission(
                emission, waveform, text_lines,
                sample_rate, dictionary, tokenizer, aligner,
            )
        except RuntimeError as e:
            if "targets length is too long" not in str(e):
                sys.stderr.write(
                    f"[WARN] Alignment failed for {audio_path}: {e}\n"
                    f"{traceback.format_exc()}\n"
                )
                return None
            if n <= 1:
                sys.stderr.write(
                    f"[WARN] Alignment failed for {audio_path}: "
                    "cannot trim further (only 1 sentence left).\n"
                )
                return None
            if first_trim:
                n -= 1
                first_trim = False
            else:
                step = min(exp_step, max_step)
                n -= min(step, n - 1)
                exp_step = min(step * 2, max_step)
            continue

        # Alignment succeeded for sentences[:n]. Assign last timestamp to trimmed tail.
        aligned = update_timestamps_from_alignment(sentences[:n], segments)
        if n < len(sentences):
            last_time = aligned[-1]["end"]
            for s in sentences[n:]:
                s2 = dict(s)
                s2["start"] = last_time
                s2["end"] = last_time
                aligned.append(s2)
            sys.stderr.write(
                f"[INFO] MMS_FA trimmed {len(sentences) - n} sentence(s) from the end "
                f"of {os.path.basename(audio_path)} due to CTC length constraint.\n"
            )
        return aligned

    return None


# ── alqalign ─────────────────────────────────────────────────────────────────

def _get_alq_align():
    try:
        from alqalign.app import align as alq_align
        return alq_align
    except ImportError:
        raise ImportError(
            "alqalign is not installed. See https://github.com/xinjli/alqalign "
            "for installation instructions."
        )


def _write_temp_text(lines: list[str]) -> str:
    """Write text lines to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="chunkseg_", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for line in lines:
            f.write((line or "").strip() + "\n")
    return path


def _align_sentences_alqalign(
    audio_path: str,
    sentences: list[dict],
    lang: str,
) -> list[dict] | None:
    """Align using alqalign with exponential trimming for long transcripts."""
    alq_align = _get_alq_align()

    remaining = list(sentences)
    used_trim_last = False
    exp_step = 1
    max_step = 128

    while True:
        tmp_txt_path = None
        try:
            text_lines = [s.get("text", "").replace("\n", " ") for s in remaining]
            tmp_txt_path = _write_temp_text(text_lines)
            segments = alq_align(audio_path, tmp_txt_path, lang)

            if not isinstance(segments, list) or not all(
                isinstance(x, dict) for x in segments
            ):
                sys.stderr.write(
                    "[WARN] Unexpected alqalign output; keeping original timestamps.\n"
                )
                return None

            if len(segments) != len(remaining):
                punct_positions = [
                    i for i, s in enumerate(remaining)
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
                        if aligned_kept else 0.0
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
                remaining = remaining[:-1]
                if remaining:
                    pivot_text = remaining[-1].get("text", "")
                    remaining = [
                        s for s in remaining[:-1]
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


# ── shared helpers ────────────────────────────────────────────────────────────

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


# ── public API ────────────────────────────────────────────────────────────────

def align_sentences(
    audio_path: str,
    sentences: list[dict],
    lang: str,
    backend: str = "mms_fa",
) -> list[dict] | None:
    """Align sentence timestamps against an audio file.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.).
        sentences: List of dicts, each with at least a ``"text"`` key.
            Sentences are typically produced by ``nltk.sent_tokenize`` via
            the chunkseg parsers.
        lang: ISO 639-3 language code (e.g. ``"eng"``).
        backend: Alignment backend. Options:

            - ``"mms_fa"`` (default) — torchaudio MMS_FA; installed via
              ``chunkseg[align]``. The acoustic model (audio → emission) runs
              once; the CTC alignment step is retried with exponential trimming
              if the transcript is too long for a single pass.
            - ``"alqalign"`` — alqalign; see https://github.com/xinjli/alqalign
              for installation. Full alignment (including audio processing) is
              retried with exponential trimming on failure.

    Returns:
        Updated sentence dicts with ``start``/``end`` timestamps, or
        ``None`` if alignment could not be completed.
    """
    if not sentences:
        return sentences

    if backend == "alqalign":
        return _align_sentences_alqalign(audio_path, sentences, lang)

    if backend != "mms_fa":
        raise ValueError(
            f"Unknown FA backend: {backend!r}. Choose 'mms_fa' or 'alqalign'."
        )

    return _align_sentences_mms_fa(audio_path, sentences, lang)


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
