"""BERTScore-based title evaluation for segmentation.

Two scoring modes:

* **TM-BS** (Temporally Matched BERTScore) — match hypothesis and reference
  titles by start time within a tolerance window, compute BERTScore on matched
  pairs only.
* **GC-BS** (Global Concatenation BERTScore) — concatenate all titles with
  newlines and compute a single BERTScore.
"""

from __future__ import annotations

_bert_scorer = None
_bert_scorer_lang = None
_rouge_scorer = None


def _get_bert_scorer(lang: str = "en"):
    """Return a cached BERTScorer instance, re-created if lang changes.

    Device selection honors ``CHUNKSEG_BERTSCORE_DEVICE`` (e.g. ``cuda:1``);
    otherwise falls back to ``cuda`` if available, else ``cpu``.
    """
    global _bert_scorer, _bert_scorer_lang
    if _bert_scorer is None or _bert_scorer_lang != lang:
        import os
        import torch
        from bert_score import BERTScorer

        device = os.environ.get("CHUNKSEG_BERTSCORE_DEVICE")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _bert_scorer = BERTScorer(lang=lang, device=device)
        _bert_scorer_lang = lang
    return _bert_scorer


def _get_rouge_scorer():
    """Return a cached RougeScorer instance (loaded once)."""
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer


def _rouge_l_scores(hyps: list[str], refs: list[str]) -> tuple[float, float, float]:
    """Compute mean ROUGE-L P/R/F over paired lists."""
    scorer = _get_rouge_scorer()
    p_sum = r_sum = f_sum = 0.0
    for h, r in zip(hyps, refs):
        s = scorer.score(r, h)["rougeL"]
        p_sum += s.precision
        r_sum += s.recall
        f_sum += s.fmeasure
    n = len(hyps)
    return p_sum / n, r_sum / n, f_sum / n


def _match_titles_by_time(
    hyp_titles: list[tuple[str, float]],
    ref_titles: list[tuple[str, float]],
    tolerance: float,
    duration: float,
) -> tuple[list[str], list[str]]:
    """Greedy 1-to-1 matching of chapters where both start and end agree within tolerance.

    Each chapter's end time is the next chapter's start time, or *duration* for
    the last chapter. A reference chapter (ts, te) is matched to a hypothesis
    chapter (ts_hat, te_hat) if |ts_hat - ts| <= tolerance AND
    |te_hat - te| <= tolerance.

    Returns:
        Tuple of (matched_hyp_texts, matched_ref_texts).
    """
    def _with_ends(titles):
        starts = [s for _, s in titles]
        ends = starts[1:] + [duration]
        return [(t, s, e) for (t, s), e in zip(titles, ends)]

    hyp_intervals = _with_ends(hyp_titles)
    ref_intervals = _with_ends(ref_titles)

    used_hyp: set[int] = set()
    matched_hyp: list[str] = []
    matched_ref: list[str] = []

    for ref_title, ref_start, ref_end in ref_intervals:
        best_idx: int | None = None
        best_dist = float("inf")
        for i, (_, hyp_start, hyp_end) in enumerate(hyp_intervals):
            if i in used_hyp:
                continue
            ds = abs(hyp_start - ref_start)
            de = abs(hyp_end - ref_end)
            if ds <= tolerance and de <= tolerance and (ds + de) < best_dist:
                best_idx = i
                best_dist = ds + de
        if best_idx is not None:
            matched_hyp.append(hyp_intervals[best_idx][0])
            matched_ref.append(ref_title)
            used_hyp.add(best_idx)

    return matched_hyp, matched_ref


def compute_title_scores(
    hyp_titles: list[tuple[str, float]],
    ref_titles: list[tuple[str, float]],
    duration: float,
    tolerance: float = 5.0,
    lang: str = "en",
) -> dict:
    """Compute BERTScore title metrics in TM-BS and GC-BS modes.

    Args:
        hyp_titles: Hypothesis titles as ``(title_text, start_seconds)`` pairs.
        ref_titles: Reference titles as ``(title_text, start_seconds)`` pairs.
        duration: Total duration in seconds; used to compute the end of the
            last chapter for TM matching.
        tolerance: Maximum time difference in seconds for TM matching on both
            start and end boundaries (default 5.0).
        lang: Language code for BERTScore (default ``"en"``).

    Returns:
        Dict with keys ``tm_bs_precision``, ``tm_bs_recall``, ``tm_bs_f1``,
        ``tm_bs_matched``, ``gc_bs_precision``, ``gc_bs_recall``, ``gc_bs_f1``.
    """
    bert = _get_bert_scorer(lang)
    result: dict = {}

    # --- Temporally Matched ---
    matched_hyp, matched_ref = _match_titles_by_time(
        hyp_titles, ref_titles, tolerance, duration,
    )
    if matched_hyp and matched_ref:
        # BERTScore
        P, R, F1 = bert.score(matched_hyp, matched_ref)
        result["tm_bs_precision"] = P.mean().item()
        result["tm_bs_recall"] = R.mean().item()
        result["tm_bs_f1"] = F1.mean().item()
        # ROUGE-L
        rp, rr, rf = _rouge_l_scores(matched_hyp, matched_ref)
        result["tm_rl_precision"] = rp
        result["tm_rl_recall"] = rr
        result["tm_rl_f1"] = rf
    result["tm_matched"] = len(matched_hyp) / len(ref_titles) if ref_titles else 0.0

    # --- Global Concatenation ---
    hyp_concat = "\n".join(t for t, _ in hyp_titles)
    ref_concat = "\n".join(t for t, _ in ref_titles)
    if hyp_concat and ref_concat:
        
        # BERTScore
        P, R, F1 = bert.score([hyp_concat], [ref_concat])
        result["gc_bs_precision"] = P.item()
        result["gc_bs_recall"] = R.item()
        result["gc_bs_f1"] = F1.item()

        # ROUGE-L
        rp, rr, rf = _rouge_l_scores([hyp_concat], [ref_concat])
        result["gc_rl_precision"] = rp
        result["gc_rl_recall"] = rr
        result["gc_rl_f1"] = rf

    return result
