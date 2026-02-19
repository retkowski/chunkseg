# chunkseg

A Python package for **comprehensive evaluation of segmentation (chaptering) quality** in audio and video content. Chunkseg implements three complementary evaluation protocols introduced in *Beyond Transcripts: A Renewed Perspective on Audio Chaptering* ([arXiv:2602.08979](https://arxiv.org/abs/2602.08979)):

1. **Discretized time evaluation** — Convert boundaries to fixed-size time chunks and apply established text segmentation metrics (Pk, WindowDiff, Boundary Similarity, GHD) plus binary classification metrics (F1, precision, recall).
2. **Continuous time evaluation** — Collar-based boundary F1 that matches predicted and reference boundaries within a time tolerance window (default ±3 s).
3. **Title evaluation** — BERTScore- and ROUGE-L-based comparison of chapter titles in two modes: *Temporally Matched* (TM) and *Global Concatenation* (GC).

By evaluating in the time domain rather than the text domain, chunkseg is **transcript-invariant** and enables comparisons across models that produce very different output formats.

## Install

```bash
pip install chunkseg
```

For **title evaluation** (BERTScore + ROUGE-L):

```bash
pip install "chunkseg[titles]"
```

For **forced alignment** from structured transcripts without timestamps:

```bash
pip install "chunkseg[align]"
```

## Evaluation Protocols

### 1. Discretized Time (Time-Chunks)

Segment boundaries are projected onto a sequence of fixed-size binary time chunks. A chunk is labelled `1` if a boundary falls within it, `0` otherwise. Standard binary classification metrics (precision, recall, F1, accuracy, specificity) and established segmentation metrics (Pk, WindowDiff, Boundary Similarity, GHD) are then computed over this sequence.

This approach is compatible with any model that produces boundary timestamps, regardless of transcript format.

### 2. Continuous Time (Collar-Based F1)

Predicted and reference boundary timestamps are compared directly in continuous time. A predicted boundary counts as a true positive if it falls within ±`collar` seconds of a reference boundary, using greedy closest-first 1-to-1 matching. Returns `collar_precision`, `collar_recall`, and `collar_f1`.

This metric is **always computed** alongside the time-chunk metrics whenever timestamps are available.

### 3. Title Evaluation

Chapter title quality is measured in two modes, both supporting BERTScore (BS) and ROUGE-L (RL):

| Mode | Description |
|------|-------------|
| **TM-BS / TM-RL** | *Temporally Matched* — pair hyp/ref titles by start time within a tolerance window, score matched pairs only |
| **GC-BS / GC-RL** | *Global Concatenation* — join all titles with `\n`, compute a single score on the two concatenated strings |
| **tm_matched** | Fraction of reference titles that were matched (0–1) |

Requires `reference_titles` (and optionally `hyp_titles`) in the input. Enable with `--titles` in the CLI or `titles=True` in `evaluate_batch()`.

## Usage

### Python API

```python
from chunkseg import evaluate, evaluate_batch, print_results

# Timestamps mode — no transcript or audio needed
result = evaluate(
    hypothesis=[120.5, 300.0],
    reference=[125.0, 310.0],
    duration=600.0,
    chunk_size=6.0,
    collar=3.0,
)

# Batch evaluation with aggregated metrics and bootstrap CIs
results = evaluate_batch(
    samples=[
        {"hypothesis": [120.5], "reference": [125.0], "duration": 600.0},
        {"hypothesis": [300.0], "reference": [310.0], "duration": 500.0},
    ],
    chunk_size=6.0,
    collar=3.0,
)
print_results(results)

# Title evaluation (requires chunkseg[titles])
result = evaluate(
    hypothesis=[24.2, 33.94],
    reference=[11.0, 23.0, 34.0],
    duration=50.0,
    hyp_titles=[("Set a background", 24.2), ("Clip the background", 33.94)],
    reference_titles=[("Wrap text with a span", 11.0),
                      ("Add a background", 23.0),
                      ("Clip background to text", 34.0)],
    tolerance=5.0,
)

# Standalone title scoring
from chunkseg import compute_title_scores

scores = compute_title_scores(
    hyp_titles=[("Introduction", 0.0), ("Methods", 60.0)],
    ref_titles=[("Intro", 0.0), ("Methodology", 58.0)],
    tolerance=5.0,
)
# Returns: tm_bs_f1, tm_rl_f1, gc_bs_f1, gc_rl_f1, tm_matched, ...
```

### CLI

```bash
# Basic timestamps mode
chunkseg samples.jsonl

# With title evaluation and custom collar
chunkseg samples.jsonl --titles --tolerance 5.0 --collar 3.0

# With WER (requires reference_transcript field)
chunkseg samples.jsonl --wer

# Structured transcript mode (forced alignment)
chunkseg transcripts.jsonl --format cstart --lang eng

# Structured transcript with embedded timestamps
chunkseg transcripts.jsonl --format cstart_ts

# Save results to JSON
chunkseg samples.jsonl --titles --output results.json
```

## Input Format

Each line of the input JSONL file must be a JSON object. The required fields depend on the evaluation mode.

### Timestamps mode (minimal)

```json
{"hypothesis": [24.2, 33.94], "reference": [11.0, 23.0, 34.0], "duration": 50.0}
```

### With title evaluation

```json
{
  "hypothesis": [24.2, 33.94],
  "reference": [11.0, 23.0, 34.0],
  "duration": 50.0,
  "reference_titles": [["Wrap text with a span", 11.0],
                       ["Add a background", 23.0],
                       ["Clip background to text", 34.0]],
  "hyp_titles": [["Set a background", 24.2],
                 ["Clip the background", 33.94]]
}
```

### Structured transcript (forced alignment)

```json
{
  "hypothesis": "[CSTART] Intro [CEND] text... [CSTART] Main [CEND] more...",
  "reference": [125.0],
  "audio": "/path/to/audio.wav",
  "duration": 600.0
}
```

### All fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hypothesis` | `list[float]` or `str` | Yes | Predicted boundaries (seconds) or structured transcript |
| `reference` | `list[float]` | Yes | Ground-truth boundary timestamps (seconds) |
| `duration` | `float` | Yes | Total duration (seconds) |
| `audio` | `str` | For alignment | Path to audio file |
| `reference_titles` | `[[title, seconds], ...]` | For `--titles` | Ground-truth chapter titles with start times |
| `hyp_titles` | `[[title, seconds], ...]` | Optional | Predicted titles (inferred from transcript if omitted) |
| `reference_transcript` | `str` | For `--wer` | Reference transcript text |

## Input Modes

| Mode | `hypothesis` type | Audio? | Notes |
|------|------------------|--------|-------|
| Timestamps | `list[float]` | No | Direct boundary seconds |
| Transcript (alignment) | `str` | Yes | `format="cstart"` etc., requires `chunkseg[align]` |
| Transcript + timestamps (use provided) | `str` | No | `format="cstart_ts"` etc. |
| Transcript + timestamps (force alignment) | `str` | Yes | `format="cstart_ts"`, `force_alignment=True` |

## Parser Presets

| Preset | Format | Timestamps? |
|--------|--------|-------------|
| `cstart` | `[CSTART] Title [CEND] text...` | No |
| `cstart_ts` | `[CSTART] 1:23:45 - Title [CEND] text...` | Yes |
| `newline` | Sections separated by blank lines | No |
| `markdown` | `# Title\nText\n## Subtitle...` | No |
| `markdown_ts` | `# 0:15:30 - Introduction\nText...` | Yes |
| `custom` | User-provided regex | No |
| `custom_ts` | User-provided regex with `(?P<timestamp>...)` | Yes |

## Metrics

### Discretized Time (Time-Chunks)

| Metric | Description |
|--------|-------------|
| `f1` | Harmonic mean of precision and recall (derived from aggregated P/R) |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `accuracy` | (TP + TN) / Total |
| `specificity` | TN / (TN + FP) |
| `pk` | Beeferman's Pk (lower is better) |
| `window_diff` | WindowDiff (lower is better) |
| `boundary_similarity` | Boundary similarity score |
| `ghd` | Generalized Hamming Distance |

### Continuous Time (Collar-Based)

| Metric | Description |
|--------|-------------|
| `collar_f1` | F1 within ±collar seconds (default ±3 s) |
| `collar_precision` | Precision within collar |
| `collar_recall` | Recall within collar |

### Title Evaluation

| Metric | Description |
|--------|-------------|
| `tm_bs_f1` | BERTScore F1 on temporally matched title pairs |
| `tm_bs_precision` | BERTScore precision on matched pairs |
| `tm_bs_recall` | BERTScore recall on matched pairs |
| `tm_rl_f1` | ROUGE-L F1 on temporally matched title pairs |
| `tm_rl_precision` | ROUGE-L precision on matched pairs |
| `tm_rl_recall` | ROUGE-L recall on matched pairs |
| `tm_matched` | Fraction of reference titles matched (0–1) |
| `gc_bs_f1` | BERTScore F1 on globally concatenated titles |
| `gc_bs_precision` | BERTScore precision on concatenated titles |
| `gc_bs_recall` | BERTScore recall on concatenated titles |
| `gc_rl_f1` | ROUGE-L F1 on globally concatenated titles |
| `gc_rl_precision` | ROUGE-L precision on concatenated titles |
| `gc_rl_recall` | ROUGE-L recall on concatenated titles |

All scalar metrics are reported as `{mean, std, ci_lower, ci_upper}` with bootstrap confidence intervals (default 100 iterations, configurable via `--num-bootstrap`).

## CLI Reference

```
chunkseg <input.jsonl> [options]

Options:
  --chunk-size FLOAT    Chunk size in seconds (default: 6.0)
  --collar FLOAT        Collar size for boundary F1 (default: 3.0)
  --format STR          Parser preset for transcript mode
  --custom-pattern STR  Regex for custom/custom_ts format
  --timestamp-format STR Timestamp format for custom_ts
  --lang STR            ISO 639-3 language code for alignment (default: eng)
  --force-alignment     Derive timestamps from audio alignment
  --wer                 Compute WER (requires reference_transcript field)
  --titles              Compute title metrics (requires reference_titles field)
  --tolerance FLOAT     Time tolerance for TM matching in seconds (default: 5.0)
  --output FILE         Write results to JSON file
  --num-bootstrap INT   Bootstrap iterations for CIs (default: 100)
```

## Dependencies

**Required:**
- `numpy`
- `segeval`
- `nltk`
- `jiwer`

**Optional — title evaluation (`chunkseg[titles]`):**
- `bert-score`
- `rouge-score`

**Optional — transcript alignment (`chunkseg[align]`):**
- `torch >= 2.1.0`
- `torchaudio >= 2.1.0`

## Citation

If you use chunkseg in your research, please cite:

```bibtex
@article{retkowski2026beyond,
  title     = {Beyond Transcripts: A Renewed Perspective on Audio Chaptering},
  author    = {Retkowski, Fabian and Z{\"u}fle, Maike and Nguyen, Thai Binh and Niehues, Jan and Waibel, Alexander},
  journal   = {arXiv preprint arXiv:2602.08979},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.08979}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
