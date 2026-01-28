# chunkseg

A lightweight Python package for evaluating **segmentation quality** (also called **chaptering**) in audio and video content. Chunkseg evaluates in the time space rather text space and thus is transcript-invariant, enabling comparability between a broad set of different models. The package converts segment boundaries into **fixed-size time chunks** and computes **established and comprehensive metrics** including binary classification scores (precision, recall, F1) and segmentation-specific measures (Pk, WindowDiff, Boundary Similarity, GHD).

The package supports **direct timestamp evaluation** or **automatic boundary extraction from structured transcripts via forced alignment**.

## Install

```bash
pip install chunkseg
```

For transcript alignment support (requires `alqalign`):

```bash
pip install chunkseg[align]
```

## Evaluation Modes

Chunkseg supports three evaluation modes depending on your input format:

### 1. Timestamps Mode
Provide boundary timestamps directly as lists of floats. Use when you already have predicted timestamps (e.g., from a timestamps-only model). **No audio or transcript needed.**

### 2. Structured Transcript (Forced Alignment)
Provide a structured transcript without timestamps (e.g., `[CSTART] Title [CEND] text...`). Use when your model produces chapter structure but no timestamps. **Requires audio file and `alqalign`.**

**How it works:** Parse transcript → sentence-tokenize → align to audio → derive boundary timestamps from aligned sections.

### 3. Structured Transcript with Timestamps
Provide a structured transcript with embedded timestamps (e.g., `[CSTART] 1:23:45 - Title [CEND] text...`). Use when your model produces both structure and timestamps.

**Two modes:**
- **Use provided timestamps** (default): Fast, no alignment needed, no audio required
- **Force alignment** (`--force-alignment`): Ignores timestamps, uses audio alignment instead (requires audio + `alqalign`)

## Quick Reference

| Mode | Format | Audio Required? | `force_alignment` | Example |
|------|--------|-----------------|-------------------|---------|
| **Timestamps** | `list[float]` | No | N/A | `hypothesis=[0.0, 125.0]` |
| **Transcript (FA)** | `cstart`, `markdown`, etc. | Yes | N/A (always aligns) | `format="cstart", audio="..."` |
| **Transcript + Timestamps (use provided)** | `cstart_ts`, `markdown_ts`, etc. | No | `False` (default) | `format="cstart_ts"` |
| **Transcript + Timestamps (force FA)** | `cstart_ts`, `markdown_ts`, etc. | Yes | `True` | `format="cstart_ts", force_alignment=True` |

## Usage

### Python API

```python
from chunkseg import evaluate, evaluate_batch, print_results

# Mode 1: Timestamps mode
result = evaluate(
    hypothesis=[0.0, 120.5, 300.0],
    reference=[0.0, 125.0, 310.0],
    duration=600.0,
    chunk_size=6.0,
)

# Mode 2: Structured transcript (forced alignment)
result = evaluate(
    hypothesis="[CSTART] Intro [CEND] text... [CSTART] Main [CEND] more...",
    reference=[0.0, 125.0],
    audio="/path/to/audio.wav",
    duration=600.0,
    format="cstart",
    lang="eng",
)

# Mode 3a: Transcript with timestamps (use provided)
result = evaluate(
    hypothesis="[CSTART] 0:00:00 - Intro [CEND] text... [CSTART] 2:05:30 - Main [CEND]...",
    reference=[0.0, 125.0],
    duration=600.0,
    format="cstart_ts",  # timestamps embedded, use them
)

# Mode 3b: Transcript with timestamps (force alignment)
result = evaluate(
    hypothesis="[CSTART] 0:00:00 - Intro [CEND] text... [CSTART] 2:05:30 - Main [CEND]...",
    reference=[0.0, 125.0],
    audio="/path/to/audio.wav",
    duration=600.0,
    format="cstart_ts",
    force_alignment=True,  # ignore timestamps, use alignment
    lang="eng",
)

# Batch evaluation
results = evaluate_batch(
    samples=[
        {"hypothesis": [0.0, 120.5], "reference": [0.0, 125.0], "duration": 600.0},
        {"hypothesis": [0.0, 300.0], "reference": [0.0, 310.0], "duration": 500.0},
    ],
    chunk_size=6.0,
)
print_results(results)
```

### CLI

```bash
# Mode 1: Timestamps mode
chunkseg timestamps.jsonl --chunk-size 6.0

# Mode 2: Structured transcript (forced alignment)
chunkseg transcripts.jsonl --chunk-size 6.0 --format cstart --lang eng

# Mode 3a: Transcript with timestamps (use provided)
chunkseg transcripts_with_ts.jsonl --chunk-size 6.0 --format cstart_ts

# Mode 3b: Transcript with timestamps (force alignment, ignore timestamps)
chunkseg transcripts_with_ts.jsonl --chunk-size 6.0 --format cstart_ts --force-alignment --lang eng

# Output to JSON file
chunkseg samples.jsonl --chunk-size 6.0 --output results.json
```

## Input Formats

### Mode 1: Timestamps Mode

```json
{"hypothesis": [0.0, 120.5, 300.0], "reference": [0.0, 125.0, 310.0], "duration": 600.0}
```

**Fields:**
- `hypothesis`: List of predicted boundary timestamps (seconds)
- `reference`: List of ground truth boundary timestamps (seconds)
- `duration`: Total audio duration (seconds)

### Mode 2: Structured Transcript (Forced Alignment)

```json
{"hypothesis": "[CSTART] Intro [CEND] text...", "reference": [0.0, 125.0], "audio": "/path/to/audio.wav", "duration": 600.0}
```

**Fields:**
- `hypothesis`: Structured transcript string (no timestamps)
- `reference`: Ground truth boundary timestamps
- `audio`: Full path to audio file (required for alignment)
- `duration`: Total audio duration

**Note:** Requires format like `cstart`, `newline`, `markdown`, or `custom` (no `_ts` suffix).

### Mode 3: Transcript with Timestamps

```json
{"hypothesis": "[CSTART] 0:00:00 - Intro [CEND] text... [CSTART] 2:05:30 - Main [CEND]...", "reference": [0.0, 125.0], "duration": 600.0}
```

**Fields:**
- `hypothesis`: Structured transcript with embedded timestamps
- `reference`: Ground truth boundary timestamps
- `duration`: Total audio duration
- `audio`: (optional) Full path to audio - only needed if using `--force-alignment`

**Format:** Use `cstart_ts`, `markdown_ts`, or `custom_ts` (with `_ts` suffix).

**Behavior:**
- Default: Uses timestamps from the text
- With `--force-alignment`: Ignores timestamps, requires `audio` field

## Parser Presets

| Preset | Description | Example |
|--------|-------------|---------|
| `cstart` | `[CSTART] Title [CEND]` tags | `[CSTART] Intro [CEND] text...` |
| `cstart_ts` | Tags with timestamps | `[CSTART] 1:23:45 - Title [CEND] text...` |
| `newline` | Double newline separator | `Chapter 1 text.\n\nChapter 2 text.` |
| `markdown` | Markdown headers | `# Title\nText\n## Subtitle\nMore text` |
| `markdown_ts` | Headers with timestamps | `# 0:15:30 - Introduction\nText...` |
| `custom` | User-provided regex | Split on any regex pattern |
| `custom_ts` | Regex with timestamp capture | Named groups: `(?P<timestamp>...)`, `(?P<title>...)` |

### Timestamp Formats for `markdown_ts`

The markdown timestamped parser supports multiple header formats:

- `# 00:15:30 - Title` (timestamp first with separator)
- `# 00:15:30 Title` (timestamp first, no separator)
- `# Title @ 00:15:30` (timestamp last with @)
- `## 1:23:45 - Chapter Title` (any header level)

### Custom Pattern with Timestamps (`custom_ts`)

For `custom_ts`, provide a regex pattern with named groups:

```python
from chunkseg import evaluate

result = evaluate(
    hypothesis="[ Intro @ 1:23:45 ] text... [ Main @ 2:00:00 ] more...",
    reference=[0.0, 125.0],
    duration=600.0,
    format="custom_ts",
    custom_pattern=r"\[\s*(?P<title>[^@]+?)\s*@\s*(?P<timestamp>\d+:\d{2}:\d{2})\s*\]",
    timestamp_format="H:MM:SS",
)
```

Supported `timestamp_format` values:
- `"HH:MM:SS"` — 2-digit hours (e.g., `01:23:45`)
- `"H:MM:SS"` — variable hours (e.g., `1:23:45`) — **default**
- `"MM:SS"` — 2-digit minutes (e.g., `23:45`)
- `"M:SS"` — variable minutes (e.g., `3:45`)
- Custom regex with named groups: `(?P<h>...)`, `(?P<m>...)`, `(?P<s>...)`

## Metrics

### Binary Classification

| Metric | Description |
|--------|-------------|
| F1 | Harmonic mean of precision and recall |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Accuracy | (TP + TN) / Total |
| Specificity | TN / (TN + FP) |

### Segmentation-Specific

| Metric | Description |
|--------|-------------|
| Pk | Beeferman's Pk metric (via segeval) |
| WindowDiff | Window difference metric (via segeval) |
| Boundary Similarity | Boundary similarity score (via segeval) |
| GHD | Generalized Hamming Distance (via nltk) |

### Statistics

- Bootstrap confidence intervals for all metrics
- Standard deviation
- Prediction and reference segment counts

## Dependencies

**Required:**
- `numpy`
- `segeval`
- `nltk`

**Optional:**
- `alqalign` — for transcript alignment (install with `pip install chunkseg[align]`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows you to use, modify, and distribute this software for any purpose, including commercial applications, with minimal restrictions.
