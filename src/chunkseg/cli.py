"""Command-line interface for chunkseg."""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chunkseg",
        description="Evaluate time-chunked segmentation quality from a JSONL file.",
    )
    p.add_argument(
        "input",
        help="Path to JSONL input file. Each line must be a JSON object with "
        '"hypothesis", "reference", and "duration" keys.',
    )
    p.add_argument(
        "--chunk-size",
        type=float,
        default=6.0,
        help="Chunk size in seconds (default: 6.0).",
    )
    p.add_argument(
        "--format",
        default=None,
        help='Parser preset for transcript mode: "cstart", "cstart_ts", '
        '"newline", "markdown", "markdown_ts", "custom", or "custom_ts".',
    )
    p.add_argument(
        "--custom-pattern",
        default=None,
        help='Regex pattern for custom/custom_ts format. For custom_ts, use '
        'named groups: (?P<timestamp>...) and optionally (?P<title>...).',
    )
    p.add_argument(
        "--timestamp-format",
        default=None,
        help='Timestamp format for custom_ts: "HH:MM:SS", "H:MM:SS", "MM:SS", '
        '"M:SS", or a custom regex with named groups h/m/s.',
    )
    p.add_argument(
        "--lang",
        default="eng",
        help="alqalign language code (default: eng).",
    )
    p.add_argument(
        "--force-alignment",
        action="store_true",
        help="Force alignment even when timestamps are embedded in transcript. "
        "Ignores provided timestamps and derives them from audio alignment instead.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write aggregated results to a JSON file instead of stdout.",
    )
    p.add_argument(
        "--num-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations for confidence intervals (default: 100).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load JSONL
    samples: list[dict] = []
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: skipping invalid JSON on line {lineno}: {e}",
                        file=sys.stderr,
                    )
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not samples:
        print("No samples found in input file.", file=sys.stderr)
        sys.exit(1)

    from .evaluate import evaluate_batch
    from .display import print_results

    results = evaluate_batch(
        samples,
        chunk_size=args.chunk_size,
        format=args.format,
        custom_pattern=args.custom_pattern,
        timestamp_format=args.timestamp_format,
        lang=args.lang,
        num_bootstrap=args.num_bootstrap,
        force_alignment=args.force_alignment,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print_results(results)


if __name__ == "__main__":
    main()
