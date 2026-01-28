"""Transcript format parsers for segmentation evaluation.

Each parser splits raw text into sections, then sentence-tokenizes each
section with ``nltk.sent_tokenize``.  All parsers return
``list[list[str]]`` — a list of sections, each being a list of sentences.

For timestamped formats, parsers also return boundary timestamps.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from nltk.tokenize import sent_tokenize

_CSTART_PATTERN = re.compile(r"\[CSTART\]\s*(.*?)\s*\[CEND\]", re.DOTALL)

_TIMESTAMPED_CSTART_PATTERN = re.compile(
    r"\[CSTART\]\s*(\d{1,2}:\d{2}:\d{2})\s*-\s*(.*?)\s*\[CEND\]", re.DOTALL
)

_MARKDOWN_HEADER = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Markdown header with optional timestamp: # [timestamp] [separator] title
# Supports: "# 00:15:30 - Title", "# Title @ 1:23:45", "# 15:30 Title"
_MARKDOWN_TIMESTAMPED = re.compile(
    r"^(#{1,6})\s+"
    r"(?:"
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]?\s*(.+?)"  # timestamp first: # 00:15:30 - Title
    r"|"
    r"(.+?)\s*[@]\s*(\d{1,2}:\d{2}(?::\d{2})?)"  # timestamp last: # Title @ 00:15:30
    r")$",
    re.MULTILINE,
)

_TIMESTAMP_FORMATS = {
    "HH:MM:SS": re.compile(r"^(\d{2}):(\d{2}):(\d{2})$"),
    "H:MM:SS": re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$"),
    "MM:SS": re.compile(r"^(\d{2}):(\d{2})$"),
    "M:SS": re.compile(r"^(\d{1,2}):(\d{2})$"),
    "HHMMSS": re.compile(r"^(\d{2})(\d{2})(\d{2})$"),
    "MMSS": re.compile(r"^(\d{2})(\d{2})$"),
}


def _parse_timestamp_hms(ts_str: str) -> float | None:
    """Parse a timestamp string in various formats to seconds.

    Supported formats: HH:MM:SS, H:MM:SS, MM:SS, M:SS
    Returns None if parsing fails.
    """
    ts_str = ts_str.strip()
    parts = ts_str.split(":")

    if len(parts) == 3:
        # HH:MM:SS or H:MM:SS
        try:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
        except ValueError:
            return None
    elif len(parts) == 2:
        # MM:SS or M:SS
        try:
            m, s = int(parts[0]), int(parts[1])
            return m * 60 + s
        except ValueError:
            return None
    return None


def _parse_timestamp_with_format(ts_str: str, fmt: str) -> float | None:
    """Parse a timestamp string using a specific format.

    Args:
        ts_str: The timestamp string to parse.
        fmt: Format string (e.g., "HH:MM:SS", "MM:SS") or a regex pattern.

    Returns:
        Seconds as float, or None if parsing fails.
    """
    ts_str = ts_str.strip()

    # Check if it's a known format
    if fmt in _TIMESTAMP_FORMATS:
        pattern = _TIMESTAMP_FORMATS[fmt]
        match = pattern.match(ts_str)
        if not match:
            return None
        groups = match.groups()
        if len(groups) == 3:
            h, m, s = int(groups[0]), int(groups[1]), int(groups[2])
            return h * 3600 + m * 60 + s
        elif len(groups) == 2:
            m, s = int(groups[0]), int(groups[1])
            return m * 60 + s
        return None

    # Treat fmt as a custom regex with named groups: h, m, s (all optional)
    try:
        pattern = re.compile(fmt)
        match = pattern.match(ts_str)
        if not match:
            return None

        gd = match.groupdict()
        h = int(gd.get("h") or gd.get("hours") or 0)
        m = int(gd.get("m") or gd.get("minutes") or gd.get("min") or 0)
        s = int(gd.get("s") or gd.get("seconds") or gd.get("sec") or 0)
        return h * 3600 + m * 60 + s
    except (re.error, ValueError, TypeError):
        return None


def _tokenize_section(text: str) -> list[str]:
    """Sentence-tokenize a section of text."""
    text = text.strip()
    if not text:
        return []
    return sent_tokenize(text)


class ParseResult(NamedTuple):
    """Result from parsing a transcript with optional timestamps."""
    sections: list[list[str]]
    titles: list[str] | None = None
    timestamps: list[float] | None = None


def parse_cstart(text: str) -> tuple[list[str], list[list[str]]]:
    """Parse ``[CSTART] Title [CEND] body`` format.

    Returns:
        Tuple of (titles, sections) where *titles* is a list of section title
        strings and *sections* is a list of sentence lists.
    """
    if not isinstance(text, str):
        return [], []

    matches = list(_CSTART_PATTERN.finditer(text))
    if not matches:
        return [], []

    titles: list[str] = []
    sections: list[list[str]] = []

    for i, match in enumerate(matches):
        titles.append(match.group(1).strip())
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append(_tokenize_section(section_text))

    return titles, sections


def parse_cstart_timestamped(
    text: str,
) -> tuple[list[float], list[str], list[list[str]]]:
    """Parse ``[CSTART] HH:MM:SS - Title [CEND] body`` format.

    Returns:
        Tuple of (timestamps, titles, sections).
    """
    if not isinstance(text, str):
        return [], [], []

    matches = list(_TIMESTAMPED_CSTART_PATTERN.finditer(text))
    if not matches:
        return [], [], []

    timestamps: list[float] = []
    titles: list[str] = []
    sections: list[list[str]] = []

    for i, match in enumerate(matches):
        ts = _parse_timestamp_hms(match.group(1))
        if ts is not None:
            timestamps.append(ts)
        titles.append(match.group(2).strip())
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(_tokenize_section(text[start:end]))

    return timestamps, titles, sections


parse_timestamped = parse_cstart_timestamped


def parse_newline(text: str) -> list[list[str]]:
    """Split on double newlines and sentence-tokenize each section.

    Returns:
        List of sentence lists (one per section).
    """
    if not isinstance(text, str):
        return []
    raw_sections = re.split(r"\n\n+", text.strip())
    return [_tokenize_section(s) for s in raw_sections if s.strip()]


def parse_markdown(text: str) -> tuple[list[str], list[list[str]]]:
    """Split on markdown headers (``# ...``) and sentence-tokenize each section.

    Returns:
        Tuple of (titles, sections).
    """
    if not isinstance(text, str):
        return [], []

    matches = list(_MARKDOWN_HEADER.finditer(text))
    if not matches:
        # No headers found, treat entire text as one section
        sections = [_tokenize_section(text)] if text.strip() else []
        return [], sections

    titles: list[str] = []
    sections: list[list[str]] = []

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        titles.append(title)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append(_tokenize_section(section_text))

    return titles, sections


def parse_markdown_timestamped(
    text: str,
) -> tuple[list[float], list[str], list[list[str]]]:
    """Parse markdown with timestamps in headers.

    Supports formats like:
        - ``# 00:15:30 - Introduction``
        - ``# 00:15:30 Introduction``
        - ``# Introduction @ 00:15:30``
        - ``## 1:23:45 - Chapter Title``

    Returns:
        Tuple of (timestamps, titles, sections).
    """
    if not isinstance(text, str):
        return [], [], []

    matches = list(_MARKDOWN_TIMESTAMPED.finditer(text))
    if not matches:
        # Fall back to regular markdown parsing (no timestamps)
        return [], *parse_markdown(text)

    timestamps: list[float] = []
    titles: list[str] = []
    sections: list[list[str]] = []

    for i, match in enumerate(matches):
        # Groups: (level, ts_first, title_first, title_last, ts_last)
        ts_first, title_first = match.group(2), match.group(3)
        title_last, ts_last = match.group(4), match.group(5)

        if ts_first:
            ts = _parse_timestamp_hms(ts_first)
            title = title_first.strip() if title_first else ""
        else:
            ts = _parse_timestamp_hms(ts_last) if ts_last else None
            title = title_last.strip() if title_last else ""

        if ts is not None:
            timestamps.append(ts)
        titles.append(title)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(_tokenize_section(text[start:end]))

    return timestamps, titles, sections


def parse_custom(text: str, pattern: str) -> list[list[str]]:
    """Split on a user-provided regex and sentence-tokenize each section.

    Args:
        text: Raw transcript text.
        pattern: Regular expression whose matches act as section delimiters.

    Returns:
        List of sentence lists (one per section).
    """
    if not isinstance(text, str):
        return []
    parts = re.split(pattern, text)
    return [_tokenize_section(s) for s in parts if s.strip()]


def parse_custom_timestamped(
    text: str,
    pattern: str,
    timestamp_format: str = "H:MM:SS",
) -> tuple[list[float], list[str], list[list[str]]]:
    """Parse text with a custom regex pattern that captures timestamps and/or titles.

    The pattern should use named groups to capture:
        - ``(?P<timestamp>...)`` — the timestamp string
        - ``(?P<title>...)`` — the section title (optional)

    Examples:
        Pattern for ``[ Title @ 1:23:45 ]``::

            r"\\[\\s*(?P<title>[^@]+?)\\s*@\\s*(?P<timestamp>\\d+:\\d{2}:\\d{2})\\s*\\]"

        Pattern for ``# HH:MM:SS``::

            r"^#\\s*(?P<timestamp>\\d{2}:\\d{2}:\\d{2})$"

    Args:
        text: Raw transcript text.
        pattern: Regex with named groups ``timestamp`` and optionally ``title``.
        timestamp_format: How to parse the timestamp. Supported values:
            - ``"HH:MM:SS"`` — 2-digit hours, minutes, seconds (e.g., ``01:23:45``)
            - ``"H:MM:SS"`` — variable hours (e.g., ``1:23:45``)
            - ``"MM:SS"`` — 2-digit minutes and seconds (e.g., ``23:45``)
            - ``"M:SS"`` — variable minutes (e.g., ``3:45``)
            - A custom regex with named groups ``h``/``m``/``s`` (or ``hours``/``minutes``/``seconds``)

    Returns:
        Tuple of (timestamps, titles, sections).
    """
    if not isinstance(text, str):
        return [], [], []

    try:
        regex = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    matches = list(regex.finditer(text))
    if not matches:
        return [], [], []

    timestamps: list[float] = []
    titles: list[str] = []
    sections: list[list[str]] = []

    for i, match in enumerate(matches):
        gd = match.groupdict()

        # Extract timestamp
        ts_str = gd.get("timestamp", "")
        if ts_str:
            ts = _parse_timestamp_with_format(ts_str, timestamp_format)
            if ts is not None:
                timestamps.append(ts)

        # Extract title
        title = gd.get("title", "")
        titles.append(title.strip() if title else "")

        # Section text is from end of this match to start of next (or end of text)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(_tokenize_section(text[start:end]))

    return timestamps, titles, sections


def parse_transcript(
    text: str,
    fmt: str,
    custom_pattern: str | None = None,
    timestamp_format: str | None = None,
) -> ParseResult:
    """Parse a transcript string into sections of sentences.

    Args:
        text: Raw transcript text.
        fmt: Parser preset — one of:
            - ``"cstart"`` — ``[CSTART] Title [CEND]`` format
            - ``"cstart_ts"`` — ``[CSTART] HH:MM:SS - Title [CEND]`` format
            - ``"newline"`` — sections separated by blank lines
            - ``"markdown"`` — markdown headers (``# Title``)
            - ``"markdown_ts"`` — markdown with timestamps (``# 00:15:30 - Title``)
            - ``"custom"`` — user-provided regex pattern
            - ``"custom_ts"`` — user-provided regex with timestamp capture
        custom_pattern: Regex pattern (required for ``custom`` and ``custom_ts``).
        timestamp_format: Timestamp format for ``custom_ts`` (default: ``"H:MM:SS"``).

    Returns:
        ParseResult with sections, and optionally titles and timestamps.
    """
    if fmt == "cstart":
        titles, sections = parse_cstart(text)
        return ParseResult(sections=sections, titles=titles)

    elif fmt == "cstart_ts":
        timestamps, titles, sections = parse_cstart_timestamped(text)
        return ParseResult(sections=sections, titles=titles, timestamps=timestamps)

    elif fmt == "newline":
        sections = parse_newline(text)
        return ParseResult(sections=sections)

    elif fmt == "markdown":
        titles, sections = parse_markdown(text)
        return ParseResult(sections=sections, titles=titles)

    elif fmt == "markdown_ts":
        timestamps, titles, sections = parse_markdown_timestamped(text)
        return ParseResult(sections=sections, titles=titles, timestamps=timestamps)

    elif fmt == "custom":
        if custom_pattern is None:
            raise ValueError("custom_pattern is required when fmt='custom'")
        sections = parse_custom(text, custom_pattern)
        return ParseResult(sections=sections)

    elif fmt == "custom_ts":
        if custom_pattern is None:
            raise ValueError("custom_pattern is required when fmt='custom_ts'")
        ts_fmt = timestamp_format or "H:MM:SS"
        timestamps, titles, sections = parse_custom_timestamped(
            text, custom_pattern, ts_fmt
        )
        return ParseResult(sections=sections, titles=titles, timestamps=timestamps)

    else:
        raise ValueError(f"Unknown format: {fmt!r}")
