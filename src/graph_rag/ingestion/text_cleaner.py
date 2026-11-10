"""Conservative text cleaning that keeps headings, paragraphs and punctuation intact."""

import re

# Invisible characters that only add noise (zero-width space/joiners, byte-order mark).
_INVISIBLE_CHARS = re.compile(r"[​‌‍⁠﻿]")
# ASCII control characters except tab (\t) and newline (\n).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A run of spaces/tabs that follows visible text (leading indentation is kept).
_INNER_WHITESPACE = re.compile(r"(?<=\S)[ \t]+")
# Three or more newlines -> one blank line (a paragraph boundary).
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace and line endings without changing the text's meaning.

    Steps, in order:
    1. Normalize line endings (``\\r\\n``/``\\r`` -> ``\\n``); form feeds become newlines.
    2. Replace non-breaking spaces with normal spaces; drop invisible/control characters.
    3. Per line: collapse repeated inner spaces/tabs and strip trailing whitespace.
       Leading indentation is kept because it can carry meaning (e.g. nested lists).
    4. Collapse 3+ consecutive newlines into one blank line, preserving paragraphs.
    5. Strip leading/trailing blank space of the whole text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.replace(" ", " ")
    text = _INVISIBLE_CHARS.sub("", text)
    text = _CONTROL_CHARS.sub("", text)

    lines = [_INNER_WHITESPACE.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
