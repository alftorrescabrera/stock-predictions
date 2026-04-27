"""Text cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TextCleaner:
    """Lightweight text cleaning for news content."""

    def clean(self, headline: Optional[str], summary: Optional[str]) -> str:
        """Combine headline and summary with minimal normalization."""
        headline = headline or ""
        summary = summary or ""
        text = f"{headline} {summary}".replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())
        return text.strip()
