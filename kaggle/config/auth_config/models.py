"""Authentication models and configurations for Kaggle multi-account support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KaggleAccount:
    """Represents a Kaggle account credential set."""

    username: str
    key: str
    token: str = ""
    index: int = 0
    name: str = ""

    def get_slug(self, prefix: str = "assistant-worker-") -> str:
        """Derive a consistent Kaggle kernel slug from the username."""
        clean_name = self.username.lower().replace("_", "-")
        return f"{prefix}{clean_name}"

    @property
    def display_name(self) -> str:
        """User-friendly name for CLI menus and status banners."""
        if self.name:
            return self.name
        if self.index > 0:
            return f"Account #{self.index} ({self.username})"
        return self.username
