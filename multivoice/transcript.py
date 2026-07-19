"""Shared conversation transcript that every participant reads from."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    speaker: str
    content: str


@dataclass
class Transcript:
    turns: list[Turn] = field(default_factory=list)

    def add(self, speaker: str, content: str) -> None:
        self.turns.append(Turn(speaker, content.strip()))

    def tail(self, n: int) -> list[Turn]:
        return self.turns[-n:] if n > 0 else list(self.turns)

    def as_text(self, n: int = 0) -> str:
        turns = self.tail(n) if n else self.turns
        return "\n".join(f"{t.speaker}: {t.content}" for t in turns)

    def is_empty(self) -> bool:
        return not self.turns
