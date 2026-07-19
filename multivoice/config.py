"""Load and validate config.yaml into typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Participant:
    name: str
    model: str
    persona: str
    voice: str | None = None


@dataclass
class ConversationSettings:
    # "conversation": each agent sees earlier agents' replies within the same
    #                 round (real-time debate).
    # "query":        all agents answer the same frozen state independently;
    #                 their replies are only revealed to everyone next round.
    context_mode: str = "conversation"
    auto_rounds: int = 1
    context_turns: int = 20
    max_reply_sentences: int = 4
    temperature: float = 0.8


@dataclass
class VoiceSettings:
    tts_enabled: bool = False
    stt_enabled: bool = False
    whisper_model: str = "base"


@dataclass
class Config:
    ollama_host: str = "http://localhost:11434"
    participants: list[Participant] = field(default_factory=list)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    participants = [
        Participant(
            name=p["name"],
            model=p["model"],
            persona=" ".join(p.get("persona", "").split()),
            voice=p.get("voice"),
        )
        for p in data.get("participants", [])
    ]
    if not participants:
        raise ValueError("config.yaml must define at least one participant")

    return Config(
        ollama_host=data.get("ollama_host", "http://localhost:11434"),
        participants=participants,
        conversation=ConversationSettings(**(data.get("conversation") or {})),
        voice=VoiceSettings(**(data.get("voice") or {})),
    )
