#!/usr/bin/env python3
"""multi-voice — local multi-agent AI conversation with text + voice I/O.

Usage:
    python run.py                 # start a conversation
    python run.py --list-voices   # list available voices (for config.yaml)
    python run.py --config other.yaml
"""
from __future__ import annotations

import argparse
import sys

# Windows consoles default to cp1252 and mangle em-dashes/quotes from the models.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from multivoice.config import load_config
from multivoice.llm import LLM
from multivoice.orchestrator import Orchestrator
from multivoice.transcript import Transcript


def build_speaker(cfg):
    """Return a say(speaker, text) callback, wiring TTS if enabled."""
    if not cfg.voice.tts_enabled:
        return None
    try:
        from multivoice.tts import TTS

        voice_map = {p.name: p.voice for p in cfg.participants}
        tts = TTS(voice_map)
        return tts.say
    except Exception as e:  # noqa: BLE001
        print(f"[warn] TTS unavailable ({e}). Continuing text-only.")
        return None


def build_listener(cfg):
    """Return an STT instance if voice input is enabled, else None."""
    if not cfg.voice.stt_enabled:
        return None
    try:
        from multivoice.stt import STT

        print("Loading Whisper model...")
        return STT(cfg.voice.whisper_model)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] STT unavailable ({e}). Falling back to typed input.")
        return None


def get_user_input(prompt: str, listener) -> str:
    """Typed input, with /voice to switch to speaking (if STT is on)."""
    raw = input(prompt).strip()
    if raw == "/voice" and listener:
        text = listener.listen()
        print(f"   heard: {text!r}")
        return text
    if raw == "/voice" and not listener:
        print("   (voice input not enabled — set stt_enabled: true in config.yaml)")
        return ""
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="multi-voice local AI roundtable")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args()

    if args.list_voices:
        from multivoice.tts import TTS

        print("Kokoro voices (set one per participant in config.yaml):")
        for name, desc in TTS.list_voices().items():
            print(f"  • {name:<14} {desc}")
        return 0

    cfg = load_config(args.config)
    llm = LLM(cfg.ollama_host)

    # Sanity-check that Ollama is up and models are pulled.
    available = set(llm.available_models())
    if not available:
        print("[warn] Couldn't reach Ollama. Is it running?  (try: ollama serve)")
    else:
        missing = [p.model for p in cfg.participants if p.model not in available]
        if missing:
            print("[warn] These models aren't pulled yet:")
            for m in missing:
                print(f"     ollama pull {m}")
            print()

    say = build_speaker(cfg)
    listener = build_listener(cfg)
    transcript = Transcript()
    orch = Orchestrator(cfg, llm, transcript)

    names = ", ".join(p.name for p in cfg.participants)
    print("\n" + "=" * 60)
    print(f"  multi-voice  —  participants: {names}")
    print("  Type a topic to begin. During your turn:")
    print("    /voice  speak instead of type      /quit  exit")
    print("=" * 60 + "\n")

    def on_reply(participant, text):
        print(f"\n{participant.name}: {text}")
        if say:
            say(participant.name, text)

    seed = get_user_input("You (opening topic): ", listener)
    if seed.lower() in {"/quit", "quit", "exit", ""}:
        print("Nothing to discuss.")
        return 0
    transcript.add("User", seed)

    try:
        while True:
            for _ in range(cfg.conversation.auto_rounds):
                orch.round(on_reply=on_reply)

            user = get_user_input(
                "\nYou (interject / Enter to continue / /quit): ", listener
            )
            if user.lower() in {"/quit", "quit", "exit"}:
                break
            if user:
                transcript.add("User", user)
    except KeyboardInterrupt:
        print()

    print("\nConversation ended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
