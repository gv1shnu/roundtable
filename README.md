# multi-voice

A local roundtable where several open-weight LLMs discuss a topic and take turns
responding, with optional speech input and human-like spoken replies. Runs
entirely offline through [Ollama](https://ollama.com) — no cloud APIs, no keys,
no per-token cost.

Each participant is a different model with its own persona and voice, so the
conversation has genuinely different reasoning styles rather than one model
talking to itself.

> Note: these are open-weight stand-ins, not the hosted GPT / Claude / Gemini
> models (those aren't available to self-host). Gemma is Google's open model,
> the closest local equivalent to Gemini.

## Features

- Multiple local models in one conversation, round-robin turns
- Type your prompts; interject at any point
- Optional voice input (Whisper) and voice output (Kokoro, GPU-accelerated)
- Two context modes: real-time debate, or independent answers revealed each round
- Personas, voices, models, and pacing all set in one YAML file

## Requirements

- [Ollama](https://ollama.com/download) (Windows or macOS)
- Python 3.12 (the ML/audio stack does not yet ship wheels for 3.13+)
- A GPU helps but isn't required. Reference setup: RTX 3080 Ti Laptop, 16 GB VRAM.

## Setup

### 1. Models

```bash
ollama pull gemma3:12b
ollama pull llama3.1:8b
ollama pull qwen3.6            # or any Qwen you already have
```

Use smaller tags (`gemma3:4b`, `llama3.2:3b`, `qwen2.5:7b`) if you're short on
VRAM. Because turns are sequential, only one model is loaded at a time, so you
only need enough VRAM for the largest single model.

### 2. Python environment

```bash
py -3.12 -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS:
source .venv/bin/activate

# GPU (NVIDIA) build of torch — needed by Kokoro:
pip install torch --index-url https://download.pytorch.org/whl/cu126
# macOS: skip the line above and let the default torch install (Metal/CPU).

pip install -r requirements.txt
```

Text-only mode needs just `ollama` and `PyYAML`; the rest is for voice.

### 3. Run

```bash
python run.py
```

Type a topic to start. After each round: press Enter to let them continue, type
to join in, or `/quit` to stop.

## Configuration

Everything lives in [`config.yaml`](config.yaml).

### Participants

```yaml
participants:
  - name: "Gemmini"
    model: "gemma3:12b"
    voice: "am_michael"
    persona: >
      Analytical and structured...
```

### Context mode

```yaml
conversation:
  context_mode: "conversation"   # or "query"
```

- `conversation` — within a round, each model sees the earlier models' replies
  and reacts to them. A live debate.
- `query` — every model answers the same frozen state independently; the replies
  are revealed together and only become visible to the others the next round.

### Voice

```yaml
voice:
  tts_enabled: true     # models speak their replies (Kokoro)
  stt_enabled: false    # set true to speak your input with /voice (Whisper)
  whisper_model: "base"
```

List the available voices with:

```bash
python run.py --list-voices
```

## Performance

Models run one at a time and Ollama swaps them between turns, so a turn's speed
is roughly that model's own generation speed plus a short reload.

**`qwen3.6` (used by "Claudia") is the slow one.** At ~23 GB it exceeds a 16 GB
GPU and spills into system RAM, so its turns take noticeably longer than the
others, which stay fully on the GPU. If you want a snappier roundtable, point
that participant at a smaller model (e.g. `qwen2.5:7b` or `qwen2.5:14b`) in
`config.yaml`.

## Cross-platform

The same code runs on Windows (NVIDIA) and macOS (Apple Silicon). Only the venv
activate command and the torch install line differ; on macOS the default torch
build uses Metal/CPU.

## Layout

```
run.py              CLI entry point
config.yaml         participants, personas, voices, settings
voice_preview.py    generate sample WAVs for a spread of voices
test_modes.py       checks the two context modes behave correctly
multivoice/
  config.py         config loader
  llm.py            Ollama client
  transcript.py     shared conversation history
  orchestrator.py   turn-taking and prompting
  stt.py            Whisper voice input
  tts.py            Kokoro voice output
```

## License

MIT — see [LICENSE](LICENSE).
