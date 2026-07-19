"""Voice output: neural TTS with Kokoro (human-like, GPU-accelerated).

Each participant maps to a distinct Kokoro voice (see VOICES below). Audio is
generated locally and played through your speakers. Heavy deps (kokoro, torch,
sounddevice) are imported lazily so text-only mode needs none of them.
"""
from __future__ import annotations

SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz

# A few good, clearly-distinct Kokoro voices. Full list: American female af_*,
# American male am_*, British female bf_*, British male bm_*.
VOICES = {
    "af_heart": "American female, warm",
    "am_michael": "American male, clear",
    "bf_emma": "British female",
    "bm_george": "British male",
    "af_bella": "American female, bright",
    "am_fenrir": "American male, deep",
}


class TTS:
    def __init__(self, voice_map: dict[str, str | None] | None = None, lang_code: str = "a"):
        import torch  # lazy
        from kokoro import KPipeline  # lazy

        # lang_code 'a' = American English, 'b' = British English.
        self._pipeline = KPipeline(lang_code=lang_code)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._voice_map = voice_map or {}
        self._default_voice = "af_heart"

    def _voice_for(self, speaker: str) -> str:
        v = self._voice_map.get(speaker)
        return v if v else self._default_voice

    def say(self, speaker: str, text: str) -> None:
        import numpy as np
        import sounddevice as sd

        voice = self._voice_for(speaker)
        chunks = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice):
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio)
        if not chunks:
            return
        wav = np.concatenate(chunks).astype("float32")
        sd.play(wav, SAMPLE_RATE)
        sd.wait()

    @staticmethod
    def list_voices() -> dict[str, str]:
        return dict(VOICES)
