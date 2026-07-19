"""Voice input: record from the mic and transcribe with faster-whisper.

Heavy deps (faster_whisper, sounddevice, numpy) are imported lazily so the
project runs in text-only mode without them installed.
"""
from __future__ import annotations

import queue
import threading

SAMPLERATE = 16000


class STT:
    def __init__(self, whisper_model: str = "base"):
        from faster_whisper import WhisperModel  # lazy

        # device="auto" uses your RTX/CUDA when available, else CPU.
        self._model = WhisperModel(whisper_model, device="auto", compute_type="auto")

    def _record_until_enter(self):
        import numpy as np
        import sounddevice as sd

        print("Recording... press Enter to stop.")
        frames: list = []
        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def callback(indata, _frames, _time, _status):
            q.put(indata.copy())

        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

        with sd.InputStream(
            samplerate=SAMPLERATE, channels=1, dtype="float32", callback=callback
        ):
            while not stop.is_set():
                try:
                    frames.append(q.get(timeout=0.1))
                except queue.Empty:
                    pass

        if not frames:
            return np.zeros(0, dtype="float32")
        return np.concatenate(frames, axis=0).flatten()

    def listen(self) -> str:
        """Record until Enter, return the transcribed text."""
        audio = self._record_until_enter()
        if audio.size == 0:
            return ""
        segments, _ = self._model.transcribe(audio, language=None)
        return " ".join(seg.text.strip() for seg in segments).strip()
