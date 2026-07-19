"""Generate a sample WAV for a spread of Kokoro voices, so you can pick.

American voices use the 'a' pipeline, British use 'b'. Output: preview_<voice>.wav
"""
import soundfile as sf
import torch
from kokoro import KPipeline

LINE = "This is voice {name}. Here is how I sound making a point in a debate."

VOICES = [
    # American female
    "af_heart", "af_bella", "af_nicole", "af_sky",
    # American male
    "am_michael", "am_fenrir", "am_puck", "am_adam",
    # British female
    "bf_emma", "bf_isabella",
    # British male
    "bm_george", "bm_lewis",
]

pipes = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}

for v in VOICES:
    pipe = pipes["b"] if v.startswith("b") else pipes["a"]
    text = LINE.format(name=v)
    chunks = [torch.as_tensor(a) for _g, _p, a in pipe(text, voice=v)]
    if not chunks:
        print(f"{v}: (no audio)")
        continue
    wav = torch.cat(chunks).cpu().numpy()
    sf.write(f"preview_{v}.wav", wav, 24000)
    print(f"{v:<12} ok  ({len(wav)/24000:.1f}s)")

print("[preview generated]")
