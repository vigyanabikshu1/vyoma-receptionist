"""Viseme timeline + mouth-energy envelope for the avatar's lip-sync.

A viseme is the mouth shape a phoneme is seen as. Whenever the
receptionist speaks, ``analyze()`` turns the synthesized WAV plus its
text into a small JSON-able dict the avatar page animates from:

- ``env``:      RMS loudness per frame (how far the mouth opens);
- ``timeline``: [start_second, viseme] pairs (which mouth shape).

The shapes come from a grapheme heuristic — letters mapped straight to
the classic Preston Blair mouth classes, with vowels held longer —
scaled to the real audio duration. No phonemizer dependency; combined
with the true energy envelope (which closes the mouth in the gaps the
letter timing gets wrong) it reads convincingly on a kiosk avatar.
"""

import io
import wave

import numpy as np

FPS = 25            # envelope frames per second (40 ms — one film frame)

# Mouth-shape classes the avatar knows how to draw.
REST = "rest"       # relaxed/closed
VISEME_OF_LETTER = {
    # wide open vowels
    "a": "AI", "i": "AI", "y": "AI",
    # spread lips
    "e": "E",
    # rounded
    "o": "O",
    "u": "U", "w": "U",
    # lips pressed together
    "m": "MBP", "b": "MBP", "p": "MBP",
    # top teeth on lower lip
    "f": "FV", "v": "FV",
    # tongue up
    "l": "L",
    # everything else articulated with teeth nearly together
    "t": "TS", "d": "TS", "s": "TS", "z": "TS", "n": "TS", "r": "TS",
    "c": "TS", "g": "TS", "h": "TS", "j": "TS", "k": "TS", "q": "TS",
    "x": "TS",
}
VOWEL_WEIGHT = 2.0  # vowels are held ~twice as long as consonants
VOWELS = set("aeiouy")


def _wav_to_mono(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                 - 128) / 128
    else:  # 32-bit
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2**31
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def _envelope(audio: np.ndarray, rate: int) -> list[float]:
    """Per-frame RMS normalised to 0..1 (the mouth-openness signal)."""
    block = max(1, rate // FPS)
    n = len(audio) // block
    if n == 0:
        return [0.0]
    rms = np.sqrt(np.mean(
        audio[:n * block].reshape(n, block) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 1e-6:
        return [0.0] * n
    env = np.sqrt(rms / peak)          # sqrt lifts quiet speech visibly
    return [round(float(v), 3) for v in env]


def _letter_visemes(text: str) -> list[tuple[str, float]]:
    """(viseme, weight) per pronounceable letter, spaces = brief rests."""
    out: list[tuple[str, float]] = []
    for ch in text.lower():
        if ch.isspace() or ch in ",.;:!?…—-":
            if out and out[-1][0] != REST:
                out.append((REST, 0.6))     # inter-word gap
            continue
        v = VISEME_OF_LETTER.get(ch)
        if v:
            out.append((v, VOWEL_WEIGHT if ch in VOWELS else 1.0))
    return out


def analyze(wav_bytes: bytes, text: str) -> dict:
    """Lip-sync metadata for a spoken reply.

    Returns ``{"duration", "fps", "env", "timeline"}`` — everything the
    avatar needs, small enough to write to a JSON state file per reply.
    """
    audio, rate = _wav_to_mono(wav_bytes)
    duration = len(audio) / rate if rate else 0.0
    env = _envelope(audio, rate)

    letters = _letter_visemes(text)
    timeline: list[list] = []
    total = sum(w for _, w in letters)
    if duration > 0 and total > 0:
        # Piper puts ~a tenth of the audio into leading/trailing silence;
        # squeeze the letters into the middle so lips match voice onset.
        pad = min(0.15, duration * 0.06)
        t = pad
        per_unit = (duration - 2 * pad) / total
        last = None
        for viseme, weight in letters:
            if viseme != last:               # merge runs of the same shape
                timeline.append([round(t, 3), viseme])
                last = viseme
            t += weight * per_unit
    if not timeline:
        timeline = [[0.0, REST]]

    return {"duration": round(duration, 3), "fps": FPS,
            "env": env, "timeline": timeline}


def estimate(text: str, wps: float = 2.5) -> dict:
    """Timeline without audio (console TTS): duration guessed from the
    word count so the avatar can still mouth along."""
    words = max(1, len(text.split()))
    duration = words / wps
    letters = _letter_visemes(text)
    total = sum(w for _, w in letters) or 1.0
    t, last, timeline = 0.0, None, []
    for viseme, weight in letters:
        if viseme != last:
            timeline.append([round(t, 3), viseme])
            last = viseme
        t += weight * duration / total
    n = max(1, int(duration * FPS))
    return {"duration": round(duration, 3), "fps": FPS,
            "env": [0.7] * n, "timeline": timeline or [[0.0, REST]]}
