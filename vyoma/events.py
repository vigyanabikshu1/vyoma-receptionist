"""Lightweight JSONL event log shared by the receptionist and the GUI.

The receptionist process appends one JSON object per line to
``data/events.jsonl`` as things happen (visitor heard, facts retrieved,
tool called, reply spoken...). The GUI tails the same file and streams
it to the browser, so "what is going on inside" is visible live without
the two processes needing a direct connection.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path

_lock = threading.Lock()
_path: Path | None = None
_level_path: Path | None = None
_speak_path: Path | None = None
_levels: deque = deque(maxlen=50)   # ~5 s window at 100 ms/block
_wave: deque = deque(maxlen=800)    # signed samples for the oscilloscope trace

MAX_BYTES = 2_000_000   # rotate: keep the newest half when the log grows
KEEP_LINES = 1000


def configure(path: str | Path, level_path: str | Path | None = None,
              speak_path: str | Path | None = None) -> None:
    global _path, _level_path, _speak_path
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    if level_path is not None:
        _level_path = Path(level_path)
        _level_path.parent.mkdir(parents=True, exist_ok=True)
    if speak_path is not None:
        _speak_path = Path(speak_path)
        _speak_path.parent.mkdir(parents=True, exist_ok=True)


def _write_speak(payload: dict) -> None:
    if _speak_path is None:
        return
    tmp = _speak_path.with_suffix(".tmp")
    try:
        with _lock:
            # Atomic replace so the avatar never reads a half-written file.
            tmp.write_text(json.dumps(payload))
            tmp.replace(_speak_path)
    except OSError:
        pass


def speak_begin(text: str, meta: dict) -> None:
    """Publish 'the receptionist starts saying this now' for the avatar.

    ``meta`` is the lip-sync dict from :func:`visemes.analyze` (duration,
    fps, env, timeline). Written to a small state file that the avatar
    page polls (same pattern as the microphone level file), so it works
    across the receptionist/GUI process boundary. No-op until
    ``configure`` is given a ``speak_path``."""
    _write_speak({"ts": time.time(), "active": True, "text": text, **meta})


def speak_end() -> None:
    """Playback finished — the avatar closes its mouth."""
    _write_speak({"ts": time.time(), "active": False})


def mic_reset() -> None:
    """Clear the rolling window at the start of a new listening turn so the
    graph doesn't open showing the tail of the previous utterance."""
    _levels.clear()
    _wave.clear()


def mic_level(rms: float, active: bool,
              wave: list[float] | None = None) -> None:
    """Record the current microphone level for the dashboard's live audio
    graph. Overwrites a tiny JSON file (rolling window) instead of appending
    to the event log, so the 10 Hz updates during listening never flood it.
    ``wave`` is a block of signed samples (downsampled) that feed the
    oscilloscope trace. A no-op until ``configure`` is given a ``level_path``."""
    if _level_path is None:
        return
    _levels.append(round(float(rms), 4))
    if wave:
        _wave.extend(wave)
    payload = json.dumps({"ts": time.time(), "active": bool(active),
                          "levels": list(_levels), "wave": list(_wave)})
    tmp = _level_path.with_suffix(".tmp")
    try:
        with _lock:
            # Atomic replace so the GUI never reads a half-written file.
            tmp.write_text(payload)
            tmp.replace(_level_path)
    except OSError:
        pass


def emit(kind: str, **data) -> None:
    """Append one event; silently a no-op if not configured or on error."""
    if _path is None:
        return
    event = {"ts": time.time(), "kind": kind, **data}
    line = json.dumps(event, ensure_ascii=False, default=str)
    try:
        with _lock:
            if _path.exists() and _path.stat().st_size > MAX_BYTES:
                lines = _path.read_text().splitlines()[-KEEP_LINES:]
                _path.write_text("\n".join(lines) + "\n")
            with open(_path, "a") as f:
                f.write(line + "\n")
    except OSError:
        pass


def read_since(path: str | Path, offset: int) -> tuple[list[dict], int]:
    """Return (events, new_offset) for everything after byte ``offset``."""
    p = Path(path)
    if not p.exists():
        return [], 0
    size = p.stat().st_size
    if size < offset:      # file was rotated
        offset = 0
    if size == offset:
        return [], offset
    with open(p, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    # Only consume complete lines; a partially written line stays for
    # the next poll.
    complete, _, partial = chunk.rpartition(b"\n")
    if not complete:
        return [], offset
    events = []
    for line in complete.decode("utf-8", "ignore").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, offset + len(complete) + 1
