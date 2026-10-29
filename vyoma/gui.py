"""Vyoma control panel — local web GUI.

Usage:
    python -m vyoma.gui              # http://127.0.0.1:7860
    python -m vyoma.gui --port 8000

One offline page (no CDN, no internet) to:
- watch what is going on inside: live feed of heard text, retrieved
  FACTS, tool calls and replies (tailed from data/events.jsonl);
- start/stop the receptionist (voice or camera mode) as a subprocess;
- view the webcam with face-detection boxes (own preview, or the frames
  shared by a running receptionist that owns the camera);
- chat with the agent by keyboard to test it;
- upload documents / crawl the organisation website into the knowledge
  base, list and delete sources;
- edit settings (organisation, website domain, voice, webhook);
- generate and download the visitors report (HTML / Excel).
"""

import argparse
import atexit
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import (Flask, Response, jsonify, request, send_file,
                   send_from_directory)

from . import events, report
from .config import PROJECT_ROOT, load_config
from .db import Database
from .ingest import EXTRACTORS, ingest_file, ingest_url
from .kb import KnowledgeBase

logger = logging.getLogger("vyoma.gui")

app = Flask("vyoma")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


class State:
    """Everything the GUI process keeps between requests."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_dir = Path(cfg["data"]["db_path"]).parent
        self.events_path = self.data_dir / "events.jsonl"
        self.frame_path = self.data_dir / "camera_frame.jpg"
        self.kb_dir = Path(cfg["knowledge_base"]["dir"])
        self.proc: subprocess.Popen | None = None
        self.proc_mode: str | None = None
        self.chat_agent = None          # lazy ReceptionistAgent for web chat
        self.ab_engines: dict = {}      # lazy STT engines for the A/B test
        self.busy: str | None = None    # current background job description
        self.lock = threading.Lock()

    def reload_config(self):
        self.cfg = load_config()

    # -- receptionist subprocess ------------------------------------------
    def proc_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start_receptionist(self, mode: str) -> str:
        if self.proc_running():
            return "already running"
        args = [sys.executable, "-u", "-m", "vyoma"]
        if mode == "camera":
            args.append("--camera")
        self.proc = subprocess.Popen(
            args, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.proc_mode = mode
        threading.Thread(target=self._pump_output, args=(self.proc,),
                         daemon=True).start()
        events.emit("gui", message=f"Receptionist started ({mode} mode)")
        return "started"

    def _pump_output(self, proc: subprocess.Popen):
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                events.emit("log", line=line)
        code = proc.wait()
        events.emit("gui", message=f"Receptionist exited (code {code})")

    def stop_receptionist(self) -> str:
        if not self.proc_running():
            return "not running"
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        events.emit("gui", message="Receptionist stopped")
        return "stopped"

    # -- background jobs (ingestion / KB rebuild) --------------------------
    def run_job(self, description: str, fn) -> bool:
        """Run fn in a thread; only one job at a time."""
        with self.lock:
            if self.busy:
                return False
            self.busy = description

        def wrapper():
            events.emit("gui", message=f"Started: {description}")
            try:
                fn()
                events.emit("gui", message=f"Finished: {description}")
            except Exception as e:      # surface, don't crash the server
                logger.exception("Job failed: %s", description)
                events.emit("gui", message=f"FAILED: {description} — {e}")
            finally:
                self.busy = None

        threading.Thread(target=wrapper, daemon=True).start()
        return True

    def rebuild_kb(self):
        kb_cfg = self.cfg["knowledge_base"]
        KnowledgeBase(kb_cfg["dir"], kb_cfg["embedding_model"],
                      self.cfg["data"]["kb_cache"],
                      kb_cfg["top_k"], kb_cfg["min_score"])
        self.chat_agent = None          # pick up the new knowledge


STATE: State = None  # set in main()


class EventLogHandler(logging.Handler):
    """Forward this process's vyoma log records into the event feed."""

    def emit(self, record):
        if record.name.startswith(("vyoma", "werkzeug")):
            if record.name == "werkzeug":
                return              # request noise
            events.emit("log", line=self.format(record))


# ---------------------------------------------------------------------------
# Pages and static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_file(Path(__file__).parent / "dashboard.html")


@app.get("/avatar")
def avatar():
    """Visitor-facing animated face (what the robot's own screen shows)."""
    return send_file(Path(__file__).parent / "avatar.html")


@app.get("/reports/<path:name>")
def serve_report(name):
    return send_from_directory(STATE.data_dir, name)


# ---------------------------------------------------------------------------
# Status + live events
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status():
    cfg = STATE.cfg
    db = Database(cfg["data"]["db_path"])
    counts = {t: db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("visitors", "visits", "notifications")}
    kb_files = sorted(STATE.kb_dir.glob("*.md"))
    cache = Path(cfg["data"]["kb_cache"])
    chunks = 0
    if cache.exists():
        try:
            chunks = len(json.loads(cache.read_text()).get("entries", []))
        except (json.JSONDecodeError, OSError):
            pass
    ollama_up = True
    try:
        import ollama
        ollama.Client(host=cfg["llm"]["host"]).list()
    except Exception:
        ollama_up = False
    return jsonify({
        "organisation": cfg["organisation"]["name"],
        "assistant": cfg["organisation"]["assistant_name"],
        "model": cfg["llm"]["model"],
        "receptionist": {"running": STATE.proc_running(),
                         "mode": STATE.proc_mode},
        "busy": STATE.busy,
        "ollama": ollama_up,
        "db": counts,
        "kb": {"files": len(kb_files), "chunks": chunks},
    })


@app.get("/api/mic")
def api_mic():
    """Latest microphone levels for the dashboard's live audio graph.

    The receptionist process overwrites ``data/mic_level.json`` while it is
    listening; a stale/absent timestamp means it is not, and the graph hides.
    """
    p = STATE.data_dir / "mic_level.json"
    try:
        return Response(p.read_text(), mimetype="application/json")
    except (OSError, ValueError):
        return jsonify({"ts": 0, "active": False, "levels": [], "wave": []})


@app.get("/api/speak")
def api_speak():
    """Current speech state + lip-sync timeline for the avatar page.

    The receptionist process overwrites ``data/speak_state.json`` when a
    reply starts/finishes playing (see events.speak_begin). ``now`` is
    the server clock so the avatar can correct for client clock skew.
    """
    p = STATE.data_dir / "speak_state.json"
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        data = {"ts": 0, "active": False}
    data["now"] = time.time()
    return jsonify(data)


@app.get("/api/events")
def api_events():
    """Server-sent events: tail events.jsonl."""
    def stream():
        # start with the recent tail so the feed isn't empty on load
        offset = 0
        if STATE.events_path.exists():
            size = STATE.events_path.stat().st_size
            offset = max(0, size - 20_000)
        while True:
            batch, offset = events.read_since(STATE.events_path, offset)
            for ev in batch:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield ": ping\n\n"
            time.sleep(0.7)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Receptionist control + web chat
# ---------------------------------------------------------------------------

@app.get("/api/receptionist/state")
def api_recep_state():
    """Cheap 'is the receptionist running' probe for the avatar kiosk.

    Unlike /api/status it touches neither the database nor Ollama, so the
    fullscreen face can poll it every couple of seconds to keep its
    Stop/Resume buttons honest.
    """
    return jsonify({"running": STATE.proc_running(), "mode": STATE.proc_mode})


@app.post("/api/receptionist/start")
def api_start():
    mode = (request.json or {}).get("mode", "voice")
    return jsonify({"result": STATE.start_receptionist(mode)})


@app.post("/api/receptionist/stop")
def api_stop():
    return jsonify({"result": STATE.stop_receptionist()})


@app.post("/api/receptionist/force-listen")
def api_force_listen():
    """Manually start a conversation without waiting for a face.

    Drops a flag file the running camera-mode receptionist polls in
    ``FaceWatcher.wait_for_visitor``. No-op unless it is running in a mode
    with the camera watcher (camera mode).
    """
    if not STATE.proc_running():
        return jsonify({"error": "receptionist not running"}), 409
    if STATE.proc_mode != "camera":
        return jsonify({"error": "force listen needs camera mode"}), 409
    try:
        (STATE.data_dir / "force_listen.flag").write_text(str(time.time()))
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    events.emit("gui", message="Force-listen requested")
    return jsonify({"result": "listening now"})


@app.post("/api/chat")
def api_chat():
    """Keyboard chat with a GUI-owned agent (independent of the
    receptionist process; shares the same Ollama server and database)."""
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400
    try:
        if STATE.chat_agent is None:
            from .agent import ReceptionistAgent
            from .tools import Tools
            cfg = STATE.cfg
            kb = KnowledgeBase(cfg["knowledge_base"]["dir"],
                               cfg["knowledge_base"]["embedding_model"],
                               cfg["data"]["kb_cache"],
                               cfg["knowledge_base"]["top_k"],
                               cfg["knowledge_base"]["min_score"])
            tools = Tools(Database(cfg["data"]["db_path"]),
                          cfg["notifications"]["webhook_url"])
            STATE.chat_agent = ReceptionistAgent(cfg, kb, tools)
            events.emit("gui", message="Web chat agent ready")
        events.emit("heard", text=text, source="web-chat")
        reply = STATE.chat_agent.respond(text)
        events.emit("speak", text=reply, source="web-chat")
        ended = STATE.chat_agent.finished
        if ended:                       # next message starts a fresh visit
            STATE.chat_agent = None
        return jsonify({"reply": reply, "ended": ended})
    except Exception as e:
        logger.exception("Chat failed")
        return jsonify({"error": str(e)}), 500


@app.post("/api/tts")
def api_tts():
    """Synthesize a reply to WAV so the chat's play button can replay it
    in the browser (using the configured voice, no server speakers)."""
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400
    from .tts import synth_wav
    try:
        wav = synth_wav(STATE.cfg, text)
    except Exception as e:
        logger.exception("TTS synthesis failed")
        return jsonify({"error": str(e)}), 500
    if not wav:
        return jsonify({"error": "no text-to-speech voice available"}), 503
    return Response(wav, mimetype="audio/wav",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _mjpeg_generator():
    """Own camera preview with face boxes; while the camera is busy
    (receptionist owns it), relay the frames it shares. The stream
    never ends on its own — it switches between the two modes as
    camera ownership changes, so the browser image stays live."""
    import cv2

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    idx = STATE.cfg["vision"]["camera_index"]
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def own_capture():
        """Yield frames from our own camera until it is lost/taken.
        Never touches the camera while the receptionist is running —
        she has priority, the preview only relays her frames then."""
        if STATE.proc_running():
            return
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        faces: list = []
        n = 0
        try:
            while True:
                if STATE.proc_running():
                    return          # release the camera for the receptionist
                ok, frame = cap.read()
                if not ok:
                    return
                # Detection is expensive: half-size grayscale, every 3rd
                # frame; boxes are reused so the video stays smooth.
                if n % 3 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(gray, None, fx=0.5, fy=0.5)
                    faces = [(x * 2, y * 2, w * 2, h * 2)
                             for (x, y, w, h) in detector.detectMultiScale(
                                 small, scaleFactor=1.2, minNeighbors=5,
                                 minSize=(40, 40))]
                n += 1
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (60, 180, 75), 2)
                cv2.putText(frame, f"faces: {len(faces)}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 180, 75), 2)
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    yield boundary + jpg.tobytes() + b"\r\n"
        finally:
            cap.release()

    while True:
        yield from own_capture()
        # Camera unavailable — relay the receptionist's shared frames
        # for a while, then try to reclaim the camera.
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if STATE.frame_path.exists():
                try:
                    yield boundary + STATE.frame_path.read_bytes() + b"\r\n"
                except OSError:
                    pass
            time.sleep(0.4)


@app.get("/camera/stream")
def camera_stream():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

@app.get("/api/knowledge")
def api_knowledge():
    files = [{"name": f.name, "size": f.stat().st_size,
              "ingested": f.name.startswith(("web-", "doc-"))}
             for f in sorted(STATE.kb_dir.glob("*.md"))]
    return jsonify({"files": files, "busy": STATE.busy})


@app.post("/api/knowledge/upload")
def api_upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "no file"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in EXTRACTORS:
        return jsonify({"error": f"unsupported type {suffix} "
                        f"(use: {', '.join(EXTRACTORS)})"}), 400
    tmp = STATE.data_dir / "upload" / Path(file.filename).name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    file.save(tmp)

    def job():
        try:
            ingest_file(tmp, STATE.kb_dir)
            STATE.rebuild_kb()
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

    if not STATE.run_job(f"ingest {tmp.name}", job):
        return jsonify({"error": f"busy: {STATE.busy}"}), 409
    return jsonify({"result": "ingesting"})


@app.post("/api/knowledge/ingest_url")
def api_ingest_url():
    body = request.json or {}
    url = body.get("url", "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not url:
        return jsonify({"error": "no url"}), 400
    single = bool(body.get("single"))
    max_pages = int(body.get("max_pages") or 0)

    def job():
        pages = ingest_url(url, STATE.kb_dir, crawl=not single,
                           max_pages=max_pages)
        events.emit("gui", message=f"Fetched {len(pages)} page(s) from {url}")
        STATE.rebuild_kb()

    if not STATE.run_job(f"crawl {url}", job):
        return jsonify({"error": f"busy: {STATE.busy}"}), 409
    return jsonify({"result": "crawling"})


@app.post("/api/knowledge/recrawl")
def api_recrawl():
    """Delete all previously crawled pages and crawl the website fresh.
    Uploaded documents (doc-*) and hand-written files are untouched."""
    body = request.json or {}
    url = (body.get("url") or "").strip() \
        or STATE.cfg["organisation"].get("website", "")
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not url:
        return jsonify({"error": "no website configured — set it in "
                        "Settings first"}), 400

    def job():
        old = sorted(STATE.kb_dir.glob("web-*.md"))
        for f in old:
            f.unlink()
        events.emit("gui", message=f"Removed {len(old)} crawled pages; "
                    f"re-crawling {url} …")
        pages = ingest_url(url, STATE.kb_dir, crawl=True)
        events.emit("gui", message=f"Fetched {len(pages)} pages from {url}")
        STATE.rebuild_kb()

    if not STATE.run_job(f"re-crawl {url}", job):
        return jsonify({"error": f"busy: {STATE.busy}"}), 409
    return jsonify({"result": "recrawling", "url": url})


@app.delete("/api/knowledge/<name>")
def api_knowledge_delete(name):
    if "/" in name or not name.endswith(".md"):
        return jsonify({"error": "bad name"}), 400
    target = STATE.kb_dir / name
    if not target.exists():
        return jsonify({"error": "not found"}), 404
    target.unlink()
    events.emit("gui", message=f"Removed knowledge file {name}")
    STATE.run_job("rebuild knowledge index", STATE.rebuild_kb)
    return jsonify({"result": "removed"})


# ---------------------------------------------------------------------------
# Settings  (targeted line edits keep config.yaml comments intact)
# ---------------------------------------------------------------------------

EDITABLE = {
    "org_name": ("organisation", "name"),
    "assistant_name": ("organisation", "assistant_name"),
    "website": ("organisation", "website"),
    "address": ("organisation", "address"),
    "phone": ("organisation", "phone"),
    "email": ("organisation", "email"),
    "voice": ("tts", "voice"),
    "webhook_url": ("notifications", "webhook_url"),
    "vision_enabled": ("vision", "enabled"),
    "camera_index": ("vision", "camera_index"),
    "hotwords": ("stt", "hotwords"),
    "noise_adaptive": ("stt", "noise_adaptive"),
    "input_device": ("stt", "input_device"),
    "llm_model": ("llm", "model"),
    "stt_engine": ("stt", "engine"),
    "tts_engine": ("tts", "engine"),
}


def _set_config_value(text: str, section: str, key: str, value) -> str:
    """Rewrite one `section.key` value in config.yaml text, preserving
    all other lines (and their comments). Inserts the key under the
    section header if it doesn't exist yet."""
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = json.dumps(str(value))
    lines = text.splitlines()
    in_section = False
    section_header = None
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(section)}:\s*(#.*)?$", line):
            in_section = True
            section_header = i
            continue
        if in_section:
            if line and not line[0].isspace() and not line.startswith("#"):
                break  # next top-level section; key not found
            m = re.match(rf"^(\s+){re.escape(key)}:", line)
            if m:
                comment = ""
                cm = re.search(r"\s(#.*)$", line)
                if cm:
                    comment = "  " + cm.group(1)
                lines[i] = f"{m.group(1)}{key}: {rendered}{comment}"
                return "\n".join(lines) + "\n"
    if section_header is not None:
        lines.insert(section_header + 1, f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"


def _v4l2_can_capture(path: str) -> bool | None:
    """True/False if /dev/videoN is a video-capture device, per V4L2's
    VIDIOC_QUERYCAP. None if we couldn't ask (non-Linux, permissions).
    QUERYCAP doesn't need exclusive access, so this is safe to call even
    while the receptionist is streaming from the camera."""
    import ctypes
    import fcntl

    V4L2_CAP_VIDEO_CAPTURE = 0x00000001
    STRUCT_SIZE = 104  # struct v4l2_capability
    VIDIOC_QUERYCAP = (2 << 30) | (STRUCT_SIZE << 16) | (ord("V") << 8) | 0
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(STRUCT_SIZE)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
        # device_caps (offset 84) is per-node; fall back to the driver's
        # combined capabilities (offset 80) when the node reports none.
        caps = (int.from_bytes(buf[84:88], "little")
                or int.from_bytes(buf[80:84], "little"))
        return bool(caps & V4L2_CAP_VIDEO_CAPTURE)
    except OSError:
        return None
    finally:
        os.close(fd)


def _list_cameras() -> list[dict]:
    """Enumerate connected webcams as [{"index", "name"}], one entry per
    physical camera. Modern laptops expose several /dev/video* nodes per
    camera (a capture node plus metadata/IR ones); we keep only the
    video-capture nodes and, among nodes sharing a device name, the
    lowest index — the one OpenCV opens by default."""
    import glob

    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda p: int(re.sub(r"\D", "", p) or 0))
    cams: list[dict] = []
    seen_names: set[str] = set()
    for path in nodes:
        m = re.search(r"(\d+)$", path)
        if not m:
            continue
        idx = int(m.group(1))
        if _v4l2_can_capture(path) is False:
            continue  # metadata/output node — not a webcam feed
        name = f"Camera {idx}"
        try:
            name = (Path(f"/sys/class/video4linux/video{idx}/name")
                    .read_text().strip() or name)
        except Exception:
            pass
        if name in seen_names:
            continue  # a duplicate node for a camera we already listed
        seen_names.add(name)
        cams.append({"index": idx, "name": f"{name} (/dev/video{idx})"})
    return cams


def _list_input_devices() -> list[dict]:
    """Enumerate audio capture devices as [{"name"}], de-duplicated by name.

    The stored setting is a device-name substring (stable across reboots,
    unlike the numeric index), which sounddevice matches at capture time.
    """
    try:
        import sounddevice as sd
    except Exception:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    try:
        for d in sd.query_devices():
            if d.get("max_input_channels", 0) > 0:
                name = d["name"]
                if name and name not in seen:
                    seen.add(name)
                    out.append({"name": name})
    except Exception:
        pass
    return out


@app.get("/api/settings")
def api_settings():
    cfg = STATE.cfg
    voices = sorted(p.stem for p in
                    Path(cfg["tts"]["voices_dir"]).glob("*.onnx"))
    # Chat-capable models installed in Ollama (embedding models are not
    # selectable as the reasoning brain)
    llm_models: list[str] = []
    try:
        import ollama
        listed = ollama.Client(host=cfg["llm"]["host"]).list().models
        llm_models = sorted(m.model for m in listed
                            if "embed" not in m.model.lower())
    except Exception:
        pass
    current = cfg["llm"]["model"]
    if current not in llm_models and f"{current}:latest" not in llm_models:
        llm_models.insert(0, current)
    # Always offer the configured camera, even if it's busy or absent now
    cameras = _list_cameras()
    cam_index = int(cfg["vision"]["camera_index"])
    if not any(c["index"] == cam_index for c in cameras):
        cameras.insert(0, {"index": cam_index,
                           "name": f"Camera {cam_index} (configured)"})
    return jsonify({
        "llm_model": current,
        "llm_models": llm_models,
        "stt_engine": cfg["stt"]["engine"],
        "stt_engines": (["faster-whisper", "indic-whisper", "vosk"]
                        + (["deepgram"] if _deepgram_key() else [])),
        "tts_engine": cfg["tts"]["engine"],
        "tts_engines": (["piper", "espeak"]
                        + (["deepgram"] if _deepgram_key() else [])),
        "org_name": cfg["organisation"]["name"],
        "assistant_name": cfg["organisation"]["assistant_name"],
        "website": cfg["organisation"].get("website", ""),
        "address": cfg["organisation"].get("address", ""),
        "phone": cfg["organisation"].get("phone", ""),
        "email": cfg["organisation"].get("email", ""),
        "voice": cfg["tts"]["voice"],
        "webhook_url": cfg["notifications"]["webhook_url"] or "",
        "vision_enabled": bool(cfg["vision"]["enabled"]),
        "camera_index": int(cfg["vision"]["camera_index"]),
        "cameras": cameras,
        "hotwords": cfg["stt"].get("hotwords", ""),
        "noise_adaptive": bool(cfg["stt"].get("noise_adaptive", False)),
        "input_device": cfg["stt"].get("input_device", ""),
        "input_devices": _list_input_devices(),
        "voices": voices,
    })


@app.post("/api/settings")
def api_settings_save():
    body = request.json or {}
    cfg_path = PROJECT_ROOT / "config.yaml"
    text = cfg_path.read_text()
    for field, (section, key) in EDITABLE.items():
        if field in body:
            value = body[field]
            if field in ("vision_enabled", "noise_adaptive"):
                value = bool(value)
            elif field == "camera_index":
                value = int(value)
            text = _set_config_value(text, section, key, value)
    cfg_path.write_text(text)
    STATE.reload_config()
    STATE.chat_agent = None
    events.emit("gui", message="Settings saved"
                + (" — restart the receptionist to apply"
                   if STATE.proc_running() else ""))
    return jsonify({"result": "saved"})


# ---------------------------------------------------------------------------
# Speech A/B test: same audio through every engine, side by side
# ---------------------------------------------------------------------------

def _load_wav_16k(raw: bytes):
    """Uploaded WAV → mono float32 at 16 kHz (what both engines expect)."""
    import io as _io
    import wave

    import numpy as np

    with wave.open(_io.BytesIO(raw)) as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM WAV files are supported")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _vosk_model_dir(cfg) -> tuple[str, str]:
    """Return (model_path, label). Prefers the configured model; falls
    back to the small en-IN model while the big one is downloading."""
    name = cfg["stt"].get("vosk_model", "vosk-model-en-in-0.5")
    cache = Path.home() / ".cache" / "vosk"
    for candidate, label in ((Path(name), Path(name).name),
                             (cache / name, Path(name).name),
                             (cache / "vosk-model-small-en-in-0.4",
                              "vosk-model-small-en-in-0.4 (fallback)")):
        if candidate.is_dir():
            return str(candidate), label
    part = next(cache.glob("*.part"), None)
    got = f" ({part.stat().st_size // 1_000_000} MB downloaded so far)" \
        if part else ""
    raise RuntimeError(f"vosk model '{name}' not downloaded yet{got}")


def _deepgram_key() -> str:
    return (STATE.cfg.get("deepgram") or {}).get("api_key", "")


def _ab_engine(kind: str):
    """Lazy-load and cache one engine for the A/B test."""
    if kind in STATE.ab_engines:
        return STATE.ab_engines[kind]
    from .stt import DeepgramSTT, IndicWhisperSTT, VoskSTT, WhisperSTT

    cfg = dict(STATE.cfg["stt"])
    if kind == "vosk":
        cfg["vosk_model"], label = _vosk_model_dir(STATE.cfg)
        engine = VoskSTT(cfg)
        engine.label = f"Vosk {label}"
    elif kind == "indic-whisper":
        cfg["local_files_only"] = True   # fail fast if not downloaded yet
        engine = IndicWhisperSTT(cfg)
        engine.label = "IndicWhisper (faster-whisper)"
    elif kind == "deepgram":
        cfg["deepgram"] = STATE.cfg.get("deepgram") or {}
        engine = DeepgramSTT(cfg)
        engine.label = (f"Deepgram {engine.model} {engine.language} (cloud)")
    else:
        cfg["local_files_only"] = True   # fail fast if not downloaded yet
        engine = WhisperSTT(cfg)
        engine.label = f"Whisper {cfg['model']} (faster-whisper)"
    STATE.ab_engines[kind] = engine
    return engine


@app.post("/api/stt_ab")
def api_stt_ab():
    file = request.files.get("file")
    if file:
        try:
            audio = _load_wav_16k(file.read())
        except Exception as e:
            return jsonify({"error": f"could not read WAV: {e}"}), 400
        source = file.filename
    else:
        if STATE.proc_running():
            return jsonify({"error": "stop the receptionist first — it is "
                            "using the microphone"}), 409
        from .stt import MicSTT
        try:
            recorder = MicSTT(dict(STATE.cfg["stt"]))
            events.emit("gui", message="Speech test: listening…")
            audio = recorder._record_utterance()
        except Exception as e:
            return jsonify({"error": f"microphone: {e}"}), 500
        if audio is None:
            return jsonify({"error": "heard nothing — speak within "
                            f"{STATE.cfg['stt']['max_utterance_seconds']}s "
                            "of pressing the button"}), 400
        source = "microphone"

    engines = ["faster-whisper", "indic-whisper", "vosk"]
    if _deepgram_key():
        engines.append("deepgram")
    results = []
    for kind in engines:
        try:
            engine = _ab_engine(kind)
            t0 = time.perf_counter()
            text = engine.transcribe(audio)
            results.append({"engine": engine.label,
                            "seconds": round(time.perf_counter() - t0, 2),
                            "text": text or "(silence)"})
        except Exception as e:
            results.append({"engine": kind, "seconds": None,
                            "error": str(e)})
    events.emit("gui", message="Speech test (" + source + "): " + " | ".join(
        f"{r['engine']}: {r.get('text', r.get('error'))}" for r in results))
    return jsonify({"source": source,
                    "duration": round(len(audio) / 16000, 1),
                    "results": results})


# ---------------------------------------------------------------------------
# Identity detection: learn who the receptionist works for from the KB
# ---------------------------------------------------------------------------

IDENTITY_QUERIES = [
    "contact us address phone number email",
    "about the organisation official name",
    "location directions how to reach office",
]

IDENTITY_PROMPT = """\
You are reading excerpts from an organisation's own website/documents.
Extract the organisation's identity for its reception desk. Reply with
ONLY a JSON object with these keys (use "" for anything not found in
the text — never invent values):
  "org_name":  the official name of the organisation these pages are about
  "address":   postal address on one line; if no full address appears,
               give what the text supports, e.g. "Alathur, Palakkad
               district, Kerala"
  "phone":     main phone number(s), comma separated
  "email":     main contact email
  "website":   website domain (the Source: URLs reveal it)

EXCERPTS:
{context}
"""


_NON_ANSWER = re.compile(
    r"not (fully )?(specified|found|available|provided|mentioned)"
    r"|unknown|n/?a\b|no .{0,20}(found|provided)|excerpt|cannot|inferred",
    re.I)


def _clean_field(key: str, value: str) -> str:
    """Drop model waffle ('Not specified in the excerpts…') and values
    that don't look like the field they claim to be."""
    value = value.strip().strip('"')
    if not value or len(value) > 160 or _NON_ANSWER.search(value):
        return ""
    if key == "email" and "@" not in value:
        return ""
    if key == "phone" and not re.search(r"\d{4}", value):
        return ""
    return value


@app.post("/api/identity/detect")
def api_identity_detect():
    cfg = STATE.cfg
    try:
        kb = KnowledgeBase(cfg["knowledge_base"]["dir"],
                           cfg["knowledge_base"]["embedding_model"],
                           cfg["data"]["kb_cache"],
                           top_k=4,
                           min_score=cfg["knowledge_base"]["min_score"])
        if not kb.entries:
            return jsonify({"error": "knowledge base is empty — upload "
                            "documents or crawl the website first"}), 400
        # Home/contact/about pages carry the identity (crawled page
        # files are named after their URL path). Crawled pages first —
        # they are the organisation's own site; hand-written notes may
        # be about something else.
        markers = ("contact", "about", "home", "index")
        by_name = sorted(STATE.kb_dir.glob("web-*.md"),
                         key=lambda f: len(f.name))
        picked = by_name[:1]            # site root page
        picked += [f for f in by_name
                   if any(m in f.name.lower() for m in markers)][:5]
        chunks = [f.read_text()[:1400] for f in dict.fromkeys(picked)]
        for q in IDENTITY_QUERIES:
            for c in kb.search(q):
                if c not in chunks:
                    chunks.append(c)
        context = "\n---\n".join(chunks)[:8000]

        import ollama
        client = ollama.Client(host=cfg["llm"]["host"])
        fields = ("org_name", "address", "phone", "email", "website")
        result = dict.fromkeys(fields, "")
        # Small models are inconsistent extractors — two passes, keeping
        # the first non-empty value per field.
        for _ in range(2):
            resp = client.chat(
                model=cfg["llm"]["model"],
                messages=[{"role": "user",
                           "content": IDENTITY_PROMPT.format(context=context)}],
                format="json",
                options={"temperature": 0.2},
                keep_alive=cfg["llm"]["keep_alive"])
            data = json.loads(resp.message.content or "{}")
            for k in fields:
                result[k] = result[k] or _clean_field(
                    k, str(data.get(k) or ""))
            if all(result.values()):
                break
        # The website is in the crawled pages' Source: URLs — no LLM needed
        m = re.search(r"^# Source: https?://([^/\s]+)", chunks[0], re.M) \
            if chunks else None
        if m:
            result["website"] = m.group(1)
        # suggest STT hotwords from the detected identity
        words = [result["org_name"],
                 cfg["organisation"]["assistant_name"]]
        result["hotwords"] = " ".join(w for w in words if w)
        events.emit("gui", message="Identity detected from knowledge base: "
                    + (result["org_name"] or "(name not found)"))
        return jsonify(result)
    except Exception as e:
        logger.exception("Identity detection failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.post("/api/report")
def api_report():
    cfg = STATE.cfg
    data = report.fetch_report_data(cfg["data"]["db_path"])
    xlsx = report.build_xlsx([
        ("Visits", report.VISITS_COLS, data["visits"]),
        ("Visitors", report.VISITORS_COLS, data["visitors"]),
        ("Notifications", report.NOTIFY_COLS, data["notifications"]),
    ])
    page = report.build_html(data, cfg["organisation"]["name"],
                             cfg["organisation"]["assistant_name"],
                             cfg["data"]["db_path"], xlsx)
    out = STATE.data_dir / "visitors_report.html"
    out.write_text(page, encoding="utf-8")
    (STATE.data_dir / "visitors_report.xlsx").write_bytes(xlsx)
    events.emit("gui", message="Visitors report generated")
    return jsonify({"html": "/reports/visitors_report.html",
                    "xlsx": "/reports/visitors_report.xlsx",
                    "visits": len(data["visits"])})


@app.get("/api/visits")
def api_visits():
    data = report.fetch_report_data(STATE.cfg["data"]["db_path"])
    return jsonify({"columns": report.VISITS_COLS,
                    "rows": data["visits"][:50]})


# ---------------------------------------------------------------------------

def _is_receptionist_cmdline(argv: list[str]) -> bool:
    """True for `python -m vyoma [--camera ...]` but not `-m vyoma.gui`.

    The receptionist is launched as the ``vyoma`` module; the control
    panel is ``vyoma.gui``. We only ever want to reap the former.
    """
    for i, tok in enumerate(argv[:-1]):
        if tok == "-m":
            return argv[i + 1] == "vyoma"
    return False


def reap_stray_receptionists() -> int:
    """Terminate leftover receptionist processes this user owns.

    A camera-mode receptionist holds /dev/video0; if a previous control
    panel was killed without stopping it, it lingers as an orphan and
    keeps the camera locked so the new panel can never acquire it.
    Called at startup, before the server binds, to clear the deck.
    """
    my_pid, my_uid = os.getpid(), os.getuid()
    reaped = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != my_uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue                    # process vanished or not ours
        argv = [a for a in argv if a]
        if not _is_receptionist_cmdline(argv):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            reaped += 1
            logger.warning("Reaped stray receptionist pid %s", pid)
        except (ProcessLookupError, PermissionError):
            pass
    if reaped:
        events.emit("gui", message=f"Cleared {reaped} stray receptionist "
                                   f"process(es) holding the camera")
    return reaped


def main():
    global STATE
    parser = argparse.ArgumentParser(description="Vyoma control panel")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to open from another device")
    parser.add_argument("--start", choices=("camera", "voice"),
                        help="launch the receptionist immediately on startup "
                             "(use 'camera' for the headless robot)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config()
    STATE = State(cfg)
    events.configure(STATE.events_path)

    handler = EventLogHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    # A previous panel may have been killed without stopping its
    # camera-mode receptionist; clear those orphans before we start so
    # the camera is free for this session.
    reap_stray_receptionists()

    # Never leave our own receptionist child running past our exit.
    def _cleanup(*_):
        if STATE is not None and STATE.proc_running():
            STATE.stop_receptionist()

    atexit.register(_cleanup)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, lambda *_: (_cleanup(), sys.exit(0)))

    events.emit("gui", message="Control panel started")
    print(f"\n  Vyoma control panel:  http://{args.host}:{args.port}\n")

    # Headless robot: bring the receptionist up with the panel so it
    # starts detecting faces on powerup. The panel owns the child, so the
    # dashboard's status, Start/Stop button and live preview all track it.
    if args.start:
        STATE.start_receptionist(args.start)

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
