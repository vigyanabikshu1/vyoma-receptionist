"""Speech-to-text: microphone capture with energy VAD + faster-whisper."""

import logging
import time
import wave
from pathlib import Path

import numpy as np

from . import events

logger = logging.getLogger(__name__)


def _fmt(x: float | None) -> str:
    return "None" if x is None else f"{x:.5f}"


WAVE_PER_BLOCK = 16   # signed samples sent per 100 ms block for the oscilloscope


def _downsample_wave(data: np.ndarray, n: int = WAVE_PER_BLOCK) -> list[float]:
    """Decimate a block of signed float samples to ``n`` points for the
    dashboard's oscilloscope trace (evenly spaced actual sample values,
    not an RMS envelope)."""
    if data.size == 0:
        return []
    idx = np.linspace(0, data.size - 1, n).astype(int)
    return [round(float(v), 3) for v in data[idx]]


def _resample_to(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float audio to ``dst_rate``. Some mics (e.g. a USB
    camera) can't capture at 16 kHz directly, so we grab their native rate
    and convert here for whisper. Uses scipy's polyphase filter when
    available (anti-aliased); falls back to dependency-free linear interp."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(src_rate), int(dst_rate))
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
    except Exception:
        n = int(round(audio.size * dst_rate / src_rate))
        if n <= 0:
            return audio
        x = np.linspace(0, audio.size - 1, n)
        return np.interp(x, np.arange(audio.size), audio).astype(np.float32)


class TextInput:
    """Fallback 'STT' that reads from the console (dev / --text mode)."""

    def __init__(self):
        self.closed = False

    def listen(self) -> str | None:
        try:
            text = input("Visitor> ").strip()
        except (EOFError, KeyboardInterrupt):
            self.closed = True
            return None
        return text or None


class MicSTT:
    """Microphone capture with energy VAD + pre-roll, shared by every
    speech-to-text engine. Subclasses implement transcribe(audio)."""

    def __init__(self, cfg: dict):
        # sounddevice needs PortAudio; transcribe-only use (the GUI A/B
        # test, vyoma.bench --wav-dir) must work without a microphone.
        try:
            import sounddevice as sd
        except Exception:
            sd = None

        self.sd = sd
        self.cfg = cfg
        self.sample_rate = cfg["sample_rate"]
        # Input device: a name substring (matched by sounddevice) or None
        # for the system default. Pinning it stops the mic from falling back
        # to, say, an unplugged headset jack that only captures silence.
        self.input_device = (cfg.get("input_device") or "").strip() or None
        # Capture rate: prefer 16 kHz, but some devices (USB camera mics)
        # only offer 44.1/48 kHz — capture at their native rate and resample.
        self.capture_rate = self._resolve_capture_rate()
        if self.input_device:
            logger.info("Microphone input device: %r (capture %d Hz)",
                        self.input_device, self.capture_rate)
        self.diagnostic = bool(cfg.get("diagnostic", False))
        self._diag: dict = {}  # stats from the last _record_utterance()

    def _resolve_capture_rate(self) -> int:
        """The rate we can actually open the mic at. Returns the target rate
        if the device supports it, else the device's native default rate."""
        if self.sd is None:
            return self.sample_rate
        try:
            self.sd.check_input_settings(
                device=self.input_device, channels=1,
                dtype="float32", samplerate=self.sample_rate)
            return self.sample_rate
        except Exception:
            try:
                info = self.sd.query_devices(self.input_device, "input")
                native = int(round(info["default_samplerate"]))
                logger.info("Mic can't do %d Hz; capturing at %d Hz",
                            self.sample_rate, native)
                return native or self.sample_rate
            except Exception:
                return self.sample_rate

    def listen(self) -> str | None:
        audio = self._record_utterance()
        if audio is None:
            if self.diagnostic:
                self._log_diag(None, None)
            return None
        text = self.transcribe(audio)
        logger.info("Heard: %s", text)
        if self.diagnostic:
            self._log_diag(audio, text)
        return text or None

    def _log_diag(self, audio: np.ndarray | None, text: str | None) -> None:
        """Persist a per-utterance diagnostic record: the noise floor and
        threshold the VAD settled on, the RMS envelope, whether the VAD ever
        detected speech (``started``) or ran to ``max_utterance_seconds``
        (``timed_out``), the recorded WAV, and the transcript. Lets you prove
        in situ whether a miss is the VAD (never started / captured noise) or
        the STT engine (clean audio in, empty text out)."""
        d = self._diag
        out_dir = Path(self.cfg.get("diagnostic_dir") or "data/diagnostics")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            # Millisecond suffix so back-to-back utterances never collide.
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) \
                + f".{int(now * 1000) % 1000:03d}"
            wav_name = ""
            if audio is not None and len(audio):
                wav_name = f"{stamp}.wav"
                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(str(out_dir / wav_name), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.sample_rate)
                    w.writeframes(pcm.tobytes())
            line = (
                f"{stamp}\t"
                f"started={d.get('started')}\t"
                f"timed_out={d.get('timed_out')}\t"
                f"base_threshold={d.get('base_threshold'):.5f}\t"
                f"noise_floor={_fmt(d.get('noise_floor'))}\t"
                f"threshold={_fmt(d.get('threshold'))}\t"
                f"rms_min={_fmt(d.get('rms_min'))}\t"
                f"rms_mean={_fmt(d.get('rms_mean'))}\t"
                f"rms_max={_fmt(d.get('rms_max'))}\t"
                f"seconds={d.get('seconds', 0.0):.2f}\t"
                f"wav={wav_name}\t"
                f"text={text!r}\n"
            )
            with open(out_dir / "diagnostics.tsv", "a") as f:
                f.write(line)
            logger.info("[diag] %s", line.strip())
        except Exception as e:  # diagnostics must never break the receptionist
            logger.warning("diagnostic logging failed: %s", e)

    def transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    def _record_utterance(self) -> np.ndarray | None:
        """Record until trailing silence (energy VAD).

        With ``noise_adaptive`` on, the speech threshold tracks the room's
        ambient noise floor instead of using the fixed ``energy_threshold``.
        The first ``noise_calibration_seconds`` are spent measuring the room
        (no speech is expected yet — the receptionist has just finished its
        own prompt): the floor is the median of those blocks, robust to a
        stray bump. A block then counts as speech only when its energy rises
        ``noise_margin``× above that floor, and the floor keeps drifting
        slowly on the quiet blocks so it follows the room (fans, corridor
        chatter) without chasing brief lulls. ``energy_threshold`` stays an
        absolute lower bound, so a silent room never becomes hair-trigger.
        """
        from collections import deque

        block = int(self.capture_rate * 0.1)  # 100 ms blocks
        silence_blocks = int(self.cfg["silence_seconds"] / 0.1)
        max_blocks = int(self.cfg["max_utterance_seconds"] / 0.1)
        base_threshold = self.cfg["energy_threshold"]
        adaptive = bool(self.cfg.get("noise_adaptive", False))
        margin = float(self.cfg.get("noise_margin", 3.0))
        calib_blocks = max(1, round(
            float(self.cfg.get("noise_calibration_seconds", 0.5)) / 0.1))
        # Keep a short pre-roll so the soft first consonant that hasn't
        # crossed the energy threshold yet isn't clipped off.
        pre_roll: deque[np.ndarray] = deque(maxlen=3)  # 300 ms

        chunks: list[np.ndarray] = []
        calib: list[float] = []               # ambient RMS during calibration
        threshold = base_threshold            # until calibration completes
        noise_floor = None                    # set once calibration completes
        rms_all: list[float] = []             # full envelope (for diagnostics)
        quiet = 0
        started = False
        timed_out = True                      # cleared if we break on silence
        if self.sd is None:
            logger.error("sounddevice is not installed — no microphone")
            return None
        print("… listening")
        events.mic_reset()
        try:
            stream_cm = self.sd.InputStream(
                samplerate=self.capture_rate, channels=1, dtype="float32",
                blocksize=block, device=self.input_device)
        except Exception as e:
            # A bad/absent mic must never crash the receptionist — skip this
            # turn so the conversation loop can recover on the next one.
            logger.error("Cannot open microphone %r: %s",
                         self.input_device or "default", e)
            return None
        with stream_cm as stream:
            for i in range(max_blocks):
                data, _ = stream.read(block)
                data = data[:, 0]
                rms = float(np.sqrt(np.mean(data ** 2)))
                rms_all.append(rms)
                # Calibration window: sample the room, don't look for speech
                # yet (so loud ambient can't false-start the recording).
                if adaptive and i < calib_blocks:
                    calib.append(rms)
                    events.mic_level(rms, False, _downsample_wave(data))
                    pre_roll.append(data.copy())
                    if i == calib_blocks - 1:
                        noise_floor = float(np.median(calib))
                        threshold = max(base_threshold, noise_floor * margin)
                    continue
                # Feed the dashboard's live audio graph (fresh timestamp =
                # "currently listening"); active = visitor is speaking now.
                events.mic_level(rms, rms >= threshold, _downsample_wave(data))
                if rms >= threshold:
                    if not started:
                        chunks.extend(pre_roll)
                        pre_roll.clear()
                    started = True
                    quiet = 0
                elif started:
                    quiet += 1
                    if quiet >= silence_blocks:
                        timed_out = False
                        break
                # Drift the floor on quiet blocks only, so a visitor's voice
                # never inflates it and the threshold follows the room.
                if adaptive and rms < threshold:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms
                    threshold = max(base_threshold, noise_floor * margin)
                if started:
                    chunks.append(data.copy())
                else:
                    pre_roll.append(data.copy())
        if self.diagnostic:
            self._diag = {
                "started": started,
                # A real utterance ends on trailing silence; running to the
                # full max_utterance_seconds usually means the VAD never saw
                # the room go quiet — i.e. it captured continuous noise.
                "timed_out": timed_out and started,
                "base_threshold": base_threshold,
                "noise_floor": noise_floor,
                "threshold": threshold,
                "rms_min": min(rms_all) if rms_all else None,
                "rms_mean": float(np.mean(rms_all)) if rms_all else None,
                "rms_max": max(rms_all) if rms_all else None,
                "seconds": len(rms_all) * 0.1,
            }
        if not chunks:
            return None
        audio = np.concatenate(chunks)
        return _resample_to(audio, self.capture_rate, self.sample_rate)

class WhisperSTT(MicSTT):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from faster_whisper import WhisperModel  # deferred heavy import

        logger.info("Loading whisper model '%s' ...", cfg["model"])
        self.model = WhisperModel(
            cfg["model"], device=cfg["device"],
            compute_type=cfg["compute_type"],
            # Store models inside the project (config
            # whisper_download_root) — system cache cleaners purge
            # ~/.cache when the disk is nearly full.
            download_root=cfg.get("whisper_download_root") or None,
            # The GUI's A/B test sets this so a missing model fails fast
            # instead of blocking on a download.
            local_files_only=bool(cfg.get("local_files_only")))

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.get("language", "en"),
            initial_prompt=self.cfg.get("initial_prompt") or None,
            hotwords=self.cfg.get("hotwords") or None,
            beam_size=int(self.cfg.get("beam_size", 5)),
            # Each utterance is independent; carrying context across
            # them makes whisper repeat/hallucinate on short phrases.
            condition_on_previous_text=False,
            vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


class IndicWhisperSTT(WhisperSTT):
    """AI4Bharat IndicWhisper — Whisper fine-tuned on Indian languages and
    Indian-accented English. It runs through faster-whisper (CTranslate2),
    just like WhisperSTT, so the model must be in CT2 format: either a local
    directory (kept inside the project, like the vosk/whisper models) or a
    CTranslate2 HuggingFace repo id. Convert an HF transformers checkpoint
    once with, e.g.:

        pip install transformers ctranslate2
        ct2-transformers-converter --model vasista22/whisper-hindi-small \\
            --output_dir models/indic-whisper --quantization int8

    then point `stt.indic_whisper_model` at models/indic-whisper.
    """

    def __init__(self, cfg: dict):
        model = (cfg.get("indic_whisper_model") or "").strip()
        if not model:
            raise RuntimeError(
                "stt.indic_whisper_model is not set in config.yaml — point it "
                "at a CTranslate2 IndicWhisper model directory or repo id "
                "(see IndicWhisperSTT for a conversion command)")
        # A local directory is used as-is (resolved absolutely, and a relative
        # path is tried against the project root so it works whatever the
        # service's CWD is); anything else is passed through to faster-whisper
        # as a HuggingFace repo id.
        from .config import PROJECT_ROOT
        p = Path(model)
        if not p.is_absolute() and not p.is_dir() and (PROJECT_ROOT / model).is_dir():
            p = PROJECT_ROOT / model
        if p.is_dir():
            model = str(p.resolve())
        scfg = dict(cfg)
        scfg["model"] = model
        super().__init__(scfg)


class VoskSTT(MicSTT):
    """Vosk (Kaldi) engine — the `vosk-model-en-in-*` models are trained
    on Indian English and run much faster than whisper on a CPU."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import vosk
        from pathlib import Path

        vosk.SetLogLevel(-1)
        name = cfg.get("vosk_model", "vosk-model-en-in-0.5")
        cache = Path.home() / ".cache" / "vosk"
        small = cache / "vosk-model-small-en-in-0.4"
        if Path(name).is_dir():
            path = Path(name)
        elif (cache / name).is_dir():
            path = cache / name
        elif small.is_dir():
            # Stopgap while the configured model is still downloading —
            # never block startup on the vosk library's own downloader.
            logger.warning("Vosk model '%s' not downloaded yet; using the "
                           "small en-IN model for now", name)
            path = small
        else:
            raise RuntimeError(
                f"Vosk model '{name}' not found. Download it into "
                f"{cache} from https://alphacephei.com/vosk/models")
        logger.info("Loading vosk model %s ...", path.name)
        self.model = vosk.Model(str(path))
        self._vosk = vosk

    def transcribe(self, audio: np.ndarray) -> str:
        import json

        rec = self._vosk.KaldiRecognizer(self.model, self.sample_rate)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        rec.AcceptWaveform(pcm.tobytes())
        return json.loads(rec.FinalResult()).get("text", "").strip()


class DeepgramSTT(MicSTT):
    """Deepgram cloud STT (nova-2, en-IN). Needs internet — visitor
    audio is sent to Deepgram's servers. Failures are logged and
    treated as 'heard nothing' so the receptionist keeps running."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dg = cfg.get("deepgram") or {}
        self.api_key = dg.get("api_key", "")
        if not self.api_key:
            raise RuntimeError("deepgram.api_key is not set in config.yaml")
        self.model = dg.get("stt_model", "nova-2")
        self.language = dg.get("language", "en-IN")

    def transcribe(self, audio: np.ndarray) -> str:
        import io
        import urllib.parse
        import wave

        import requests

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(
                (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        params = [("model", self.model), ("language", self.language),
                  ("smart_format", "true")]
        for word in (self.cfg.get("hotwords") or "").split():
            params.append(("keywords", f"{word}:2"))
        url = ("https://api.deepgram.com/v1/listen?"
               + urllib.parse.urlencode(params))
        try:
            resp = requests.post(
                url, data=buf.getvalue(), timeout=20,
                headers={"Authorization": f"Token {self.api_key}",
                         "Content-Type": "audio/wav"})
            resp.raise_for_status()
            alt = resp.json()["results"]["channels"][0]["alternatives"][0]
            return alt["transcript"].strip()
        except Exception as e:
            logger.error("Deepgram STT failed: %s", e)
            return ""


def create_stt(cfg: dict):
    engine = cfg["stt"]["engine"]
    if engine == "faster-whisper":
        return WhisperSTT(cfg["stt"])
    if engine == "indic-whisper":
        return IndicWhisperSTT(cfg["stt"])
    if engine == "vosk":
        return VoskSTT(cfg["stt"])
    if engine == "deepgram":
        scfg = dict(cfg["stt"])
        scfg["deepgram"] = cfg.get("deepgram") or {}
        return DeepgramSTT(scfg)
    return TextInput()
