"""Text-to-speech: Piper (preferred), espeak-ng, or console print."""

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path

from . import events

logger = logging.getLogger(__name__)


def _speak_wav(text: str, wav: bytes, play_command: list[str]):
    """Shared playback path: publish the lip-sync state for the avatar
    page, play the WAV on the speakers, then mark speech finished. The
    avatar must never be able to break speech — analysis errors are
    swallowed."""
    print(f"Vyoma> {text}")
    try:
        from . import visemes
        events.speak_begin(text, visemes.analyze(wav, text))
    except Exception as e:
        logger.debug("lip-sync analysis failed: %s", e)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(wav)
            f.flush()
            subprocess.run(play_command + [f.name], check=False)
    finally:
        events.speak_end()


class ConsoleTTS:
    def speak(self, text: str):
        print(f"Vyoma> {text}")


class EspeakTTS:
    play_command = ["aplay", "-q"]

    def speak(self, text: str):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                subprocess.run(["espeak-ng", "-w", f.name, text], check=True)
                wav = Path(f.name).read_bytes()
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            print(f"Vyoma> {text}")
            subprocess.run(["espeak-ng", text], check=False)
            return
        _speak_wav(text, wav, self.play_command)


class PiperTTS:
    def __init__(self, cfg: dict):
        from piper import PiperVoice  # deferred heavy import

        voices_dir = Path(cfg["voices_dir"])
        model_path = voices_dir / f"{cfg['voice']}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found: {model_path}\n"
                f"Download it with:\n"
                f"  python -m piper.download_voices {cfg['voice']} "
                f"--data-dir {voices_dir}"
            )
        logger.info("Loading piper voice '%s' ...", cfg["voice"])
        self.voice = PiperVoice.load(str(model_path))
        self.play_command = shlex.split(cfg["play_command"])

    def speak(self, text: str):
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self.voice.synthesize_wav(text, wav)
        _speak_wav(text, buf.getvalue(), self.play_command)


class DeepgramTTS:
    """Deepgram Aura cloud voice. Needs internet — replies are sent to
    Deepgram for synthesis. Falls back to a local voice on failure."""

    def __init__(self, cfg: dict, dg: dict, fallback=None):
        self.api_key = dg.get("api_key", "")
        if not self.api_key:
            raise RuntimeError("deepgram.api_key is not set in config.yaml")
        self.model = dg.get("tts_model", "aura-2-asteria-en")
        self.play_command = shlex.split(cfg["play_command"])
        self.fallback = fallback or ConsoleTTS()

    def speak(self, text: str):
        import requests

        try:
            resp = requests.post(
                "https://api.deepgram.com/v1/speak"
                f"?model={self.model}&encoding=linear16"
                "&container=wav&sample_rate=24000",
                json={"text": text}, timeout=20,
                headers={"Authorization": f"Token {self.api_key}"})
            resp.raise_for_status()
        except Exception as e:
            logger.error("Deepgram TTS failed (%s); using local voice", e)
            self.fallback.speak(text)
            return
        _speak_wav(text, resp.content, self.play_command)


_PIPER_VOICES: dict = {}   # model_path -> loaded PiperVoice (cache)


def _piper_voice(model_path: Path):
    """Load a Piper voice once and reuse it across synth calls."""
    key = str(model_path)
    voice = _PIPER_VOICES.get(key)
    if voice is None:
        from piper import PiperVoice
        logger.info("Loading piper voice '%s' for playback ...", model_path.stem)
        voice = PiperVoice.load(str(model_path))
        _PIPER_VOICES[key] = voice
    return voice


def synth_wav(cfg: dict, text: str) -> bytes:
    """Synthesize ``text`` to WAV bytes for the browser to play back.

    Uses the configured engine (deepgram → piper → espeak), degrading
    gracefully. Returns ``b""`` if no audio engine is available. Unlike
    the ``speak()`` methods this never plays on the server's speakers —
    the audio is handed to the caller.
    """
    tts = cfg["tts"]
    engine = tts["engine"]

    if engine == "deepgram":
        dg = cfg.get("deepgram") or {}
        key = dg.get("api_key", "")
        if key:
            try:
                import requests

                model = dg.get("tts_model", "aura-2-asteria-en")
                resp = requests.post(
                    "https://api.deepgram.com/v1/speak"
                    f"?model={model}&encoding=linear16"
                    "&container=wav&sample_rate=24000",
                    json={"text": text}, timeout=20,
                    headers={"Authorization": f"Token {key}"})
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                logger.error("Deepgram TTS failed (%s); using local voice", e)

    if engine in ("deepgram", "piper"):
        try:
            import io
            import wave

            model_path = Path(tts["voices_dir"]) / f"{tts['voice']}.onnx"
            voice = _piper_voice(model_path)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                voice.synthesize_wav(text, wav)
            return buf.getvalue()
        except Exception as e:
            logger.warning("Piper synth unavailable (%s); trying espeak", e)

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            subprocess.run(["espeak-ng", "-w", f.name, text], check=True)
            return Path(f.name).read_bytes()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning("espeak synth unavailable (%s)", e)

    return b""


def create_tts(cfg: dict):
    engine = cfg["tts"]["engine"]
    if engine == "deepgram":
        try:
            fallback = None
            try:
                fallback = PiperTTS(cfg["tts"])   # offline safety net
            except Exception:
                pass
            return DeepgramTTS(cfg["tts"], cfg.get("deepgram") or {},
                               fallback)
        except Exception as e:
            logger.warning("Deepgram TTS unavailable (%s); falling back", e)
            engine = "piper"
    if engine == "piper":
        try:
            return PiperTTS(cfg["tts"])
        except Exception as e:
            logger.warning("Piper unavailable (%s); falling back", e)
    if engine in ("piper", "espeak"):
        try:
            subprocess.run(["espeak-ng", "--version"],
                           capture_output=True, check=True)
            return EspeakTTS()
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("espeak-ng not found; printing replies instead")
    return ConsoleTTS()
