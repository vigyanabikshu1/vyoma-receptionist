"""Latency / accuracy benchmark for the receptionist stack (phase 5).

Measures each stage of a conversation turn separately, so tuning work
(model size, beam width, num_ctx…) can be judged with numbers instead
of feel:

- TTS:  synthesis time per sentence and real-time factor (RTF);
- STT:  transcription latency, RTF and word error rate (WER). Test
  audio is a TTS round-trip of known sentences by default — fully
  offline and repeatable — or your own recordings via ``--wav-dir``
  (pairs of ``x.wav`` + ``x.txt`` reference transcript, e.g. utterances
  saved by the ``stt.diagnostic`` mode with hand-written references);
- LLM:  agent turn time over a scripted visitor conversation (RAG
  search + reasoning + tool calls), against a throwaway database so the
  real visitor log is untouched;
- E2E:  the visitor-perceived response gap those medians add up to.

Usage (from the project directory, with the venv python):
    python -m vyoma.bench                    # configured STT engine only
    python -m vyoma.bench --engines all      # compare every local engine
    python -m vyoma.bench --engines faster-whisper,vosk
    python -m vyoma.bench --wav-dir data/bench_audio
    python -m vyoma.bench --skip-llm --skip-tts

Results are printed as tables and appended to ``data/bench-*.json`` for
before/after comparison. Safe to run while the receptionist service is
up (it never touches the microphone or camera), though a busy GPU/CPU
will inflate the numbers — stop the service for clean measurements.
"""

import argparse
import json
import logging
import re
import statistics
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("vyoma.bench")

# Receptionist-domain sentences (the {org} placeholder gets the
# configured organisation name, so its hotwords are exercised too).
SENTENCES = [
    "Good morning, I want to meet the principal.",
    "What time does the school open in the morning?",
    "My name is Ramesh Kumar and I am here for an admission enquiry.",
    "Can you tell me the phone number of the office?",
    "Is this {org}?",
    "Thank you very much for your help. Goodbye.",
]

# Scripted visitor turns for the agent benchmark.
CONVERSATION = [
    "What time does the school open?",
    "I would like to meet the principal please.",
    "My name is Benchmark Visitor, I am here for an admission enquiry.",
    "Thank you, goodbye.",
]

STT_ENGINES = ("faster-whisper", "indic-whisper", "vosk", "deepgram")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level edit distance / reference length."""
    ref, hyp = _norm_words(reference), _norm_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        row = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            row[j] = min(prev[j] + 1,          # deletion
                         row[j - 1] + 1,       # insertion
                         prev[j - 1] + (rw != hw))  # substitution
        prev = row
    return prev[-1] / len(ref)


def _wav_bytes_to_16k(wav_bytes: bytes):
    """WAV bytes → mono float32 at 16 kHz (what the engines expect)."""
    from .stt import _resample_to
    from .visemes import _wav_to_mono
    audio, rate = _wav_to_mono(wav_bytes)
    return _resample_to(audio, rate, 16000)


def _table(title: str, headers: list[str], rows: list[list]) -> str:
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    out = [f"\n== {title} ==", line(headers), line(["-" * w for w in widths])]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _median(xs):
    return round(statistics.median(xs), 2) if xs else None


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------

def bench_tts(cfg: dict, sentences: list[str]) -> dict:
    """Synthesis latency + RTF for the configured voice."""
    from .tts import synth_wav
    from .visemes import _wav_to_mono
    results = []
    synth_wav(cfg, "Warming up.")          # load the voice off the clock
    for text in sentences:
        t0 = time.perf_counter()
        wav = synth_wav(cfg, text)
        secs = time.perf_counter() - t0
        if not wav:
            results.append({"text": text, "error": "no TTS engine"})
            continue
        audio, rate = _wav_to_mono(wav)
        dur = len(audio) / rate
        results.append({"text": text, "synth_s": round(secs, 2),
                        "audio_s": round(dur, 2),
                        "rtf": round(secs / dur, 3) if dur else None})
    ok = [r for r in results if "error" not in r]
    return {"engine": cfg["tts"]["engine"], "voice": cfg["tts"]["voice"],
            "median_synth_s": _median([r["synth_s"] for r in ok]),
            "median_rtf": _median([r["rtf"] for r in ok]),
            "utterances": results}


def _make_stt(kind: str, cfg: dict):
    from .stt import DeepgramSTT, IndicWhisperSTT, VoskSTT, WhisperSTT
    scfg = dict(cfg["stt"])
    scfg["local_files_only"] = True        # never download mid-benchmark
    if kind == "faster-whisper":
        return WhisperSTT(scfg)
    if kind == "indic-whisper":
        return IndicWhisperSTT(scfg)
    if kind == "vosk":
        return VoskSTT(scfg)
    if kind == "deepgram":
        scfg["deepgram"] = cfg.get("deepgram") or {}
        return DeepgramSTT(scfg)
    raise ValueError(f"unknown engine {kind!r}")


def _stt_samples(cfg: dict, wav_dir: str | None) -> list[tuple[str, "object", str]]:
    """(name, audio16k, reference) triples to transcribe."""
    samples = []
    if wav_dir:
        for wav in sorted(Path(wav_dir).glob("*.wav")):
            ref = wav.with_suffix(".txt")
            if not ref.exists():
                logger.warning("skipping %s (no %s reference)", wav.name, ref.name)
                continue
            samples.append((wav.name, _wav_bytes_to_16k(wav.read_bytes()),
                            ref.read_text().strip()))
        if not samples:
            raise SystemExit(f"no wav+txt pairs found in {wav_dir}")
    else:
        from .tts import synth_wav
        for text in SENTENCES:
            text = text.format(org=cfg["organisation"]["name"])
            wav = synth_wav(cfg, text)
            if not wav:
                raise SystemExit("no TTS engine available to make test audio; "
                                 "use --wav-dir with your own recordings")
            samples.append((f"tts:{text[:34]}…", _wav_bytes_to_16k(wav), text))
    return samples


def bench_stt(cfg: dict, engines: list[str], wav_dir: str | None) -> dict:
    samples = _stt_samples(cfg, wav_dir)
    source = wav_dir or "TTS round-trip (synthetic voice — expect lower WER " \
                        "than a real reception mic)"
    out = {"source": source, "engines": []}
    for kind in engines:
        try:
            t0 = time.perf_counter()
            engine = _make_stt(kind, cfg)
            load_s = round(time.perf_counter() - t0, 2)
        except Exception as e:
            out["engines"].append({"engine": kind, "error": str(e)})
            continue
        utts, lats, rtfs, wers = [], [], [], []
        engine.transcribe(samples[0][1])   # warm-up off the clock
        for name, audio, ref in samples:
            dur = len(audio) / 16000
            t0 = time.perf_counter()
            try:
                text = engine.transcribe(audio)
            except Exception as e:
                utts.append({"sample": name, "error": str(e)})
                continue
            secs = time.perf_counter() - t0
            e = wer(ref, text)
            lats.append(secs); rtfs.append(secs / dur if dur else 0); wers.append(e)
            utts.append({"sample": name, "seconds": round(secs, 2),
                         "audio_s": round(dur, 2), "wer": round(e, 3),
                         "text": text})
        out["engines"].append({
            "engine": kind, "load_s": load_s,
            "median_latency_s": _median(lats),
            "median_rtf": _median(rtfs),
            "mean_wer": round(sum(wers) / len(wers), 3) if wers else None,
            "utterances": utts})
        del engine                          # free the model before the next
    return out


def bench_llm(cfg: dict) -> dict:
    """Agent turn latency on a scripted conversation, throwaway DB."""
    from .agent import ReceptionistAgent
    from .db import Database
    from .kb import KnowledgeBase
    from .tools import Tools

    kb = KnowledgeBase(cfg["knowledge_base"]["dir"],
                       cfg["knowledge_base"]["embedding_model"],
                       cfg["data"]["kb_cache"],
                       cfg["knowledge_base"]["top_k"],
                       cfg["knowledge_base"]["min_score"])
    t0 = time.perf_counter()
    n = len(kb.search(CONVERSATION[0]))
    kb_s = round(time.perf_counter() - t0, 3)

    with tempfile.TemporaryDirectory() as tmp:
        tools = Tools(Database(str(Path(tmp) / "bench.db")), "")
        t0 = time.perf_counter()
        agent = ReceptionistAgent(cfg, kb, tools)
        agent.greet(None)
        init_s = round(time.perf_counter() - t0, 2)

        turns = []
        for text in CONVERSATION:
            t0 = time.perf_counter()
            reply = agent.respond(text)
            turns.append({"visitor": text,
                          "seconds": round(time.perf_counter() - t0, 2),
                          "reply": reply})
    return {"model": agent.model, "kb_search_s": kb_s,
            "kb_hits": n, "agent_init_s": init_s,
            "median_turn_s": _median([t["seconds"] for t in turns]),
            "turns": turns}


def record_samples(cfg: dict, out_dir: str) -> None:
    """Interactively record the benchmark sentences from the real mic.

    Prompts each sentence, records with the same VAD the receptionist
    uses, and saves ``uttNN.wav`` + ``uttNN.txt`` pairs ready for
    ``--wav-dir``. Because you are told what to say, the reference
    transcript is exact. Stop the receptionist first — it owns the mic.
    """
    import wave

    import numpy as np

    from .stt import MicSTT

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mic = MicSTT(cfg["stt"])
    sentences = [s.format(org=cfg["organisation"]["name"]) for s in SENTENCES]
    print(f"\nRecording {len(sentences)} sentences to {out}/ — read each "
          "prompt aloud, then pause; silence ends the take.")
    for i, text in enumerate(sentences, 1):
        while True:
            print(f'\n[{i}/{len(sentences)}] Say:  "{text}"')
            audio = mic._record_utterance()
            if audio is None or not len(audio):
                print("  heard nothing — trying again (Ctrl-C to abort)")
                continue
            secs = len(audio) / mic.sample_rate
            if secs < 0.6:
                print(f"  only {secs:.1f}s captured (noise?) — again")
                continue
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(out / f"utt{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(mic.sample_rate)
                w.writeframes(pcm.tobytes())
            (out / f"utt{i:02d}.txt").write_text(text + "\n")
            print(f"  saved utt{i:02d}.wav ({secs:.1f}s)")
            break
    print(f"\nDone. Benchmark them with:\n"
          f"  python -m vyoma.bench --skip-llm --skip-tts "
          f"--engines all --wav-dir {out}")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    if "tts" in results:
        t = results["tts"]
        rows = [[r.get("synth_s"), r.get("audio_s"), r.get("rtf"),
                 r.get("error") or r["text"][:56]] for r in t["utterances"]]
        print(_table(f"TTS — {t['engine']} / {t['voice']}",
                     ["synth s", "audio s", "RTF", "sentence"], rows))
        print(f"median synth {t['median_synth_s']}s, RTF {t['median_rtf']} "
              "(RTF < 1 = faster than real time)")

    if "stt" in results:
        s = results["stt"]
        rows = []
        for e in s["engines"]:
            if "error" in e:
                rows.append([e["engine"], "—", "—", "—", e["error"][:60]])
            else:
                rows.append([e["engine"], e["median_latency_s"],
                             e["median_rtf"],
                             f"{e['mean_wer'] * 100:.1f}%" if e["mean_wer"]
                             is not None else "—", ""])
        print(_table(f"STT — audio: {s['source']}",
                     ["engine", "median s", "RTF", "WER", "note"], rows))

    if "llm" in results:
        l = results["llm"]
        rows = [[t["seconds"], t["visitor"][:34], (t["reply"] or "")[:52]]
                for t in l["turns"]]
        print(_table(f"Agent — {l['model']} "
                     f"(KB search {l['kb_search_s']}s, {l['kb_hits']} hits)",
                     ["turn s", "visitor", "reply"], rows))

    parts = {}
    if results.get("stt", {}).get("engines"):
        ok = [e for e in results["stt"]["engines"] if "error" not in e
              and e["median_latency_s"] is not None]
        if ok:
            parts["hear (STT)"] = min(e["median_latency_s"] for e in ok)
    if "llm" in results:
        parts["think (agent)"] = results["llm"]["median_turn_s"]
    if "tts" in results:
        parts["voice (TTS synth)"] = results["tts"]["median_synth_s"]
    if parts:
        rows = [[f"{v}s", k] for k, v in parts.items()]
        rows.append([f"{round(sum(parts.values()), 2)}s",
                     "estimated silence between visitor and reply"])
        print(_table("End-to-end turn (medians)", ["seconds", "stage"], rows))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the receptionist's STT / LLM / TTS stages")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--engines", default="",
                        help="STT engines, comma separated, or 'all' "
                             f"(default: the configured engine; known: "
                             f"{', '.join(STT_ENGINES)})")
    parser.add_argument("--wav-dir",
                        help="Directory of real recordings: x.wav + x.txt "
                             "reference transcripts (default: synthesize "
                             "test audio with the configured TTS voice)")
    parser.add_argument("--record", metavar="DIR",
                        help="Record the benchmark sentences from the "
                             "microphone into DIR (wav+txt pairs), then exit. "
                             "Stop the receptionist first — it owns the mic.")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s")

    from .config import load_config
    cfg = load_config(args.config)

    if args.record:
        record_samples(cfg, args.record)
        return

    if args.engines == "all":
        engines = [e for e in STT_ENGINES
                   if e != "deepgram" or (cfg.get("deepgram") or {}).get("api_key")]
    elif args.engines:
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    else:
        engines = [cfg["stt"]["engine"]] if cfg["stt"]["engine"] in STT_ENGINES \
            else []

    results: dict = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "config": {"stt_engine": cfg["stt"]["engine"],
                                "stt_model": cfg["stt"]["model"],
                                "llm_model": cfg["llm"]["model"],
                                "tts_voice": cfg["tts"]["voice"]}}
    sentences = [s.format(org=cfg["organisation"]["name"]) for s in SENTENCES]

    if not args.skip_tts:
        print("Benchmarking TTS …")
        results["tts"] = bench_tts(cfg, sentences)
    if not args.skip_stt and engines:
        print(f"Benchmarking STT ({', '.join(engines)}) …")
        results["stt"] = bench_stt(cfg, engines, args.wav_dir)
    if not args.skip_llm:
        print("Benchmarking the agent (needs Ollama running) …")
        try:
            results["llm"] = bench_llm(cfg)
        except Exception as e:
            print(f"  agent benchmark failed: {e}")
            results["llm_error"] = str(e)

    print_report(results)

    out = Path(cfg["data"]["db_path"]).parent / \
        f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
