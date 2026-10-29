# Vyoma — Offline Agentic Receptionist Robot

An AI-powered receptionist that runs **entirely offline** on modest
hardware (4 GB GPU class). It greets visitors, answers questions from
your organisation's own documents and website, logs visits, and
notifies staff — with no cloud APIs, no subscriptions, and no visitor
data leaving the premises.

Built per `Receptionist_Robot_Synopsis.docx` (V-Robotics).

## Quick start

```bash
cd ~/esp6/receptionist/vyoma

.venv/bin/python -m vyoma              # voice conversation (mic + speaker)
.venv/bin/python -m vyoma --camera     # greet when a face is detected, loop per visitor
.venv/bin/python -m vyoma --text       # keyboard chat, for development
.venv/bin/python -m vyoma.gui          # control panel → http://127.0.0.1:7860
```

Teach it your organisation:

```bash
.venv/bin/python -m vyoma.ingest brochure.pdf policies.docx   # documents
.venv/bin/python -m vyoma.ingest https://www.yourcompany.com  # whole website
.venv/bin/python -m vyoma.ingest https://www.yourcompany.com --single  # one page only
.venv/bin/python -m vyoma.ingest --list                       # show sources
.venv/bin/python -m vyoma.ingest --remove web-yourcompany.md  # forget one
.venv/bin/python -m vyoma.ingest --clear                      # forget all ingested
```

## Control panel (GUI)

`python -m vyoma.gui` serves a fully offline dashboard at
`http://127.0.0.1:7860` (use `--host 0.0.0.0` to open it from a phone
on the same network):

- **Dashboard** — start/stop the receptionist (voice or camera mode),
  watch a live feed of what is happening inside (speech heard, FACTS
  retrieved, tool calls, replies), and a webcam preview with
  face-detection boxes right next to it; if the receptionist owns the
  camera, the preview relays the frames she is analysing.
- **Test chat** — keyboard conversation with the same agent, for
  development and noisy-hall demos.
- **Speech test** — A/B the speech engines (Whisper vs Vosk
  Indian-English) on one recording of your own voice or an uploaded
  WAV; pick the winner in Settings → Speech engine.
- **Knowledge** — upload documents, crawl the organisation website
  (set the domain once in Settings), list and delete sources.
- **Reports** — generate/open the visitors report, download Excel.
- **Settings** — organisation identity (auto-detected from the
  knowledge base), AI model, voice, speech engine, staff webhook,
  camera default; edits `config.yaml` in place.
- **Avatar** (`/avatar`) — the visitor-facing animated face: an SVG
  robot that blinks and idles, shows a pulsing ring while listening,
  and lip-syncs Vyoma's replies with viseme mouth shapes driven by the
  real audio envelope, with the spoken text as a caption. The kiosk
  Chrome on the robot's screen points here
  (`deploy/vyoma-kiosk.sh`); the dashboard stays reachable over the
  LAN.

Internals: the receptionist appends events to `data/events.jsonl`
(see `vyoma/events.py`) and the GUI tails that file, so the live view
works no matter which process is doing the talking.

## What it does

A visitor walks up. The camera notices them; if they have visited
before (and enrolled), Vyoma greets them by name. They speak; Vyoma
transcribes, reasons over your organisation's knowledge base, and
answers aloud in one or two natural sentences. If the visitor came to
meet someone, Vyoma asks who they are, registers them, logs the visit,
and notifies the staff member. Everything is stored locally in SQLite.

## Components

| Stage | Technology | Runs on |
|---|---|---|
| Speech-to-text | faster-whisper `small` (multilingual, Indian-English friendly) + energy VAD | CPU |
| Reasoning | Qwen 2.5 3B (4-bit) via Ollama, tool calling | GPU (resident) |
| Grounding (RAG) | `nomic-embed-text` embeddings over `knowledge/*.md` | CPU |
| Text-to-speech | Piper `en_US-lessac-medium` | CPU |
| Vision | OpenCV Haar cascade; optional dlib `face_recognition` | CPU |
| Data | SQLite: visitors, visits, notifications, face embeddings | disk |

The LLM stays loaded in VRAM (`keep_alive: -1`); the lighter components
run on the CPU so nothing competes for the 4 GB budget.

## Architecture

```
camera ──▶ FaceWatcher ──┐
                         ▼
mic ──▶ WhisperSTT ──▶ ReceptionistAgent ──▶ PiperTTS ──▶ speaker
                         │        ▲
              tool calls │        │ FACTS (top-k chunks)
                         ▼        │
        Tools(log_visit, capture_visitor,   KnowledgeBase
              notify_staff, end_conversation)  (knowledge/*.md)
                         │
                         ▼
                     SQLite (data/vyoma.db)
```

Each turn: transcribe → embed the query → retrieve top-k FAQ chunks →
LLM (receptionist persona + FACTS + tools) → execute any tool calls →
speak the reply. The agent decides on its own when to log a visit,
notify staff, or end the conversation.

## Project layout

```
vyoma/
├── config.yaml          # models, thresholds, org name, webhook
├── knowledge/           # markdown knowledge base (edit or ingest)
├── voices/              # Piper voice models (several installed)
├── samples/             # short WAVs to audition each voice
├── data/                # SQLite DB + embedding cache (created at runtime)
└── vyoma/
    ├── main.py          # event loop / entry point
    ├── agent.py         # agentic controller (LLM + tool loop)
    ├── tools.py         # log_visit, capture_visitor, notify_staff, ...
    ├── kb.py            # RAG: chunking, embeddings, retrieval
    ├── ingest.py        # document / website → knowledge base CLI
    ├── stt.py           # microphone + VAD + faster-whisper
    ├── tts.py           # Piper (espeak/console fallback)
    ├── visemes.py       # lip-sync timeline + envelope for the avatar
    ├── bench.py         # latency/accuracy benchmark (python -m vyoma.bench)
    ├── avatar.html      # visitor-facing animated face (served at /avatar)
    ├── vision.py        # presence detection + face recognition
    └── db.py            # SQLite store
```

## Setup from scratch

```bash
# 1. Ollama (https://ollama.com) and models
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Piper voice (one-time download)
.venv/bin/python -m piper.download_voices en_US-lessac-medium --data-dir voices

# 4. (optional) espeak-ng as a TTS fallback
sudo apt install espeak-ng
```

The whisper STT model downloads automatically on first voice run.

## Configuration

Everything lives in `config.yaml`. It is not tracked in git (it can hold
API keys) — create it from the template first:

```bash
cp config.example.yaml config.yaml
```


- **organisation** — name and assistant name used in the persona.
- **llm** — primary (`qwen2.5:3b`) and fallback (`llama3.2:3b`) models,
  temperature, context size.
- **knowledge_base** — embedding model, top-k, similarity threshold.
- **stt** — whisper model size (`small` by default: multilingual,
  handles Indian-accented English well; `base.en` is faster but less
  accent-tolerant), spoken language, a vocabulary hint
  (`initial_prompt`) that biases recognition toward your organisation's
  names, and VAD silence/energy thresholds. Set `engine: indic-whisper`
  with a CTranslate2 IndicWhisper model in `indic_whisper_model` to use
  AI4Bharat's Whisper fine-tuned on Indian languages.
- **tts** — Piper voice and playback command (see [Voices](#voices)).
- **vision** — camera index, recognition tolerance, presence time.
- **notifications.webhook_url** — set to e.g. an https://ntfy.sh topic
  so `notify_staff` reaches a phone; otherwise notifications are
  recorded in the database only.

## Knowledge base

The receptionist only states facts found in `knowledge/*.md` (injected
as FACTS context each turn). Three ways to fill it:

1. `python -m vyoma.ingest <files>` — PDF, DOCX, TXT, MD.
2. `python -m vyoma.ingest <url>` — your company website. By default
   it crawls the whole site: every same-site link is followed,
   subdomains included (e.g. `blog.yourcompany.com`), up to 5 link
   levels deep. Use `--single` for just one page, `--max-pages N` to
   cap the total, or `--depth N` to change how deep it goes.
3. Edit the markdown by hand.

The embedding cache rebuilds automatically whenever the files change.

## Voices

Four Piper voices are installed in `voices/`; audition clips for the
alternatives are in `samples/` (`aplay samples/<name>.wav`):

| Voice | Language | Notes |
|---|---|---|
| `en_US-lessac-medium` | English (US) | default |
| `en_US-hfc_female-medium` | English (US) | female |
| `en_GB-cori-medium` | English (UK) | female |
| `hi_IN-priyamvada-medium` | Hindi | female |

To switch, set `tts.voice` in `config.yaml` to one of the names above.
More voices can be added with:

```bash
.venv/bin/python -m piper.download_voices <voice-name> --data-dir voices
```

(Full catalogue: https://github.com/OHF-Voice/piper1-gpl — note the STT
model is `base.en`, so a Hindi voice only affects the spoken output,
not what the receptionist can understand.)

## Visitor data

`data/vyoma.db` (SQLite):

- `visitors` — name, phone, company, optional face embedding.
- `visits` — who came, purpose, whom to meet, timestamp.
- `notifications` — staff alerts and delivery status.

Inspect with `sqlite3 data/vyoma.db 'SELECT * FROM visits;'`, or
generate the visitors report:

```bash
.venv/bin/python -m vyoma.report            # writes data/visitors_report.html
.venv/bin/python -m vyoma.report --xlsx visitors.xlsx   # also save Excel directly
```

The report is a single offline HTML page (search + date filters) with an
**Export to Excel** button; the .xlsx workbook (Visits / Visitors /
Notifications sheets) is embedded in the page, so no internet or extra
software is needed.

## Face recognition by name (optional)

Presence detection works out of the box. To greet returning visitors by
name, install `face_recognition` (needs cmake + a C++ toolchain to
build dlib), then enroll visitors via `FaceWatcher.enroll()` — the
embedding is stored locally in SQLite. Enrolment should be
consent-based; embeddings never leave the machine.

## Hardware validated on

RTX 2050 (4 GB VRAM) laptop, webcam, sof-hda-dsp microphone — the
repurposed-laptop class targeted by the synopsis. End-to-end turn
latency is a few seconds.

## Avatar lip-sync (phase 4)

When any TTS engine speaks, `tts.py` runs the synthesized WAV through
`visemes.py`: the audio's RMS envelope (25 fps) says how far the mouth
opens, and a letter→viseme timeline (Preston Blair mouth classes: AI,
E, O, U, MBP, FV, L, TS) says which shape it takes, scaled to the true
audio duration. The result is published to `data/speak_state.json`
(same atomic-overwrite pattern as the mic-level file) a moment before
playback starts; the `/avatar` page polls it (plus `/api/mic` for the
listening state) and animates the SVG face, correcting for client
clock skew with the server timestamp in `/api/speak`.

## Benchmarking (phase 5)

```bash
.venv/bin/python -m vyoma.bench                 # configured engines
.venv/bin/python -m vyoma.bench --engines all   # compare STT engines
.venv/bin/python -m vyoma.bench --wav-dir data/bench_audio
```

Measures each stage of a turn — TTS synthesis (latency + real-time
factor), STT (latency, RTF, word error rate) and agent turns over a
scripted conversation against a throwaway database — then prints the
end-to-end response gap the medians add up to, and saves a JSON
snapshot to `data/bench-*.json` for before/after comparison when
tuning. STT test audio is a TTS round-trip of known sentences by
default (offline, repeatable, but a synthetic voice — expect better
WER than a real mic); point `--wav-dir` at `x.wav` + `x.txt`
reference pairs (e.g. utterances saved by `stt.diagnostic` mode) to
measure your real microphone and room.

## Roadmap (synopsis phases)

- [x] STT, LLM, TTS, vision modules
- [x] Agentic controller with tool calling
- [x] Knowledge grounding (documents + website ingestion)
- [x] Visitor logging and staff notification
- [x] Animated avatar with viseme lip-sync (phase 4) — `/avatar` page
- [x] Latency/accuracy benchmarking and tuning (phase 5) — `python -m vyoma.bench`
