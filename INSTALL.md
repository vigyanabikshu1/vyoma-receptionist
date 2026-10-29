# Vyoma Receptionist — Installation Manual

Version 1.0 · July 2026

Vyoma is an offline AI receptionist. She greets visitors, answers
questions from your organisation's own documents and website, logs
visits, and notifies staff. After installation **no internet connection
is required** — all AI models run on the reception computer and no
visitor data ever leaves it.

This manual takes you from an empty computer to a working receptionist.
Allow 30–60 minutes; most of it is model downloads.

## 1. What you need

**Hardware**

- A PC or laptop with a 4 GB (or better) NVIDIA graphics card.
  A repurposed laptop of the RTX 2050 class works well.
  (Without a GPU Vyoma still runs, just with slower replies.)
- Webcam (built-in is fine) — for detecting visitors.
- Microphone (built-in is fine; a USB directional microphone is
  better in noisy lobbies).
- Speakers.
- At least 15 GB free disk space.

**Software**

- Ubuntu Linux 22.04 or newer (other distributions work; commands
  below are for Ubuntu).
- Internet connection **during installation only**.

## 2. Install system packages

Open a terminal and run:

```
sudo apt update
sudo apt install -y python3-venv python3-pip alsa-utils espeak-ng
```

`alsa-utils` provides the audio player; `espeak-ng` is an emergency
fallback voice.

If the computer has an NVIDIA GPU and drivers are not installed yet:

```
sudo ubuntu-drivers autoinstall
sudo reboot
```

## 3. Install Ollama (the local AI engine)

```
curl -fsSL https://ollama.com/install.sh | sh
```

Then download the AI models Vyoma uses (about 4 GB total):

```
ollama pull qwen2.5:3b          # the reasoning "brain"
ollama pull nomic-embed-text    # knowledge-base search
ollama pull llama3.2:3b         # optional fallback brain
```

Check it works:

```
ollama list
```

You should see the models listed. Ollama runs automatically as a
background service from now on, including after reboots.

## 4. Install Vyoma

Copy the `vyoma` folder from your release archive / USB stick to the
home directory (this manual assumes `~/vyoma`), then:

```
cd ~/vyoma
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download a voice for the receptionist (one-time):

```
.venv/bin/python -m piper.download_voices en_US-lessac-medium --data-dir voices
```

Other included options: `en_US-hfc_female-medium`,
`en_GB-cori-medium` (female voices), `hi_IN-priyamvada-medium`
(Hindi). You can switch voices later in the control panel.

The speech-recognition model (Whisper) downloads itself the first
time voice mode runs.

## 5. First test

Text-only test (no microphone needed):

```
cd ~/vyoma
.venv/bin/python -m vyoma --text
```

Type `hello` — Vyoma should greet you. Type `goodbye` to end.
If this works, the AI engine, knowledge base and agent are all fine.

Voice test:

```
.venv/bin/python -m vyoma
```

Speak after "… listening" appears. Press Ctrl+C to stop.

## 6. Start the control panel

```
.venv/bin/python -m vyoma.gui
```

Open **http://127.0.0.1:7860** in the browser. Everything below
happens in the control panel.

## 7. Teach Vyoma about your organisation

1. Go to the **Knowledge** tab.
2. Enter your website domain (e.g. `www.yourschool.org`) and press
   **Crawl & learn**. Vyoma reads every page of your site. Progress
   appears in the Dashboard activity feed.
3. Optionally upload brochures, prospectuses, price lists, staff
   directories — PDF, DOCX, TXT or MD files.
4. Delete the sample knowledge files that shipped with Vyoma
   (anything you did not add yourself) using the ✕ buttons.

## 8. Set the identity

1. Go to the **Settings** tab.
2. Press **🔍 Detect from knowledge base**. Vyoma reads what you just
   ingested and fills in the organisation name, phone, website and
   address where it can find them.
3. Review every field — correct or complete them (address and email
   are often missing from websites), rename the receptionist if you
   like (default: Vyoma), choose a voice.
4. **Speech hotwords**: keep the short list of your organisation's
   names and local place names — this makes the microphone recognise
   them accurately in Indian English.
5. Press **Save settings**.

## 9. Test like a visitor

- **Test chat** tab: ask the questions your visitors actually ask
  ("What are the admission dates?", "I'm here to meet the principal").
- **Camera** tab: press Start preview and check your face gets a
  green box.
- **Dashboard** tab: press **▶ Start (camera + voice)** and walk up
  to the machine. Vyoma should greet you when she sees you; speak
  normally. Press **■ Stop** when done.

Fix wrong answers by adding the missing information to the knowledge
base (a simple .txt file with question-and-answer lines works well).

## 10. Optional: staff notifications on phones

1. Install the free **ntfy** app (Android/iPhone) and subscribe to a
   secret topic, e.g. `yourschool-reception-8317`.
2. In **Settings → Staff notification webhook**, enter
   `https://ntfy.sh/yourschool-reception-8317` and save.

When a visitor comes to meet someone, the staff member's phone gets
the message. (This one feature uses the internet; leave it empty for
a fully offline installation — notifications are then stored in the
database and shown in the visitors report.)

## 11. Start automatically on boot (headless robot)

For a robot that should just power up and start greeting visitors — no
keyboard, no clicking — install a single **systemd user service**. On
powerup it brings up **both**:

- the **receptionist** in camera mode — it watches the chosen webcam and
  starts a spoken conversation the instant a face appears, then loops for
  the next visitor forever; and
- the **control-panel dashboard** on the network, so you can watch the
  live camera, read reports and change settings from a phone or laptop.

One service runs both (`python -m vyoma.gui --host 0.0.0.0 --start
camera`): the panel launches and owns the receptionist, so the
dashboard's status, its Start/Stop button and the live preview all track
it. Do **not** also run a separate receptionist service — two of them
would fight over the camera and microphone.

It must be a *user* service (not a system-wide one) because the
microphone (PortAudio) and speaker (aplay) reach the audio hardware
through your PipeWire session, which only exists inside your user session.

A ready-made unit lives in the repo at
`deploy/vyoma-receptionist.service`. Install it once:

```
# Let your session (and its services) start at powerup without logging in
sudo loginctl enable-linger $USER

# Install and enable the service
mkdir -p ~/.config/systemd/user
cp deploy/vyoma-receptionist.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now vyoma-receptionist
```

Check it / read its logs:

```
systemctl --user status vyoma-receptionist
journalctl --user -u vyoma-receptionist -f
```

The robot will now start face detection, conversation and the dashboard
on every powerup. The dashboard is at `http://<robot-ip>:7860`. Pick
which webcam it uses in **Settings → Webcam** (or set
`vision.camera_index` in `config.yaml`). Ollama must also be running at
boot — install it as a service per Ollama's docs so the LLM is ready.

> The unit assumes the project is at `~/esp6/receptionist/vyoma`. If you
> cloned it elsewhere, edit the two paths in the file before copying.

## 12. Daily use

- Reports: **Reports** tab → Generate visitors report → open in the
  browser or download as Excel.
- Updating knowledge: upload new documents or re-crawl the website
  from the **Knowledge** tab any time; Vyoma re-learns automatically.
- Backup: copy three things — `config.yaml`, the `knowledge/` folder
  and the `data/` folder. That is the entire personality, memory and
  visitor history of your receptionist.

## 13. Troubleshooting

**"Neither model is available in Ollama"** — Ollama is not running or
models are missing. Run `ollama list`; re-run the `ollama pull`
commands from step 3.

**No sound** — test with `aplay /usr/share/sounds/alsa/Front_Center.wav`.
Select the right output device in Ubuntu's sound settings.

**Microphone not heard** — test with `arecord -d 3 test.wav && aplay
test.wav`. Choose the correct input device in sound settings and
raise the input volume. In noisy rooms lower `energy_threshold`
(more sensitive) or raise it (less sensitive) in `config.yaml`.

**Names misheard** — add them to **Speech hotwords** in Settings. If
accuracy is still poor, open the **Speech test** tab and compare the
Whisper engine against the Vosk Indian-English engine with your own
voice, then pick the better one under **Settings → Speech engine**
(the Vosk model downloads once, ~1 GB).

**Camera busy / black preview** — only one program can use the webcam:
stop other apps (or the receptionist) and retry. Check the camera
index in `config.yaml` (`camera_index: 0` → try `1`).

**Replies are slow** — confirm the GPU is used: `nvidia-smi` should
show `ollama` using memory. Without a GPU, expect several extra
seconds per reply.

**Port 7860 already in use** — start with another port:
`.venv/bin/python -m vyoma.gui --port 7861`.

**Wrong or invented answers** — Vyoma only knows what is in the
Knowledge tab. Remove stale files, add correct ones, and keep the
identity fields in Settings accurate.

## 14. Privacy notes for the operator

- All conversations, faces and visitor records stay in
  `~/vyoma/data/` on this computer.
- Face recognition (greeting returning visitors by name) is optional,
  off by default, and must only be enabled with visitors' consent.
- Set a data-retention practice: periodically delete old visit
  records (`data/vyoma.db`) in line with your organisation's policy
  and India's DPDP Act.
