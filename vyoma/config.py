"""Configuration loading for the Vyoma receptionist."""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the project root
    for section, key in (
        ("knowledge_base", "dir"),
        ("tts", "voices_dir"),
        ("data", "db_path"),
        ("data", "kb_cache"),
        ("stt", "vosk_model"),
        ("stt", "whisper_download_root"),
        ("stt", "diagnostic_dir"),
    ):
        value = cfg.get(section, {}).get(key)
        if value and not os.path.isabs(value):
            cfg[section][key] = str(PROJECT_ROOT / value)
    return cfg
