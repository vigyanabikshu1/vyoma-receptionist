"""Vyoma receptionist — event loop tying perception, reasoning and speech.

Usage:
    python -m vyoma            # voice conversation (mic + speaker)
    python -m vyoma --text     # keyboard conversation (development)
    python -m vyoma --camera   # wait for a face before greeting
"""

import argparse
import logging
import time
from pathlib import Path

from . import events
from .agent import ReceptionistAgent
from .config import load_config
from .db import Database
from .kb import KnowledgeBase
from .stt import create_stt
from .tools import Tools
from .tts import create_tts

logger = logging.getLogger("vyoma")


def run_conversation(cfg: dict, db: Database, kb: KnowledgeBase,
                     stt, tts, visitor_name: str | None):
    tools = Tools(db, cfg["notifications"]["webhook_url"])
    agent = ReceptionistAgent(cfg, kb, tools)

    events.emit("conversation_start", visitor=visitor_name)
    greeting = agent.greet(visitor_name)
    events.emit("speak", text=greeting)
    tts.speak(greeting)

    idle_timeout = cfg["conversation"]["idle_timeout_seconds"]
    last_heard = time.monotonic()
    for _ in range(cfg["conversation"]["max_turns"]):
        events.emit("listening")
        text = stt.listen()
        if text is None:
            if getattr(stt, "closed", False):
                logger.info("Input closed")
                break
            if time.monotonic() - last_heard > idle_timeout:
                logger.info("Conversation timed out")
                break
            continue
        last_heard = time.monotonic()
        events.emit("heard", text=text)

        reply = agent.respond(text)
        if reply:
            events.emit("speak", text=reply)
            tts.speak(reply)
        if agent.finished:
            break
    logger.info("Conversation ended")
    events.emit("conversation_end")


def main():
    parser = argparse.ArgumentParser(description="Vyoma receptionist robot")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--text", action="store_true",
                        help="Type instead of speaking (no mic/speaker)")
    parser.add_argument("--camera", action="store_true",
                        help="Use the camera to detect visitors")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.text:
        cfg["stt"]["engine"] = "none"
        cfg["tts"]["engine"] = "none"

    data_dir = Path(cfg["data"]["db_path"]).parent
    events.configure(data_dir / "events.jsonl",
                     level_path=data_dir / "mic_level.json",
                     speak_path=data_dir / "speak_state.json")
    events.emit("session_start",
                mode="text" if args.text else
                     ("camera" if args.camera else "voice"))
    cfg["vision"]["frame_path"] = str(data_dir / "camera_frame.jpg")
    # Manual "force listen" trigger: the GUI writes this flag file to start a
    # conversation without waiting for a face (see FaceWatcher.wait_for_visitor).
    force_listen_path = data_dir / "force_listen.flag"
    cfg["vision"]["force_listen_path"] = str(force_listen_path)
    force_listen_path.unlink(missing_ok=True)  # drop any stale flag from a prior run

    db = Database(cfg["data"]["db_path"])
    kb = KnowledgeBase(cfg["knowledge_base"]["dir"],
                       cfg["knowledge_base"]["embedding_model"],
                       cfg["data"]["kb_cache"],
                       cfg["knowledge_base"]["top_k"],
                       cfg["knowledge_base"]["min_score"])
    stt = create_stt(cfg)
    tts = create_tts(cfg)

    watcher = None
    if args.camera or cfg["vision"]["enabled"]:
        from .vision import FaceWatcher

        watcher = FaceWatcher(cfg["vision"], db)

    try:
        while True:
            visitor_name = None
            if watcher:
                logger.info("Waiting for a visitor ...")
                _, visitor_name = watcher.wait_for_visitor()
            run_conversation(cfg, db, kb, stt, tts, visitor_name)
            if not watcher:
                break  # single conversation in mic/text mode
    except KeyboardInterrupt:
        print()
        logger.info("Shutting down")
    finally:
        events.emit("session_end")
        if watcher:
            watcher.release()


if __name__ == "__main__":
    main()
