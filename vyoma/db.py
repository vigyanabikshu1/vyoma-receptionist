"""SQLite store for visitors, visits, notifications and face embeddings."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS visitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT,
    company TEXT,
    face_embedding TEXT,          -- JSON list of floats (optional)
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_name TEXT NOT NULL,
    purpose TEXT,
    whom_to_meet TEXT,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    staff TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # -- visitors ---------------------------------------------------------
    def upsert_visitor(self, name: str, phone: str | None = None,
                       company: str | None = None,
                       face_embedding: list[float] | None = None) -> int:
        now = self._now()
        row = self.conn.execute(
            "SELECT id FROM visitors WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE visitors SET last_seen = ?, "
                "phone = COALESCE(?, phone), company = COALESCE(?, company), "
                "face_embedding = COALESCE(?, face_embedding) WHERE id = ?",
                (now, phone, company,
                 json.dumps(face_embedding) if face_embedding else None,
                 row["id"]),
            )
            visitor_id = row["id"]
        else:
            cur = self.conn.execute(
                "INSERT INTO visitors (name, phone, company, face_embedding, "
                "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                (name, phone, company,
                 json.dumps(face_embedding) if face_embedding else None,
                 now, now),
            )
            visitor_id = cur.lastrowid
        self.conn.commit()
        return visitor_id

    def known_faces(self) -> list[tuple[str, list[float]]]:
        rows = self.conn.execute(
            "SELECT name, face_embedding FROM visitors "
            "WHERE face_embedding IS NOT NULL"
        ).fetchall()
        return [(r["name"], json.loads(r["face_embedding"])) for r in rows]

    def find_visitor(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM visitors WHERE lower(name) = lower(?)", (name,)
        ).fetchone()

    # -- visits -----------------------------------------------------------
    def log_visit(self, visitor_name: str, purpose: str | None,
                  whom_to_meet: str | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO visits (visitor_name, purpose, whom_to_meet, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (visitor_name, purpose, whom_to_meet, self._now()),
        )
        self.conn.commit()
        logger.info("Visit logged: %s (%s)", visitor_name, purpose)
        return cur.lastrowid

    # -- notifications ----------------------------------------------------
    def add_notification(self, staff: str, message: str, delivered: bool) -> int:
        cur = self.conn.execute(
            "INSERT INTO notifications (staff, message, timestamp, delivered) "
            "VALUES (?, ?, ?, ?)",
            (staff, message, self._now(), int(delivered)),
        )
        self.conn.commit()
        return cur.lastrowid
