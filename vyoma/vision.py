"""Visitor presence detection and (optional) face recognition.

Detection uses OpenCV's bundled Haar cascade — cheap enough to run on the
CPU continuously. If the `face_recognition` package (dlib) is installed,
detected faces are matched against embeddings stored in the visitor
database so returning guests can be greeted by name.
"""

import logging
import os
import threading
import time

from . import events

logger = logging.getLogger(__name__)


class FaceWatcher:
    def __init__(self, cfg: dict, db):
        import cv2

        self.cv2 = cv2
        self.cfg = cfg
        self.db = db
        self.cap = cv2.VideoCapture(cfg["camera_index"])
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cfg['camera_index']}")
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # A pump thread reads the camera continuously — detection keeps
        # running and the GUI's preview frame stays live even while the
        # receptionist is busy talking to a visitor.
        self._latest = None            # (frame, faces)
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._pump, daemon=True).start()

        self.recognizer = None
        if cfg.get("recognition"):
            try:
                import face_recognition
                self.recognizer = face_recognition
            except ImportError:
                logger.info("face_recognition not installed; "
                            "presence detection only")

    def _pump(self):
        last_shared = 0.0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.5)
                continue
            faces = self._detect(frame)
            with self._lock:
                self._latest = (frame, faces)
            if time.monotonic() - last_shared >= 0.4:
                self._share_frame(frame, faces)
                last_shared = time.monotonic()
            time.sleep(0.05)

    def _detect(self, frame):
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        return self.detector.detectMultiScale(gray, scaleFactor=1.2,
                                              minNeighbors=5,
                                              minSize=(80, 80))

    def _share_frame(self, frame, faces):
        """Write an annotated snapshot for the GUI's live camera view."""
        path = self.cfg.get("frame_path")
        if not path:
            return
        annotated = frame.copy()
        for (x, y, w, h) in faces:
            self.cv2.rectangle(annotated, (x, y), (x + w, y + h),
                               (60, 180, 75), 2)
        self.cv2.imwrite(path, annotated,
                         [self.cv2.IMWRITE_JPEG_QUALITY, 70])

    def _recognise(self, frame) -> str | None:
        if not self.recognizer:
            return None
        known = self.db.known_faces()
        if not known:
            return None
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        encodings = self.recognizer.face_encodings(rgb)
        if not encodings:
            return None
        import numpy as np

        names = [n for n, _ in known]
        vectors = np.array([v for _, v in known])
        distances = self.recognizer.face_distance(vectors, encodings[0])
        best = int(distances.argmin())
        if distances[best] <= self.cfg["match_tolerance"]:
            return names[best]
        return None

    def enroll(self, frame, name: str) -> bool:
        """Store a face embedding for a consenting new visitor."""
        if not self.recognizer:
            return False
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        encodings = self.recognizer.face_encodings(rgb)
        if not encodings:
            return False
        self.db.upsert_visitor(name=name,
                               face_embedding=encodings[0].tolist())
        logger.info("Enrolled face for %s", name)
        return True

    def _force_listen_requested(self) -> bool:
        """True if the GUI has asked for a manual listen (a fresh flag file).

        Consumes the flag so it fires once. Stale flags (>60 s old, e.g. left
        over from a click made during a long conversation) are ignored.
        """
        path = self.cfg.get("force_listen_path")
        if not path or not os.path.exists(path):
            return False
        try:
            fresh = time.time() - os.path.getmtime(path) < 60
        except OSError:
            fresh = False
        try:
            os.remove(path)
        except OSError:
            pass
        return fresh

    def wait_for_visitor(self) -> tuple[object, str | None]:
        """Block until a face is continuously present, then try to name it.

        Also returns immediately if the GUI requests a manual "force listen"
        (visitor_name is None in that case, same as an unrecognised face).

        Returns (frame, visitor_name_or_None).
        """
        required = self.cfg["presence_seconds"]
        first_seen = None
        was_present = False
        while True:
            if self._force_listen_requested():
                events.emit("force_listen")
                with self._lock:
                    item = self._latest
                return (item[0] if item else None), None
            with self._lock:
                item = self._latest
            if item is None:
                time.sleep(0.1)
                continue
            frame, faces = item
            present = len(faces) > 0
            if present != was_present:
                events.emit("face", present=present, count=len(faces))
                was_present = present
            if present:
                first_seen = first_seen or time.monotonic()
                if time.monotonic() - first_seen >= required:
                    name = self._recognise(frame)
                    events.emit("visitor_detected", name=name)
                    return frame, name
            else:
                first_seen = None
            time.sleep(0.1)

    def release(self):
        self._running = False
        time.sleep(0.2)
        self.cap.release()
