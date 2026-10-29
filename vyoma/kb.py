"""FAQ knowledge base with retrieval via Ollama embeddings.

Markdown files in the knowledge directory are split into chunks (by
heading/blank-line blocks), embedded once with a local embedding model,
and cached to disk. At query time the visitor's question is embedded and
the top-k most similar chunks are returned as grounding context.
"""

import hashlib
import json
import logging
import math
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


MAX_CHUNK_CHARS = 1200


def _split_long(section: str) -> list[str]:
    """Break an oversized section into paragraph groups under the limit."""
    if len(section) <= MAX_CHUNK_CHARS:
        return [section]
    parts: list[str] = []
    current = ""
    for para in section.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        parts.append(current.strip())
    return parts


def _page_title(text: str) -> str:
    """First real heading of the page (skipping the '# Source:' line)."""
    for line in text.splitlines():
        if line.startswith("#") and not line.startswith("# Source:"):
            return line.lstrip("# ").strip()
    return ""


def _is_nav_junk(chunk: str) -> bool:
    """Menu/link lists that crawling leaves behind carry no facts."""
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    bullets = sum(1 for ln in lines if ln.lstrip().startswith("- "))
    return len(lines) >= 5 and bullets / len(lines) > 0.7


def _has_body(chunk: str) -> bool:
    """A chunk that is only headings ('# Contact Us') states nothing;
    keeping it just crowds real content out of the top-k results."""
    return any(ln.strip() and not ln.lstrip().startswith("#")
               for ln in chunk.splitlines())


def _chunk_markdown(text: str) -> list[str]:
    """Split on headings; oversized sections split by paragraph groups.

    Each chunk is prefixed with the page title so a fact stays findable
    when its heading alone doesn't mention the topic (e.g. a principal's
    name under a bare '## <name>' heading on the Principal's Message
    page)."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    chunks = [part for s in sections for part in _split_long(s)]
    title = _page_title(text)
    prefix = f"[Page: {title}]\n" if title else ""
    return [prefix + c for c in chunks
            if len(c) > 20 and _has_body(c) and not _is_nav_junk(c)]


class KnowledgeBase:
    def __init__(self, kb_dir: str, embedding_model: str, cache_path: str,
                 top_k: int = 3, min_score: float = 0.35):
        self.kb_dir = Path(kb_dir)
        self.model = embedding_model
        self.cache_path = Path(cache_path)
        self.top_k = top_k
        self.min_score = min_score
        self.entries: list[dict] = []  # {text, embedding}
        self._load()

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for f in sorted(self.kb_dir.glob("*.md")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        h.update(self.model.encode())
        return h.hexdigest()

    def _load(self):
        fp = self._fingerprint()
        if self.cache_path.exists():
            cached = json.loads(self.cache_path.read_text())
            if cached.get("fingerprint") == fp:
                self.entries = cached["entries"]
                logger.info("KB loaded from cache (%d chunks)", len(self.entries))
                return
        self._build(fp)

    def _build(self, fingerprint: str):
        chunks: list[str] = []
        for f in sorted(self.kb_dir.glob("*.md")):
            chunks.extend(_chunk_markdown(f.read_text()))
        logger.info("Embedding %d KB chunks with %s ...", len(chunks), self.model)
        self.entries = []
        for chunk in chunks:
            resp = ollama.embeddings(model=self.model, prompt=chunk)
            self.entries.append({"text": chunk, "embedding": resp["embedding"]})
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: another process (e.g. the GUI chat agent) may
        # read the cache at any moment — it must never see a half-
        # written file.
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fingerprint": fingerprint, "entries": self.entries})
        )
        tmp.replace(self.cache_path)
        logger.info("KB built and cached (%d chunks)", len(self.entries))

    def search(self, query: str) -> list[str]:
        if not self.entries:
            return []
        q = ollama.embeddings(model=self.model, prompt=query)["embedding"]
        scored = sorted(
            ((_cosine(q, e["embedding"]), e["text"]) for e in self.entries),
            key=lambda t: t[0], reverse=True,
        )
        return [text for score, text in scored[: self.top_k]
                if score >= self.min_score]
