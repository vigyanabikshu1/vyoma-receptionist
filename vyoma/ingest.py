"""Ingest documents and company web pages into the knowledge base.

Usage:
    python -m vyoma.ingest brochure.pdf policies.docx notes.txt
    python -m vyoma.ingest https://www.example.com      # crawls whole site
    python -m vyoma.ingest https://www.example.com --single
    python -m vyoma.ingest --list
    python -m vyoma.ingest --clear                       # remove all ingested

Each source is converted to a markdown file in the knowledge directory
(prefixed ``doc-`` or ``web-``) and the embedding cache is rebuilt, so
the receptionist can answer from it immediately.
"""

import argparse
import logging
import re
import sys
import time
import zipfile
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from .config import load_config
from .kb import KnowledgeBase

logger = logging.getLogger("vyoma.ingest")

USER_AGENT = "VyomaReceptionist/0.1 (knowledge-base ingestion)"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "untitled"


# -- document extractors ---------------------------------------------------

def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def extract_docx(path: Path) -> str:
    """Pull paragraph text out of a .docx without extra dependencies."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    paragraphs = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        text = (text.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&apos;", "'")
                    .replace("&quot;", '"'))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)


def extract_text_file(path: Path) -> str:
    return path.read_text(errors="ignore")


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_text_file,
    ".md": extract_text_file,
}


# -- web extractor ----------------------------------------------------------

def html_to_markdown(html: str) -> tuple[str, list[str]]:
    """Return (markdown text, same-page links) from an HTML document."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "header", "form", "iframe", "svg"]):
        tag.decompose()

    links = [a.get("href") for a in soup.find_all("a", href=True)]

    lines: list[str] = []
    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td",
                             "th", "dt", "dd"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text or len(text) < 3:
            continue
        if el.name.startswith("h"):
            lines.append(f"{'#' * int(el.name[1])} {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    # de-duplicate consecutive repeats (menus rendered twice etc.)
    deduped = [line for i, line in enumerate(lines)
               if i == 0 or line != lines[i - 1]]
    return "\n\n".join(deduped), links


def page_links(html: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return [a.get("href") for a in soup.find_all("a", href=True)]


def extract_content(html: str) -> str:
    """Main-content extraction. Prefers trafilatura, which is much
    better at keeping real content (e.g. the address in a footer)
    while dropping boilerplate; falls back to the built-in heuristic
    converter when it is not installed or extracts nothing."""
    try:
        import trafilatura
    except ImportError:
        return html_to_markdown(html)[0]
    text = trafilatura.extract(html, output_format="markdown",
                               include_comments=False,
                               include_tables=True, favor_recall=True)
    if not text or len(text.strip()) < 80:
        return html_to_markdown(html)[0]
    return text.strip()


def fetch_page(url: str) -> str:
    resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and not resp.text.lstrip().startswith("<"):
        raise ValueError(f"Not an HTML page: {ctype}")
    return resp.text


# Link targets that are never HTML pages worth fetching
SKIP_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
                 ".css", ".js", ".zip", ".gz", ".mp3", ".mp4", ".webm",
                 ".woff", ".woff2", ".ttf", ".xml", ".json", ".pdf",
                 ".apk", ".exe", ".deb"}


def canonical(url: str) -> str:
    """Key for de-duplication: ignore scheme, fragment, a leading
    ``www.`` and a trailing slash, so http(s)://(www.)site/page/ are
    all the same page."""
    p = urlparse(url)
    key = f"{p.netloc.lower().removeprefix('www.')}{p.path.rstrip('/')}"
    return f"{key}?{p.query}" if p.query else key


def ingest_url(url: str, kb_dir: Path, crawl: bool = True,
               max_pages: int = 0, max_depth: int = 5) -> list[Path]:
    """Fetch a URL (same-site crawl by default) into markdown files.

    Crawling is breadth-first up to ``max_depth`` link levels from the
    start page, staying on the same site (subdomains such as
    ``blog.example.com`` included). ``max_pages`` of 0 means no page
    limit; pass ``crawl=False`` to fetch only the given page.
    """
    start = urlparse(url)
    site = start.netloc.lower().removeprefix("www.")
    queue: deque[tuple[str, int]] = deque([(url, 0)])
    seen: set[str] = set()
    written: list[Path] = []
    limit = max_pages if max_pages > 0 else float("inf")

    while queue and len(written) < (limit if crawl else 1):
        page_url, depth = queue.popleft()
        normalized = canonical(page_url)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            html = fetch_page(page_url)
        except Exception as e:
            logger.warning("Skipping %s: %s", page_url, e)
            continue
        markdown = extract_content(html)
        links = page_links(html)
        if len(markdown) < 80:
            logger.info("Skipping %s (no useful text)", page_url)
        else:
            name = slugify(normalized.split("?")[0]) or "home"
            out = kb_dir / f"web-{name}.md"
            out.write_text(f"# Source: {page_url}\n\n{markdown}\n")
            written.append(out)
            logger.info("Saved %s (depth %d, %d chars)",
                        out.name, depth, len(markdown))

        if crawl and depth < max_depth:
            for href in links:
                target = urljoin(page_url, href)
                p = urlparse(target)
                netloc = p.netloc.lower().removeprefix("www.")
                if p.scheme not in ("http", "https"):
                    continue
                if netloc != site and not netloc.endswith("." + site):
                    continue
                if Path(p.path).suffix.lower() in SKIP_SUFFIXES:
                    continue
                if canonical(target) not in seen:
                    queue.append((target, depth + 1))
            time.sleep(0.2)  # be polite to the site
    return written


def ingest_file(path: Path, kb_dir: Path) -> Path:
    extractor = EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}' "
            f"(supported: {', '.join(EXTRACTORS)})")
    text = extractor(path)
    if len(text.strip()) < 40:
        raise ValueError(f"No text could be extracted from {path.name}")
    out = kb_dir / f"doc-{slugify(path.stem)}.md"
    out.write_text(f"# Source: {path.name}\n\n{text}\n")
    logger.info("Saved %s (%d chars)", out.name, len(text))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Add documents or a company website to Vyoma's "
                    "knowledge base")
    parser.add_argument("sources", nargs="*",
                        help="Files (.pdf .docx .txt .md) and/or URLs")
    parser.add_argument("--single", action="store_true",
                        help="Fetch only the given page instead of "
                             "crawling the whole site")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Page limit when crawling (default 0 = "
                             "no limit)")
    parser.add_argument("--depth", type=int, default=5,
                        help="Link levels to follow when crawling "
                             "(default 5)")
    # kept for backward compatibility; crawling is now the default
    parser.add_argument("--crawl", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true",
                        help="List current knowledge base files")
    parser.add_argument("--remove", metavar="FILENAME",
                        help="Remove a knowledge file by name")
    parser.add_argument("--clear", action="store_true",
                        help="Remove all ingested sources (web-*/doc-* "
                             "files); hand-written files are kept")
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    kb_dir = Path(cfg["knowledge_base"]["dir"])
    kb_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        for f in sorted(kb_dir.glob("*.md")):
            print(f"{f.name}  ({f.stat().st_size} bytes)")
        return

    if args.clear:
        ingested = sorted(kb_dir.glob("web-*.md")) + sorted(kb_dir.glob("doc-*.md"))
        for f in ingested:
            f.unlink()
            print(f"Removed {f.name}")
        print(f"Cleared {len(ingested)} ingested files.")
    elif args.remove:
        target = kb_dir / args.remove
        if not target.exists():
            sys.exit(f"No such knowledge file: {args.remove}")
        target.unlink()
        print(f"Removed {args.remove}")
    elif not args.sources:
        parser.error("Give at least one file or URL "
                     "(or --list / --remove / --clear)")

    for source in args.sources:
        if source.startswith(("http://", "https://")):
            ingest_url(source, kb_dir, not args.single,
                       args.max_pages, args.depth)
        else:
            ingest_file(Path(source), kb_dir)

    # Rebuild the embedding cache now so the next app start is instant.
    kb = KnowledgeBase(str(kb_dir),
                       cfg["knowledge_base"]["embedding_model"],
                       cfg["data"]["kb_cache"],
                       cfg["knowledge_base"]["top_k"],
                       cfg["knowledge_base"]["min_score"])
    print(f"Knowledge base ready: {len(kb.entries)} chunks from "
          f"{len(list(kb_dir.glob('*.md')))} files.")


if __name__ == "__main__":
    main()
