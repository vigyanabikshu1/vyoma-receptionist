"""Generate a visitors report as a self-contained HTML page.

Usage:
    python -m vyoma.report                      # writes data/visitors_report.html
    python -m vyoma.report -o /tmp/report.html
    python -m vyoma.report --xlsx report.xlsx   # also write the Excel file to disk

The page shows visits, visitors and notifications from the SQLite
database with search and date filtering, and has an "Export to Excel"
button. The Excel workbook (a real .xlsx, built with only the standard
library) is embedded in the page as base64, so the report is a single
file that works completely offline.
"""

import argparse
import base64
import html
import io
import logging
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from .config import load_config

logger = logging.getLogger("vyoma.report")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

VISITS_COLS = ["#", "Visitor", "Purpose", "Whom to meet", "Date & time"]
VISITORS_COLS = ["#", "Name", "Phone", "Company", "First seen", "Last seen",
                 "Face enrolled"]
NOTIFY_COLS = ["#", "Staff", "Message", "Date & time", "Delivered"]


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).strftime("%d %b %Y  %H:%M")
    except ValueError:
        return ts


def fetch_report_data(db_path: str) -> dict[str, list[list[str]]]:
    """Read all three tables into plain rows of display strings."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    visits = [
        [str(r["id"]), r["visitor_name"] or "(not given)",
         r["purpose"] or "", r["whom_to_meet"] or "", _fmt_ts(r["timestamp"])]
        for r in conn.execute("SELECT * FROM visits ORDER BY timestamp DESC")
    ]
    visitors = [
        [str(r["id"]), r["name"] or "(not given)", r["phone"] or "",
         r["company"] or "", _fmt_ts(r["first_seen"]), _fmt_ts(r["last_seen"]),
         "yes" if r["face_embedding"] else "no"]
        for r in conn.execute("SELECT * FROM visitors ORDER BY last_seen DESC")
    ]
    notifications = [
        [str(r["id"]), r["staff"], r["message"], _fmt_ts(r["timestamp"]),
         "yes" if r["delivered"] else "no"]
        for r in conn.execute(
            "SELECT * FROM notifications ORDER BY timestamp DESC")
    ]
    conn.close()
    return {"visits": visits, "visitors": visitors,
            "notifications": notifications}


# ---------------------------------------------------------------------------
# Minimal .xlsx writer (standard library only)
# ---------------------------------------------------------------------------

_XLSX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{sheet_overrides}
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

_XLSX_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_XLSX_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill></fills>
<borders count="1"><border/></borders>
<cellStyleXfs count="1"><xf/></cellStyleXfs>
<cellXfs count="2"><xf/><xf fontId="1" applyFont="1"/></cellXfs>
</styleSheet>"""


def _sheet_xml(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str, style: int) -> str:
        return (f'<c t="inlineStr" s="{style}"><is><t xml:space="preserve">'
                f"{escape(value)}</t></is></c>")

    widths = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{max(12, len(h) + 6)}" '
        'customWidth="1"/>' for i, h in enumerate(headers))
    body = ["<row>" + "".join(cell(h, 1) for h in headers) + "</row>"]
    body += ["<row>" + "".join(cell(v, 0) for v in row) + "</row>"
             for row in rows]
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main">'
            f"<cols>{widths}</cols><sheetData>{''.join(body)}</sheetData>"
            "</worksheet>")


def build_xlsx(sheets: list[tuple[str, list[str], list[list[str]]]]) -> bytes:
    """Build an .xlsx workbook: [(sheet_name, headers, rows), ...]."""
    sheet_overrides, sheet_tags, rel_tags = [], [], []
    for i, (name, _, _) in enumerate(sheets, start=1):
        sheet_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.spreadsheetml.'
            'worksheet+xml"/>')
        sheet_tags.append(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        rel_tags.append(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.'
            'org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>')
    styles_rid = len(sheets) + 1
    rel_tags.append(
        f'<Relationship Id="rId{styles_rid}" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>')

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
        '2006/main" xmlns:r="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships">'
        f"<sheets>{''.join(sheet_tags)}</sheets></workbook>")
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
        f"2006/relationships\">{''.join(rel_tags)}</Relationships>")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _XLSX_CONTENT_TYPES.format(
            sheet_overrides="\n".join(sheet_overrides)))
        z.writestr("_rels/.rels", _XLSX_ROOT_RELS)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/styles.xml", _XLSX_STYLES)
        for i, (_, headers, rows) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(headers, rows))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{org} — Visitors Report</title>
<style>
  :root {{
    --accent: #1f4e79; --bg: #f5f7fa; --card: #ffffff;
    --line: #e3e8ef; --muted: #64748b;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 15px/1.5 system-ui, "Segoe UI", sans-serif;
         background: var(--bg); color: #1e293b; }}
  header {{ background: var(--accent); color: #fff; padding: 20px 28px; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header p {{ margin: 4px 0 0; opacity: .85; font-size: 13px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 22px 20px 60px; }}
  .cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }}
  .card {{ background: var(--card); border: 1px solid var(--line);
           border-radius: 10px; padding: 14px 22px; min-width: 150px; }}
  .card .num {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
  .card .lbl {{ font-size: 12px; color: var(--muted);
                text-transform: uppercase; letter-spacing: .5px; }}
  .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
              margin-bottom: 18px; }}
  .toolbar input {{ padding: 8px 12px; border: 1px solid var(--line);
                    border-radius: 8px; font: inherit; background: var(--card); }}
  .toolbar input[type=search] {{ flex: 1; min-width: 200px; }}
  .btn {{ background: #1d7a3a; color: #fff; border: 0; border-radius: 8px;
          padding: 9px 18px; font: inherit; font-weight: 600; cursor: pointer;
          text-decoration: none; display: inline-block; }}
  .btn:hover {{ background: #17632f; }}
  h2 {{ font-size: 16px; color: var(--accent); margin: 26px 0 8px; }}
  .tablewrap {{ overflow-x: auto; background: var(--card);
                border: 1px solid var(--line); border-radius: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 9px 14px;
            border-bottom: 1px solid var(--line); white-space: nowrap; }}
  td.wrap {{ white-space: normal; min-width: 260px; }}
  th {{ background: #eef2f7; font-size: 12px; text-transform: uppercase;
        letter-spacing: .4px; color: #334155; position: sticky; top: 0; }}
  tr:last-child td {{ border-bottom: 0; }}
  tr:hover td {{ background: #f8fafc; }}
  .empty {{ color: var(--muted); padding: 14px; }}
  footer {{ text-align: center; color: var(--muted); font-size: 12px;
            padding: 18px; }}
</style>
</head>
<body>
<header>
  <h1>{org} — Visitors Report</h1>
  <p>Generated by {assistant} on {generated}</p>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="num">{n_visits}</div>
      <div class="lbl">Visits</div></div>
    <div class="card"><div class="num">{n_visitors}</div>
      <div class="lbl">Visitors</div></div>
    <div class="card"><div class="num">{n_notifications}</div>
      <div class="lbl">Notifications</div></div>
  </div>

  <div class="toolbar">
    <input type="search" id="q" placeholder="Search name, purpose, staff…">
    <label>From <input type="date" id="from"></label>
    <label>To <input type="date" id="to"></label>
    <a class="btn" id="xlsx" download="{xlsx_name}"
       href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{xlsx_b64}">
       ⬇ Export to Excel</a>
  </div>

  <h2>Visits</h2>
  <div class="tablewrap">{visits_table}</div>

  <h2>Visitors</h2>
  <div class="tablewrap">{visitors_table}</div>

  <h2>Staff notifications</h2>
  <div class="tablewrap">{notifications_table}</div>
</main>
<footer>Vyoma offline receptionist · data source: {db_path} ·
  all data stays on this machine</footer>
<script>
  const q = document.getElementById("q");
  const from = document.getElementById("from");
  const to = document.getElementById("to");

  function applyFilters() {{
    const text = q.value.trim().toLowerCase();
    const lo = from.value ? new Date(from.value + "T00:00:00") : null;
    const hi = to.value ? new Date(to.value + "T23:59:59") : null;
    document.querySelectorAll("tbody tr").forEach(tr => {{
      let show = !text || tr.textContent.toLowerCase().includes(text);
      const iso = tr.dataset.ts;
      if (show && iso && (lo || hi)) {{
        const t = new Date(iso);
        if (lo && t < lo) show = false;
        if (hi && t > hi) show = false;
      }}
      tr.style.display = show ? "" : "none";
    }});
  }}
  [q, from, to].forEach(el => el.addEventListener("input", applyFilters));
</script>
</body>
</html>
"""


def _html_table(headers: list[str], rows: list[list[str]],
                ts_index: int | None = None,
                wrap_index: int | None = None) -> str:
    if not rows:
        return '<div class="empty">No records yet.</div>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        attr = ""
        if ts_index is not None and row[ts_index]:
            try:
                iso = datetime.strptime(
                    row[ts_index], "%d %b %Y  %H:%M").isoformat()
                attr = f' data-ts="{iso}"'
            except ValueError:
                pass
        cells = "".join(
            f'<td{" class=" + chr(34) + "wrap" + chr(34) if i == wrap_index else ""}>'
            f"{html.escape(v)}</td>" for i, v in enumerate(row))
        body.append(f"<tr{attr}>{cells}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def build_html(data: dict[str, list[list[str]]], org: str, assistant: str,
               db_path: str, xlsx_bytes: bytes) -> str:
    stamp = datetime.now()
    return _PAGE.format(
        org=html.escape(org),
        assistant=html.escape(assistant),
        generated=stamp.strftime("%d %b %Y, %H:%M"),
        db_path=html.escape(db_path),
        n_visits=len(data["visits"]),
        n_visitors=len(data["visitors"]),
        n_notifications=len(data["notifications"]),
        visits_table=_html_table(VISITS_COLS, data["visits"], ts_index=4),
        visitors_table=_html_table(VISITORS_COLS, data["visitors"],
                                   ts_index=5),
        notifications_table=_html_table(NOTIFY_COLS, data["notifications"],
                                        ts_index=3, wrap_index=2),
        xlsx_name=f"visitors_report_{stamp:%Y%m%d}.xlsx",
        xlsx_b64=base64.b64encode(xlsx_bytes).decode("ascii"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate the visitors report (HTML with Excel export).")
    parser.add_argument("-o", "--output", default=None,
                        help="output HTML path "
                             "(default: data/visitors_report.html)")
    parser.add_argument("--xlsx", default=None, metavar="PATH",
                        help="also write the Excel workbook to this path")
    args = parser.parse_args()

    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    db_path = str(root / cfg["data"]["db_path"])
    if not Path(db_path).exists():
        logger.error("Database not found: %s (run the receptionist first)",
                     db_path)
        return 1

    data = fetch_report_data(db_path)
    xlsx = build_xlsx([
        ("Visits", VISITS_COLS, data["visits"]),
        ("Visitors", VISITORS_COLS, data["visitors"]),
        ("Notifications", NOTIFY_COLS, data["notifications"]),
    ])
    page = build_html(data, cfg["organisation"]["name"],
                      cfg["organisation"]["assistant_name"], db_path, xlsx)

    out = Path(args.output) if args.output else root / "data" / "visitors_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    logger.info("Report written: %s  (%d visits, %d visitors)",
                out, len(data["visits"]), len(data["visitors"]))

    if args.xlsx:
        Path(args.xlsx).write_bytes(xlsx)
        logger.info("Excel workbook written: %s", args.xlsx)
    logger.info("Open it with:  xdg-open %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
