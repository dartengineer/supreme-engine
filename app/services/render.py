"""Turns a structured CV (or a cover letter) into HTML, PDF and DOCX files.

- HTML: Jinja2 templates in app/templates (classic, modern, compact; one cover letter template).
- PDF: the HTML is sent to the WeasyPrint renderer running in Docker (pdf_renderer/), at settings.pdf_renderer_url.
- DOCX: built with python-docx from the same data, with real Word headings and bullet lists (ATS-friendly).
"""
import io
import re
from pathlib import Path

import httpx
from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import settings
from app.schemas import CVDocument
from app.services.cv_document import contact_line

TEMPLATES = ("classic", "modern", "compact")
_env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parent.parent / "templates"),
                   autoescape=select_autoescape(["html"]))


class RendererUnavailable(Exception):
    """The PDF renderer container isn't reachable or failed. The message says how to fix it."""


# ---------------------------------------------------------------- HTML

def cv_html(cv: CVDocument, template: str = "classic") -> str:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}")
    return _env.get_template(f"cv_{template}.html").render(cv=cv, contact=contact_line(cv),
                                                           compact=(template == "compact"))


def split_letter(text: str) -> dict:
    """Splits a plain-text letter into greeting, paragraphs and sign-off lines."""
    lines = text.strip().splitlines()
    greeting = lines.pop(0).strip() if lines and lines[0].strip().lower().startswith("dear") else None
    body = "\n".join(lines)
    parts = re.split(r"^\s*(?=(?:kind regards|best regards|regards|sincerely|yours sincerely)\b)", body,
                     maxsplit=1, flags=re.I | re.M)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", parts[0]) if p.strip()]
    sign_off = [l.strip() for l in parts[1].splitlines() if l.strip()] if len(parts) > 1 else []
    return {"greeting": greeting, "paragraphs": paragraphs, "sign_off": sign_off}


def letter_html(text: str, name: str, contact: str) -> str:
    return _env.get_template("cover_letter.html").render(name=name, contact=contact, **split_letter(text))


# ---------------------------------------------------------------- PDF

def html_to_pdf(html: str) -> bytes:
    url = settings.pdf_renderer_url.rstrip("/") + "/pdf"
    try:
        r = httpx.post(url, content=html.encode("utf-8"), headers={"Content-Type": "text/html; charset=utf-8"},
                       timeout=60)
    except httpx.HTTPError:
        raise RendererUnavailable(
            "The PDF renderer isn't running. Start it with: docker start cv-pdf "
            "(first time: docker build -t cv-pdf-renderer pdf_renderer, then docker run -d --name cv-pdf "
            "--restart unless-stopped -p 127.0.0.1:5001:5001 cv-pdf-renderer)")
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        raise RendererUnavailable(f"The PDF renderer failed ({r.status_code}): {r.text[:200]}")
    return r.content


# ---------------------------------------------------------------- DOCX

DOCX_STYLES = {
    #           body font            heading font         size  name  accent
    "classic": ("Times New Roman", "Times New Roman", 10.5, 22, RGBColor(0x1D, 0x1D, 0x1D)),
    "modern": ("Calibri", "Calibri", 10, 24, RGBColor(0x0E, 0x6E, 0x6B)),
    "compact": ("Arial", "Arial", 9, 16, RGBColor(0x20, 0x20, 0x20)),
}


def _new_document(template: str):
    body_font, heading_font, size, _, accent = DOCX_STYLES[template]
    doc = Document()
    section = doc.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    margin = Cm(1.4) if template == "compact" else Cm(1.9)
    section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = margin
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = body_font, Pt(size)
    normal.paragraph_format.space_after = Pt(2)
    heading = doc.styles["Heading 1"]
    heading.font.name, heading.font.size, heading.font.bold = heading_font, Pt(size + 1), True
    heading.font.color.rgb = accent
    heading.paragraph_format.space_before, heading.paragraph_format.space_after = Pt(10), Pt(3)
    return doc


def _header(doc, template: str, name: str, contact: str) -> None:
    _, _, _, name_size, accent = DOCX_STYLES[template]
    p = doc.add_paragraph()
    run = p.add_run(name)
    run.bold, run.font.size = True, Pt(name_size)
    run.font.color.rgb = accent if template == "modern" else RGBColor(0x1D, 0x1D, 0x1D)
    if contact:
        doc.add_paragraph(contact)


def _bullets(doc, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def _line_with_dates(doc, text: str, dates: str, subtitle: str | None = None) -> None:
    """Bold title on the left, dates right-aligned on the same line (a tab stop, not a table: ATS-friendly)."""
    p = doc.add_paragraph()
    usable = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    p.paragraph_format.tab_stops.add_tab_stop(usable, WD_TAB_ALIGNMENT.RIGHT)
    p.paragraph_format.keep_with_next = True
    p.add_run(text).bold = True
    if dates:
        p.add_run(f"\t{dates}")
    if subtitle:
        doc.add_paragraph(subtitle).runs[0].italic = True


def cv_docx(cv: CVDocument, template: str = "classic") -> bytes:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}")
    doc = _new_document(template)
    _header(doc, template, cv.name, contact_line(cv))
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(cv.summary)
    if cv.experience:
        doc.add_heading("Experience", level=1)
        for j in cv.experience:
            title = f"{j.job_title} — {j.company}" if template == "compact" else j.job_title
            subtitle = None if template == "compact" else j.company + (f", {j.location}" if j.location else "")
            _line_with_dates(doc, title, f"{j.start} – {j.end}", subtitle)
            _bullets(doc, j.bullets)
    if cv.other_experience:
        doc.add_heading("Other Experience", level=1)
        for j in cv.other_experience:
            doc.add_paragraph(f"{j.job_title}, {j.company}, {j.start} – {j.end}")
    if cv.projects:
        doc.add_heading("Projects", level=1)
        for p in cv.projects:
            _line_with_dates(doc, p.name + (f" | {p.technologies}" if p.technologies else ""), "")
            _bullets(doc, p.bullets)
    if cv.education:
        doc.add_heading("Education", level=1)
        for ed in cv.education:
            _line_with_dates(doc, ed.qualification + (f", {ed.field_of_study}" if ed.field_of_study else ""),
                             ed.dates or "", ed.institution)
            if ed.details:
                doc.add_paragraph(ed.details)
    if cv.skills:
        doc.add_heading("Skills", level=1)
        doc.add_paragraph(" · ".join(cv.skills))
    if cv.certifications:
        doc.add_heading("Certifications", level=1)
        _bullets(doc, [c.name + (f", {c.issuer}" if c.issuer else "") for c in cv.certifications])
    return _save(doc)


def letter_docx(text: str, name: str, contact: str) -> bytes:
    doc = _new_document("classic")
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Normal"].paragraph_format.space_after = Pt(8)
    _header(doc, "classic", name, contact)
    parts = split_letter(text)
    if parts["greeting"]:
        doc.add_paragraph(parts["greeting"])
    for para in parts["paragraphs"]:
        doc.add_paragraph(para)
    if parts["sign_off"]:
        doc.add_paragraph("\n".join(parts["sign_off"]))
    return _save(doc)


def _save(doc) -> bytes:
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
