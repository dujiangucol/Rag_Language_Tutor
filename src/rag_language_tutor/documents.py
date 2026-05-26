from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    section: str
    page: int | None = None


def load_documents(paths: list[Path], chunk_size: int = 900, overlap: int = 140) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            chunks.extend(load_pdf(path, chunk_size=chunk_size, overlap=overlap))
        elif path.suffix.lower() in {".md", ".txt"}:
            chunks.extend(load_text(path, chunk_size=chunk_size, overlap=overlap))
    return chunks


def load_text(path: Path, chunk_size: int = 900, overlap: int = 140) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    sections = split_markdown_sections(text)
    chunks: list[DocumentChunk] = []
    for section_title, section_text in sections:
        for i, chunk_text in enumerate(chunk_text_by_sentence(section_text, chunk_size, overlap)):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{path.stem}:{slugify(section_title)}:{i}",
                    text=chunk_text,
                    source=path.name,
                    section=section_title,
                )
            )
    return chunks


def load_pdf(path: Path, chunk_size: int = 900, overlap: int = 140) -> list[DocumentChunk]:
    if PdfReader is None:
        return []

    reader = PdfReader(str(path))
    chunks: list[DocumentChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for i, chunk_text in enumerate(chunk_text_by_sentence(text, chunk_size, overlap)):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{path.stem}:page-{page_number}:{i}",
                    text=chunk_text,
                    source=path.name,
                    section=f"page {page_number}",
                    page=page_number,
                )
            )
    return chunks


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE))
    if not matches:
        return [("Document", text.strip())]

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def chunk_text_by_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:]
        overlapped.append(f"{prefix} {chunk}".strip())
    return overlapped


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value or "section"
