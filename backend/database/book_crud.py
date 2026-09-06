"""Async CRUD for the Book model — used by the read-aloud reader.

Sentences and page_map are large JSON payloads (a 300-page book is ~500KB).
We pull them out of the list/get response by default and only return them
when the caller asks (e.g. the first time a reader opens the book, to seed
its sentence list; subsequent opens can stream without the JSON).
"""
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Book


def _book_to_dict(book: Book, include_text: bool = False) -> Dict[str, Any]:
    """Serialize a Book row. include_text=False drops sentences (large JSON)
    so list endpoints stay small; include_text=True returns the full payload."""
    d: Dict[str, Any] = {
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "filepath": book.filepath,
        "file_type": book.file_type,
        "size_bytes": book.size_bytes,
        "source_type": getattr(book, "source_type", None) or "file",
        "source_url": getattr(book, "source_url", None),
        "domain": getattr(book, "domain", None),
        "has_article": bool(getattr(book, "html_path", None)),
        "total_sentences": book.total_sentences,
        "current_sentence_idx": book.current_sentence_idx,
        "current_page": book.current_page,
        "last_read_at": book.last_read_at.isoformat() if book.last_read_at else None,
        "created_at": book.created_at.isoformat() if book.created_at else None,
    }
    if include_text:
        d["sentences"] = book.sentences or []
        d["page_map"] = book.page_map or []
    return d


async def create_book(
    db: AsyncSession,
    title: str,
    filepath: str,
    file_type: str,
    sentences: List[Dict[str, Any]],
    page_map: List[int],
    size_bytes: int = 0,
    author: Optional[str] = None,
    source_type: str = "file",
    source_url: Optional[str] = None,
    domain: Optional[str] = None,
    html_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a Book row with the extraction cache populated."""
    book = Book(
        title=title,
        author=author,
        filepath=filepath,
        file_type=file_type,
        size_bytes=size_bytes,
        sentences=sentences,
        page_map=page_map,
        total_sentences=len(sentences),
        source_type=source_type,
        source_url=source_url,
        domain=domain,
        html_path=html_path,
    )
    db.add(book)
    await db.flush()
    return _book_to_dict(book, include_text=False)


async def get_book_by_source_url(db: AsyncSession, source_url: str) -> Optional[Dict[str, Any]]:
    """Active book previously saved from this URL (dedupe for re-saves)."""
    result = await db.execute(
        select(Book).where(Book.source_url == source_url, Book.is_active == 1)
    )
    book = result.scalar_one_or_none()
    return _book_to_dict(book, include_text=False) if book else None


async def list_books(db: AsyncSession, limit: int = 500) -> List[Dict[str, Any]]:
    """All active books, light payload (no sentences)."""
    result = await db.execute(
        select(Book).where(Book.is_active == 1).order_by(Book.created_at.desc()).limit(limit)
    )
    return [_book_to_dict(b, include_text=False) for b in result.scalars().all()]


async def get_book(
    db: AsyncSession, book_id: str, include_text: bool = True
) -> Optional[Dict[str, Any]]:
    """Single book. include_text=True returns sentences + page_map (the heavy
    payload) so the reader can render its sentence list without a second
    round-trip."""
    result = await db.execute(select(Book).where(Book.id == book_id, Book.is_active == 1))
    book = result.scalar_one_or_none()
    return _book_to_dict(book, include_text=include_text) if book else None


async def update_book_progress(
    db: AsyncSession, book_id: str, sentence_idx: int, page: int
) -> bool:
    """Persist the read-along position. Cheap; called per sentence by the
    stream endpoint so a client disconnect doesn't lose progress."""
    from datetime import datetime, timezone
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        return False
    book.current_sentence_idx = sentence_idx
    book.current_page = page
    book.last_read_at = datetime.now(timezone.utc)
    return True


async def update_book_sentence_progress(
    db: AsyncSession, book_id: str, sentence_idx: int
) -> bool:
    """Persist ONLY the read-along sentence cursor (used by the stream
    endpoint's per-sentence updates). Deliberately does NOT touch
    current_page: the page the user is actually viewing is owned by the
    reader overlay's progress endpoint. A stream finishing late — or a
    stale stream saved from a past session — must not yank the resume
    page backwards (e.g. open the book at page 100, but a leftover
    stream walks current_page back to page 1 one sentence at a time).
    """
    from datetime import datetime, timezone
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        return False
    book.current_sentence_idx = sentence_idx
    book.last_read_at = datetime.now(timezone.utc)
    return True


async def soft_delete_book(db: AsyncSession, book_id: str) -> bool:
    """Soft delete — keeps the row + file for safety, hides from list."""
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        return False
    book.is_active = 0
    # Best-effort file cleanup; failure here doesn't fail the delete.
    for _path in (book.filepath, getattr(book, "html_path", None)):
        if _path and os.path.exists(_path):
            try:
                os.remove(_path)
            except OSError as e:
                print(f"[books] could not remove {_path}: {e}")
    return True


async def set_book_sentences(
    db: AsyncSession, book_id: str, sentences: List[Dict[str, Any]], page_map: List[int],
    html_path: Optional[str] = None,
) -> bool:
    """Replace the cached extraction on a book. Used by the re-extract
    endpoint after the extraction pipeline has improved (e.g. soft-wrap
    fix) so the cached sentences can be refreshed without re-uploading.
    html_path replaces the article snapshot when given (url/text saves)."""
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        return False
    book.sentences = sentences
    book.page_map = page_map
    book.total_sentences = len(sentences)
    if html_path is not None and hasattr(book, "html_path"):
        book.html_path = html_path
    return True
