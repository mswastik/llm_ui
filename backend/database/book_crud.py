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
    )
    db.add(book)
    await db.flush()
    return _book_to_dict(book, include_text=False)


async def list_books(db: AsyncSession, limit: int = 100) -> List[Dict[str, Any]]:
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


async def soft_delete_book(db: AsyncSession, book_id: str) -> bool:
    """Soft delete — keeps the row + file for safety, hides from list."""
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        return False
    book.is_active = 0
    # Best-effort file cleanup; failure here doesn't fail the delete.
    if book.filepath and os.path.exists(book.filepath):
        try:
            os.remove(book.filepath)
        except OSError as e:
            print(f"[books] could not remove {book.filepath}: {e}")
    return True
