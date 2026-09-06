"""
RAG (Retrieval-Augmented Generation) Service for document querying.
"""

import asyncio
import sqlite3
import re
import numpy as np
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from tools.base import get_embedding, rerank

# Heading patterns for structure-aware chunking
_MD_HEADING = re.compile(r'^\s*#{1,6}\s+\S.*$')
_CHAPTER_HEADING = re.compile(r'^\s*chapter\b.*$', re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r'^\s*\d+(\.\d+)*\.?\s+[A-Z].*$')
_UNDERLINE_HEADING = re.compile(r'^\s*(=+|-+)\s*$')
_PAGE_NUMBER = re.compile(r'^\s*\d{1,6}\s*$')


@dataclass
class Chunk:
    """A chunk of document text with its inherited section context."""
    content: str
    start_char: int
    end_char: int
    section: Optional[str]

# Try to import PDF extraction
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# Try to import docx
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class DocumentProcessor:
    """Process various document formats and extract text"""
    
    @staticmethod
    def extract_text(filepath: str, file_type: str) -> str:
        """Extract text from a document based on its type"""
        if file_type == "text" or filepath.endswith(('.txt', '.md')):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif file_type == "pdf":
            if not HAS_PYPDF2:
                raise ValueError("PDF extraction requires PyPDF2 — install it (pip install PyPDF2)")
            _, text = DocumentProcessor._extract_pdf(filepath)
            return text
        
        elif file_type == "document" and HAS_DOCX:
            return DocumentProcessor._extract_docx_text(filepath)
        
        elif file_type == "data":
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except:
                return ""
    
    @staticmethod
    def _extract_pdf_poppler(filepath: str) -> Optional[List[str]]:
        """Extract per-page text via poppler's `pdftotext` (system binary).

        Returns None when poppler is missing or fails — the caller falls
        back to PyPDF2. Preferred when available: for print-typeset PDFs
        PyPDF2 inserts spurious spaces between letters ("M onk e y l uv"),
        which breaks sentence splitting, citation stripping, the TTS
        read-aloud and the highlight match. pdftotext reads the same file
        cleanly. Single subprocess call; pages split on form feeds.
        ponytail: no new dep (binary probe + stdlib subprocess only).
        """
        import shutil
        import subprocess
        if not shutil.which("pdftotext"):
            return None
        try:
            # Reading-order text (no -layout: avoids alignment padding).
            # '-' writes to stdout; Lambda-free list args, no shell.
            proc = subprocess.run(
                ["pdftotext", filepath, "-"],
                capture_output=True,
                timeout=180,
            )
            if proc.returncode != 0:
                return None
            full = proc.stdout.decode("utf-8", errors="ignore")
            if not full.strip():
                return None
            pages = [p.strip() for p in full.split("\x0c")]
            # Poppler ends output with a form feed → drop the trailing
            # empty, but keep interior blanks so page numbers stay aligned.
            while pages and not pages[-1]:
                pages.pop()
            return pages or None
        except Exception:
            return None

    @staticmethod
    def _extract_pdf(filepath: str) -> Tuple[List[str], str]:
        raw_pages = DocumentProcessor._extract_pdf_poppler(filepath)
        if raw_pages is None:
            raw_pages = []
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    raw_pages.append(page.extract_text() or "")
        page_texts = []
        for raw in raw_pages:
                # PyPDF2 sometimes returns text with 2+ spaces between letters
                # (e.g. "A L S O  B Y" for "ALSO BY" in print-typeset books).
                # Collapse them — single spaces are real word boundaries.
                # This applies to BOTH RAG and the read-aloud reader.
                cleaned = re.sub(r' {2,}', ' ', raw)
                # PyPDF2 also spaces out individual letters for some print-
                # typeset PDFs (e.g. 'A L S O  B Y'). Glued runs of
                # single-letter tokens back together. Edge case: 'I A' becomes
                # 'IA' (we accept the miss — it's vanishingly rare in prose).
                cleaned = re.sub(
                    r'(?:\b[a-zA-Z]\b\s+)+\b[a-zA-Z]\b',
                    lambda m: m.group(0).replace(' ', ''),
                    cleaned,
                )
                # PyPDF2 emits a newline between every visual line of
                # the page, even when those lines are a soft-wrap of
                # the same sentence. TTS that splits on \n reads the
                # book as if every line were a new sentence, with a
                # pause after each. Collapse single newlines into
                # spaces — real paragraph breaks survive because
                # PyPDF2 emits an empty text item (a blank line) as
                # two consecutive newlines.
                cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)
                page_texts.append(cleaned)
        return page_texts, "\n\n".join(page_texts)

    @staticmethod
    def _extract_pdf_outline(filepath: str) -> List[Tuple[int, str, int]]:
        """PDF bookmarks → (level, title, char_offset) into the joined extracted text."""
        headings = []
        try:
            page_texts, _ = DocumentProcessor._extract_pdf(filepath)
            if not page_texts:
                return headings
            offsets = []
            pos = 0
            for t in page_texts:
                offsets.append(pos)
                pos += len(t) + 2  # "\n\n" separator
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                def find_title(page_text: str, title: str) -> int:
                    pattern = r'\s+'.join(re.escape(w) for w in title.split())
                    m = re.search(pattern, page_text)
                    return m.start() if m else 0

                def walk(items, depth):
                    for item in items:
                        if isinstance(item, list):
                            walk(item, depth + 1)
                            continue
                        title = " ".join((item.title or "").split())
                        if not title:
                            continue
                        try:
                            page_idx = reader.get_destination_page_number(item)
                        except Exception:
                            continue
                        if not (0 <= page_idx < len(page_texts)):
                            continue
                        headings.append((depth + 1, title,
                                         offsets[page_idx] + find_title(page_texts[page_idx], title)))

                walk(reader.outline or [], 0)
        except Exception as e:
            print(f"Error extracting PDF outline: {e}")
        return headings

    def extract_structure(self, filepath: str, file_type: str) -> List[Tuple[int, str, int]]:
        """Return an authoritative document outline: (level, title, char_offset).

        PDFs use the embedded outline (bookmarks) when present; everything else
        has no outline and relies on heuristic heading detection in the Chunker.
        """
        if file_type == "pdf":
            if not HAS_PYPDF2:
                return []
            return DocumentProcessor._extract_pdf_outline(filepath)
        return []
    
    @staticmethod
    def _extract_docx_text(filepath: str) -> str:
        text = []
        try:
            doc = DocxDocument(filepath)
            for para in doc.paragraphs:
                if para.text:
                    text.append(para.text)
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
        return "\n\n".join(text)


class Chunker:
    """Split text into overlapping chunks, preserving document structure.

    Headings (markdown, "Chapter N", numbered, underline style) start a new
    section; every chunk inherits the section heading it falls under so
    retrieval can scope to a chapter and sources show the real section.
    """

    @staticmethod
    def _is_bare_title(s: str) -> bool:
        """Heuristic for a chapter/section title without any markers:
        short, title-cased, no trailing punctuation, standalone line."""
        if len(s) < 4 or len(s) > 80:
            return False
        if s.endswith(('.', '!', '?', ':', ',', ';')):
            return False
        words = s.split()
        if not 2 <= len(words) <= 12:
            return False
        if not words[0][0].isupper():
            return False
        upper = sum(1 for w in words if w[0].isupper())
        return upper / len(words) >= 0.5

    @staticmethod
    def _is_heading(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if _MD_HEADING.match(s) or _CHAPTER_HEADING.match(s) or _NUMBERED_HEADING.match(s):
            return True
        return Chunker._is_bare_title(s)

    @staticmethod
    def _heading_level(s: str) -> int:
        """Infer outline depth: markdown '#', numbered '2.3' → level 2, else level 1."""
        m = re.match(r'^\s*(#{1,6})\s+', s)
        if m:
            return len(m.group(1))
        m = re.match(r'^\s*(\d+(\.\d+)*)\.?\s+', s)
        if m:
            return m.group(1).count('.') + 1
        return 1

    @staticmethod
    def _make_chunk(words: List[Tuple[str, int, int]], section: Optional[str]) -> Chunk:
        return Chunk(" ".join(w[0] for w in words), words[0][1], words[-1][2], section)

    @staticmethod
    def _chunk_section(section: Optional[str], buffer: List[Tuple[str, int]],
                       chunk_size: int, overlap: int) -> List[Chunk]:
        words = []
        for line, off in buffer:
            for m in re.finditer(r'\S+', line):
                words.append((m.group(), off + m.start(), off + m.end()))
        n = len(words)
        if not n:
            return []
        if n <= chunk_size:
            return [Chunker._make_chunk(words, section)]

        # Group words into sentences — a word ending in . ! ? closes one.
        # ponytail: abbreviations ("Dr.") create shorter units — smaller chunks,
        # but never a split sentence; upgrade to regex word-boundary if it matters.
        sentences = []
        cur = []
        for w in words:
            cur.append(w)
            if w[0].endswith(('.', '!', '?')):
                sentences.append(cur)
                cur = []
        if cur:
            sentences.append(cur)

        # Hard-split only a single sentence longer than a chunk (rare, unavoidable)
        units = []
        for sent in sentences:
            if len(sent) <= chunk_size:
                units.append(sent)
            else:
                for k in range(0, len(sent), chunk_size):
                    units.append(sent[k:k + chunk_size])

        # Pack whole sentences into chunks; carry `overlap` words across boundaries.
        # Chunks stay <= chunk_size + overlap (same bound as word-based chunking).
        chunks = []
        cur_chunk = []
        for unit in units:
            if cur_chunk and len(cur_chunk) + len(unit) > chunk_size:
                chunks.append(Chunker._make_chunk(cur_chunk, section))
                cur_chunk = cur_chunk[-overlap:] if overlap else []
            cur_chunk.extend(unit)
        if cur_chunk:
            chunks.append(Chunker._make_chunk(cur_chunk, section))
        return chunks

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50,
                   headings: List[Tuple[int, str, int]] = None) -> List[Chunk]:
        """Chunk text into sections.

        headings: authoritative (level, title, char_offset) entries, e.g. a PDF
        outline. When provided they define section boundaries exclusively (no
        heuristics); otherwise headings are detected heuristically from the text.
        """
        lines = text.split('\n')
        records = []
        pos = 0
        for line in lines:
            records.append((line, pos))
            pos += len(line) + 1

        chunks = []
        heading_stack = []          # (level, heading_text)
        section_buffer = []         # (line, offset)
        buffer_has_content = False

        def section_path() -> str:
            return " / ".join(t for _, t in heading_stack) or None

        def apply_heading(level: int, title: str):
            nonlocal section_buffer, buffer_has_content
            if section_buffer and buffer_has_content:
                chunks.extend(Chunker._chunk_section(section_path(), section_buffer, chunk_size, overlap))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            section_buffer = []
            buffer_has_content = False

        pending = sorted(headings or [], key=lambda h: h[2])
        pi = 0
        use_outline = bool(pending)

        for i, (line, off) in enumerate(records):
            stripped = line.strip()

            # Skip standalone page numbers
            if _PAGE_NUMBER.match(stripped):
                continue

            # Outline mode: only the provided headings create section boundaries
            if use_outline:
                applied = False
                while pi < len(pending) and pending[pi][2] < off + len(line):
                    level, title, _ = pending[pi]
                    pi += 1
                    apply_heading(level, title)
                    applied = True
                section_buffer.append((line, off))
                if not applied and stripped:
                    buffer_has_content = True
                continue

            # Heuristic mode: underline heading — === / --- marks the previous line as a heading
            if _UNDERLINE_HEADING.match(stripped) and section_buffer and not Chunker._is_heading(section_buffer[-1][0]):
                heading_line, heading_off = section_buffer.pop()
                buffer_has_content = bool(section_buffer)
                apply_heading(Chunker._heading_level(heading_line), heading_line.strip().lstrip('#').strip())
                section_buffer.append((heading_line, heading_off))
                continue
            if _UNDERLINE_HEADING.match(stripped):
                continue

            if Chunker._is_heading(stripped):
                apply_heading(Chunker._heading_level(stripped), stripped.lstrip('#').strip())
                section_buffer.append((line, off))
                continue

            section_buffer.append((line, off))
            if stripped:
                buffer_has_content = True

        if section_buffer and buffer_has_content:
            chunks.extend(Chunker._chunk_section(section_path(), section_buffer, chunk_size, overlap))
        return chunks


class EmbeddingStore:
    """Store and retrieve document embeddings."""

    def __init__(self, db_path: str = "llm_ui.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # WAL mode + busy_timeout let background processing and queries run
        # concurrently without "database is locked" errors.
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_char INTEGER,
                    end_char INTEGER,
                    section TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON document_chunks(document_id)
            """)
            # Migration: add section column to pre-existing tables
            cols = {r[1] for r in cursor.execute("PRAGMA table_info(document_chunks)")}
            if "section" not in cols:
                cursor.execute("ALTER TABLE document_chunks ADD COLUMN section TEXT")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_section
                ON document_chunks(section)
            """)
            conn.commit()
        finally:
            conn.close()

    def store_chunks(self, document_id: str, chunks: List[Chunk], embeddings: List[np.ndarray]):
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM document_embeddings WHERE chunk_id IN "
                "(SELECT id FROM document_chunks WHERE document_id = ?)",
                (document_id,)
            )
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO document_chunks (id, document_id, chunk_index, content, start_char, end_char, section)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, document_id, i, chunk.content, chunk.start_char, chunk.end_char, chunk.section))
                conn.execute("INSERT INTO document_embeddings (chunk_id, embedding) VALUES (?, ?)",
                             (chunk_id, embedding.astype(np.float16).tobytes()))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete_document_chunks(self, document_id: str):
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM document_embeddings WHERE chunk_id IN "
                "(SELECT id FROM document_chunks WHERE document_id = ?)", (document_id,))
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_sections(self, document_ids: List[str] = None) -> List[str]:
        """Distinct top-level sections (no '/' in the path) for tool guidance."""
        conn = self._connect()
        try:
            q = "SELECT DISTINCT section FROM document_chunks WHERE section IS NOT NULL AND section != ''"
            params = []
            if document_ids:
                q += f" AND document_id IN ({','.join('?' * len(document_ids))})"
                params.extend(document_ids)
            return sorted(s for (s,) in conn.execute(q, params) if "/" not in s)
        finally:
            conn.close()

    def _resolve_sections(self, conn: sqlite3.Connection, section_filter: str,
                          document_ids: List[str]) -> List[str]:
        """Map a loose section reference ('Chapter 2', '2.1 Cells') to the exact
        stored section strings so the IN filter is precise.

        Direct substring match first ('The Law of Envy'), then a leading-number
        match ('Chapter 2' / 'Section 1.1' → sections starting '2.' / '1.1.').
        """
        q = "SELECT DISTINCT section FROM document_chunks"
        params = []
        if document_ids:
            q += f" WHERE document_id IN ({','.join('?' * len(document_ids))})"
            params.extend(document_ids)
        sections = [r[0] for r in conn.execute(q, params) if r[0]]
        if not sections:
            return []

        ref = section_filter.strip().lower()
        direct = [s for s in sections if ref in s.lower()]
        if direct:
            return direct

        m = re.match(r'^\s*(?:chapter|section)\s+(\S+).*$', ref)
        num = m.group(1) if m else ref.split()[0]
        if re.fullmatch(r'[\d.]+', num):
            matched = [s for s in sections if s.lower().startswith(num + ".")]
            if matched:
                return matched
        return []

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10,
                       document_ids: List[str] = None, section: str = None) -> List[Dict]:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            sql = """
                SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                       dc.start_char, dc.end_char, de.embedding, dc.section, d.filename
                FROM document_chunks dc
                JOIN document_embeddings de ON dc.id = de.chunk_id
                LEFT JOIN documents d ON d.id = dc.document_id
            """
            clauses = []
            params = []
            if document_ids:
                placeholders = ",".join("?" * len(document_ids))
                clauses.append(f"dc.document_id IN ({placeholders})")
                params.extend(document_ids)
            if section:
                matched = self._resolve_sections(conn, section, document_ids)
                if not matched:
                    # no stored section matches — caller falls back to unscoped search
                    return []
                clauses.append(f"dc.section IN ({','.join('?' * len(matched))})")
                params.extend(matched)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            cursor.execute(sql, params)

            dim = query_embedding.shape[0]
            results = []
            for row in cursor.fetchall():
                chunk_id, doc_id, chunk_idx, content, start_char, end_char, emb_blob, sec, filename = row
                # float16 blobs are half the bytes of float32; detect by length so
                # pre-float16 rows (stored as float32) keep working.
                dtype = np.float16 if len(emb_blob) == dim * 2 else np.float32
                embedding = np.frombuffer(emb_blob, dtype=dtype).astype(np.float32)
                # Guard against dimension mismatch from mixing embedding models
                if embedding.shape[0] != dim:
                    continue
                norm = np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                similarity = float(np.dot(query_embedding, embedding) / norm) if norm else 0.0
                results.append({
                    "chunk_id": chunk_id, "document_id": doc_id, "chunk_index": chunk_idx,
                    "content": content, "start_char": start_char, "end_char": end_char,
                    "section": sec, "filename": filename, "similarity": similarity
                })
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        finally:
            conn.close()


class RAGService:
    """
    Main RAG service that coordinates document processing,
    embedding generation, and retrieval.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 similarity_threshold: float = 0.3, max_chunks: int = 20, max_retries: int = 3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        self.max_retries = max_retries
        self.processor = DocumentProcessor()
        self.chunker = Chunker()
        self.store = EmbeddingStore()
    
    async def process_document(self, document_id: str, filepath: str, file_type: str, progress_callback=None) -> Dict:
        """Process a document: extract text, chunk, embed, and store."""
        try:
            if progress_callback:
                progress_callback("Extracting text from document...", 10)
            text = self.processor.extract_text(filepath, file_type)
            if not text.strip():
                return {"success": False, "error": "No text could be extracted from document"}
            if progress_callback:
                progress_callback("Chunking document...", 30)
            headings = self.processor.extract_structure(filepath, file_type)
            chunks = self.chunker.chunk_text(text, chunk_size=self.chunk_size,
                                             overlap=self.chunk_overlap, headings=headings)
            if not chunks:
                return {"success": False, "error": "Document could not be chunked"}
            if progress_callback:
                progress_callback(f"Generating embeddings for {len(chunks)} chunks...", 50)
            embeddings = []
            for i, chunk in enumerate(chunks):
                embedding = await self._get_embedding(chunk.content)
                embeddings.append(embedding)
                if progress_callback and i % 5 == 0:
                    progress_callback(f"Embedding chunk {i+1}/{len(chunks)}...", 50 + int(40 * i / len(chunks)))
            dims = {e.shape[0] for e in embeddings}
            if len(dims) > 1:
                return {"success": False, "error": f"Embedding dimension mismatch ({sorted(dims)}). Ensure the embedding model is unchanged between documents."}
            if progress_callback:
                progress_callback("Storing embeddings...", 95)
            await asyncio.to_thread(self.store.store_chunks, document_id, chunks, embeddings)
            return {"success": True, "chunk_count": len(chunks), "total_chars": len(text)}
        except Exception as e:
            print(f"process_document failed: {e}")
            import traceback; traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def query(self, query: str, document_ids: List[str] = None, top_k: int = None,
                    section: str = None, progress_callback=None) -> Dict:
        """Query documents using semantic search."""
        try:
            top_k = min(top_k or self.max_chunks, 50)
            if progress_callback:
                progress_callback("Generating query embedding...", 20)
            query_embedding = await self._get_embedding(query)
            if progress_callback:
                progress_callback("Searching documents...", 50)
            results = await asyncio.to_thread(
                self.store.search_similar, query_embedding, top_k * 2, document_ids, section)
            if not results and section:
                # Section scoping missed (LLM guessed a heading, doc re-uploaded,
                # etc.) — fall back to an unscoped search rather than returning
                # nothing. Full-book retrieval is the LLM's choice via omitting
                # 'section', so don't broaden here beyond a genuine miss.
                print(f"[RAG] Section '{section}' matched no chunks; falling back to unscoped search")
                results = await asyncio.to_thread(
                    self.store.search_similar, query_embedding, top_k * 2, document_ids, None)
            results = [r for r in results if r["similarity"] >= self.similarity_threshold]
            if not results:
                return {"results": [], "context": "No relevant information found in the documents."}
            if progress_callback:
                progress_callback("Reranking results...", 70)
            if len(results) > 1:
                reranked_indices = await self._rerank(query, [r["content"] for r in results])
                results = [results[i] for i in reranked_indices[:top_k]]
            else:
                results = results[:top_k]
            if progress_callback:
                progress_callback("Formatting results...", 90)
            context = self._format_context(results, query)
            sources = []
            for i, result in enumerate(results, 1):
                label = result.get("filename") or result.get("document_id", "Unknown")
                title = f"{result.get('section')} — {label}" if result.get("section") else f"{label} — Chunk {result.get('chunk_index', i)}"
                sources.append({
                    "id": i, "type": "chunk",
                    "title": title,
                    "url": f"#chunk-{result.get('document_id', 'unknown')}-{result.get('chunk_index', i)}",
                    "snippet": (result.get("content", "")[:300] + "...") if len(result.get("content", "")) > 300 else result.get("content", ""),
                    "chunk_content": result.get("content", "")
                })
            return {"results": results, "context": context, "sources": sources}
        except Exception as e:
            print(f"query failed: {e}")
            import traceback; traceback.print_exc()
            return {"results": [], "context": f"Error: {str(e)}"}
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        from settings import settings_manager
        settings = settings_manager.get_settings()
        model = settings.get('embedding_model', 'Qwen3-4B-Embedding')
        return await get_embedding(text, model=model, max_retries=self.max_retries)
    
    async def _rerank(self, query: str, chunks: List[str]) -> List[int]:
        from settings import settings_manager
        settings = settings_manager.get_settings()
        model = settings.get('reranking_model', 'Qwen3-4B-Embedding')
        return await rerank(query, chunks, model=model, max_retries=self.max_retries)
    
    def _format_context(self, results: List[Dict], query: str) -> str:
        context = "# 📄 Document Search Results\n\n"
        context += f"**Query:** {query}\n\n## Relevant Excerpts\n\n"
        for i, result in enumerate(results, 1):
            similarity = result.get("similarity", 0)
            content = result.get("content", "")
            section = result.get("section")
            header = f"### Result {i} (relevance: {similarity:.2f})"
            if section:
                header += f" — [{section}]"
            context += f"{header}\n\n{content} [{i}]\n\n---\n\n"
        return context
    
    async def delete_document(self, document_id: str):
        await asyncio.to_thread(self.store.delete_document_chunks, document_id)


# Tool definition for LLM function calling
RAG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_documents",
        "description": "Search through uploaded documents to find relevant information. Omit the 'section' parameter to search the whole document(s), and only provide it to narrow the search when the user explicitly asks about a specific chapter or section (e.g. 'what does chapter 3 say about...'). For broad or cross-chapter questions (e.g. 'summarize the book'), never pass 'section' so context from every chapter is found. Use this when the user asks about content from their uploaded files or documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document content. Use retrieval keywords describing the topic, not the full user request (e.g. 'key concepts' rather than 'summarize chapter 2')."
                },
                "section": {
                    "type": "string",
                    "description": "Optional chapter/section heading to narrow the search. Use the EXACT section title from the 'Available sections' list in the description (e.g. 'The Law of Narcissism'). If no exact match exists, omit it."
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific document IDs to search within"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}


if __name__ == "__main__":
    import sys
    sample = (
        "# Chapter 1\n"
        "Introduction text.\n"
        "## Section 1.1\n"
        "Detail about one.\n"
        "Chapter 2\n"
        "==========\n"
        "Body of chapter two.\n"
        "3. Results\n"
        "The findings here.\n"
    )
    # Large chunk size → each section is one chunk; headings inherited correctly
    chunks = Chunker.chunk_text(sample, chunk_size=100, overlap=1)
    sections = [c.section for c in chunks]
    assert len(chunks) == 4, f"expected 4 section chunks, got {len(chunks)}"
    assert "Chapter 1" in sections and "Chapter 1 / Section 1.1" in sections
    assert "Chapter 2" in sections and "3. Results" in sections
    joined = " ".join(c.content for c in chunks)
    assert "==========" not in joined, "underline markers leaked into content"
    assert "Detail about one." in joined and "Body of chapter two." in joined
    # Overlap on a single section produces sliding chunks covering all words
    chunks = Chunker.chunk_text("one two three four five six seven eight nine ten", chunk_size=4, overlap=2)
    assert len(chunks) == 3, f"expected 3 overlapping chunks, got {len(chunks)}"
    assert all(c.start_char < c.end_char for c in chunks)
    assert chunks[-1].content.rstrip().endswith("ten")
    # Sentences are never split by chunk boundaries
    chunks = Chunker.chunk_text("First sentence here. Second sentence. Third one here.", chunk_size=3, overlap=0)
    assert [c.content for c in chunks] == ["First sentence here.", "Second sentence.", "Third one here."], \
        f"sentences split across chunks: {[c.content for c in chunks]}"
    # Bare chapter titles (no 'Chapter' word/number) are detected, plus level 2/3 nesting
    sample = (
        "The Immune System\n"
        "\n"
        "The immune system defends the body.\n"
        "42\n"
        "2.1 Cells\n"
        "White blood cells are important.\n"
        "2.1.1 B Cells\n"
        "B cells produce antibodies.\n"
    )
    chunks = Chunker.chunk_text(sample, chunk_size=100, overlap=1)
    sections = [c.section for c in chunks]
    assert "The Immune System" in sections, f"bare title not detected: {sections}"
    assert "The Immune System / 2.1 Cells" in sections, f"level 2 not nested: {sections}"
    assert "The Immune System / 2.1 Cells / 2.1.1 B Cells" in sections, f"level 3 not nested: {sections}"
    joined = " ".join(c.content for c in chunks)
    assert "42" not in joined, "page number leaked into content"
    # Outline mode: boundaries come from the outline only, heuristics are disabled
    text = (
        "Front matter paragraph.\n"
        "Chapter One\n"
        "Body of the first chapter.\n"
        "Extra Section Title\n"
        "More details.\n"
        "Section 1.1\n"
        "Detail text here.\n"
    )
    chunks = Chunker.chunk_text(text, chunk_size=100, overlap=1,
                                headings=[(1, "Chapter One", 24), (2, "Section 1.1", 97)])
    sections = [c.section for c in chunks]
    assert None in sections, f"front matter should have no section: {sections}"
    assert "Chapter One" in sections and "Chapter One / Section 1.1" in sections, f"sections: {sections}"
    assert "Extra Section Title" not in sections, "heuristic headings must be ignored in outline mode"
    assert " ".join(c.content for c in chunks).find("Extra Section Title") != -1
    print(f"OK: structure chunks={sections}, overlap chunks={len(chunks)}")
    sys.exit(0)
