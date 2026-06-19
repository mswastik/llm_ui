"""
RAG (Retrieval-Augmented Generation) Service for document querying.
"""

import os
import json
import asyncio
import sqlite3
import numpy as np
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from tools.base import get_embedding, rerank

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
        
        elif file_type == "pdf" and HAS_PYPDF2:
            return DocumentProcessor._extract_pdf_text(filepath)
        
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
    def _extract_pdf_text(filepath: str) -> str:
        text = []
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
        return "\n\n".join(text)
    
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
    """Split text into overlapping chunks"""
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Tuple[str, int, int]]:
        words = text.split()
        chunks = []
        if len(words) <= chunk_size:
            return [(text, 0, len(text))]
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            char_start = len(" ".join(words[:start])) + (1 if start > 0 else 0)
            char_end = char_start + len(chunk_text)
            chunks.append((chunk_text, char_start, char_end))
            start = end - overlap
            if start >= len(words) - overlap:
                break
        return chunks


class EmbeddingStore:
    """Store and retrieve document embeddings"""
    
    def __init__(self, db_path: str = "llm_ui.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
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
        conn.commit()
        conn.close()
    
    def store_chunks(self, document_id: str, chunks: List[Tuple[str, int, int]], embeddings: List[np.ndarray]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM document_embeddings WHERE chunk_id IN "
            "(SELECT id FROM document_chunks WHERE document_id = ?)",
            (document_id,)
        )
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        for i, ((content, start_char, end_char), embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO document_chunks (id, document_id, chunk_index, content, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, document_id, i, content, start_char, end_char))
            cursor.execute("INSERT INTO document_embeddings (chunk_id, embedding) VALUES (?, ?)",
                           (chunk_id, embedding.tobytes()))
        conn.commit()
        conn.close()
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, chunk_index, content, start_char, end_char
            FROM document_chunks WHERE document_id = ? ORDER BY chunk_index
        """, (document_id,))
        chunks = [{"id": r[0], "chunk_index": r[1], "content": r[2], "start_char": r[3], "end_char": r[4]}
                   for r in cursor.fetchall()]
        conn.close()
        return chunks
    
    def delete_document_chunks(self, document_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM document_embeddings WHERE chunk_id IN "
            "(SELECT id FROM document_chunks WHERE document_id = ?)", (document_id,))
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        conn.commit()
        conn.close()
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10,
                       document_ids: List[str] = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            cursor.execute(f"""
                SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                       dc.start_char, dc.end_char, de.embedding
                FROM document_chunks dc
                JOIN document_embeddings de ON dc.id = de.chunk_id
                WHERE dc.document_id IN ({placeholders})
            """, document_ids)
        else:
            cursor.execute("""
                SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                       dc.start_char, dc.end_char, de.embedding
                FROM document_chunks dc
                JOIN document_embeddings de ON dc.id = de.chunk_id
            """)
        results = []
        for row in cursor.fetchall():
            chunk_id, doc_id, chunk_idx, content, start_char, end_char, emb_blob = row
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            results.append({
                "chunk_id": chunk_id, "document_id": doc_id, "chunk_index": chunk_idx,
                "content": content, "start_char": start_char, "end_char": end_char,
                "similarity": float(similarity)
            })
        conn.close()
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


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
            chunks = self.chunker.chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            if not chunks:
                return {"success": False, "error": "Document could not be chunked"}
            if progress_callback:
                progress_callback(f"Generating embeddings for {len(chunks)} chunks...", 50)
            embeddings = []
            for i, (chunk_text, _, _) in enumerate(chunks):
                embedding = await self._get_embedding(chunk_text)
                embeddings.append(embedding)
                if progress_callback and i % 5 == 0:
                    progress_callback(f"Embedding chunk {i+1}/{len(chunks)}...", 50 + int(40 * i / len(chunks)))
            if progress_callback:
                progress_callback("Storing embeddings...", 95)
            self.store.store_chunks(document_id, chunks, embeddings)
            return {"success": True, "chunk_count": len(chunks), "total_chars": len(text)}
        except Exception as e:
            print(f"process_document failed: {e}")
            import traceback; traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def query(self, query: str, document_ids: List[str] = None, top_k: int = None, progress_callback=None) -> Dict:
        """Query documents using semantic search."""
        try:
            top_k = top_k or self.max_chunks
            if progress_callback:
                progress_callback("Generating query embedding...", 20)
            query_embedding = await self._get_embedding(query)
            if progress_callback:
                progress_callback("Searching documents...", 50)
            results = self.store.search_similar(query_embedding, top_k=top_k * 2, document_ids=document_ids)
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
                sources.append({
                    "id": i, "type": "chunk",
                    "title": f"Document {result.get('document_id', 'Unknown')} - Chunk {result.get('chunk_index', i)}",
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
            context += f"### Result {i} (relevance: {similarity:.2f})\n\n{content} [{i}]\n\n---\n\n"
        return context
    
    def delete_document(self, document_id: str):
        self.store.delete_document_chunks(document_id)


# Tool definition for LLM function calling
RAG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_documents",
        "description": "Search through uploaded documents to find relevant information. Use this when the user asks about content from their uploaded files or documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document content"
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
