import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    score: float
    content: str
    source: str
    chunk_id: int


class VectorStore:
    """Lightweight vector store backed by numpy arrays."""

    def __init__(
        self,
        store_dir: Path,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.store_dir = store_dir
        self.model_name = model_name
        self.documents_path = self.store_dir / "documents.json"
        self.embeddings_path = self.store_dir / "embeddings.npy"
        self.encoder = SentenceTransformer(self.model_name)
        self._documents: List[dict] = []
        self._embeddings: np.ndarray | None = None

    @property
    def documents(self) -> List[dict]:
        if not self._documents:
            if not self.documents_path.exists():
                raise FileNotFoundError(
                    "Vector store documents not found. Run build_index() first."
                )
            self._documents = json.loads(self.documents_path.read_text())
        return self._documents

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            if not self.embeddings_path.exists():
                raise FileNotFoundError(
                    "Vector store embeddings not found. Run build_index() first."
                )
            self._embeddings = np.load(self.embeddings_path)
        return self._embeddings

    def build_index(self, pdf_paths: Sequence[Path], chunk_size: int = 160, overlap: int = 30) -> None:
        """(Re)build the vector index from the provided PDFs."""

        LOGGER.info("Building vector store with %d PDF(s).", len(pdf_paths))
        texts: List[str] = []
        documents: List[dict] = []

        for pdf_path in pdf_paths:
            pdf_path = pdf_path.resolve()
            if not pdf_path.exists():
                LOGGER.warning("PDF not found: %s", pdf_path)
                continue
            LOGGER.info("Extracting text from %s", pdf_path.name)
            raw_text = _read_pdf(pdf_path)
            chunks = _chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
            LOGGER.info(" -> %d chunks", len(chunks))
            texts.extend(chunks)
            documents.extend(
                {
                    "content": chunk,
                    "source": pdf_path.name,
                    "chunk_id": idx,
                }
                for idx, chunk in enumerate(chunks)
            )

        if not texts:
            raise RuntimeError("No text extracted from PDFs; cannot build vector store.")

        self.store_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Embedding %d chunks with %s", len(texts), self.model_name)
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        np.save(self.embeddings_path, embeddings.astype(np.float32))

        self.documents_path.write_text(json.dumps(documents, indent=2))
        # reset caches
        self._documents = documents
        self._embeddings = embeddings.astype(np.float32)
        LOGGER.info("Vector store built at %s", self.store_dir)

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        query_embedding = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        matrix = self.embeddings
        scores = np.dot(matrix, query_embedding)
        top_k_idx = scores.argsort()[::-1][:k]
        results: List[RetrievedChunk] = []
        for idx in top_k_idx:
            doc = self.documents[idx]
            results.append(
                RetrievedChunk(
                    score=float(scores[idx]),
                    content=doc["content"],
                    source=doc["source"],
                    chunk_id=doc["chunk_id"],
                )
            )
        return results


def _read_pdf(pdf_path: Path) -> str:
    text_segments: List[str] = []
    with pdf_path.open("rb") as fh:
        reader = PdfReader(fh)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_segments.append(page_text)
    text = "\n".join(text_segments)
    text = _clean_text(text)
    return text


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 160, overlap: int = 30) -> List[str]:
    """Chunk text by words keeping modest overlap."""

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ensure_vector_store(base_dir: str | os.PathLike[str], rebuild: bool = False) -> VectorStore:
    base_path = Path(base_dir)
    store_dir = base_path / "data" / "vector_store"
    vector_store = VectorStore(store_dir)

    documents_file = vector_store.documents_path
    embeddings_file = vector_store.embeddings_path

    if rebuild or not (documents_file.exists() and embeddings_file.exists()):
        pdf_paths = [
            base_path / "All Videos + Exercises.pdf",
            base_path / "Copy of Week 1 prompts.pdf",
        ]
        vector_store.build_index(pdf_paths)

    return vector_store

