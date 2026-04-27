"""Simple in-memory RAG pipeline over markdown knowledge base files."""

import os
from pathlib import Path

import numpy as np
from openai import OpenAI

from agent.telemetry import tracer

EMBED_MODEL = "text-embedding-3-small"
KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown file into chunks by top-level heading (## sections).

    Falls back to the entire file as one chunk if no headings are found.
    """
    chunks: list[dict] = []
    current_heading = source
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            # save previous chunk
            if current_lines:
                chunks.append({
                    "text": "\n".join(current_lines).strip(),
                    "source": source,
                    "heading": current_heading,
                })
            current_heading = line.lstrip("# ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # last chunk
    if current_lines:
        chunks.append({
            "text": "\n".join(current_lines).strip(),
            "source": source,
            "heading": current_heading,
        })

    return chunks


class RAG:
    """Retrieval-Augmented Generation over the kb/ markdown files."""

    def __init__(self, client: OpenAI | None = None):
        self._client = client or OpenAI()
        self._chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None

    # ---- indexing ----

    def build_index(self) -> None:
        """Read all kb/*.md files, chunk them, and embed."""
        with tracer.start_as_current_span("rag.build_index") as span:
            self._chunks = []
            for md_file in sorted(KB_DIR.glob("*.md")):
                text = md_file.read_text()
                self._chunks.extend(_chunk_markdown(text, md_file.name))

            span.set_attribute("rag.total_chunks", len(self._chunks))

            texts = [c["text"] for c in self._chunks]
            self._embeddings = self._embed_batch(texts)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI embeddings API."""
        with tracer.start_as_current_span("rag.embed_batch") as span:
            span.set_attribute("rag.batch_size", len(texts))
            resp = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
            vecs = [item.embedding for item in resp.data]
            return np.array(vecs, dtype=np.float32)

    # ---- retrieval ----

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most relevant KB chunks for a query."""
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query", query[:512])

            if self._embeddings is None or len(self._chunks) == 0:
                span.set_attribute("rag.documents_retrieved", 0)
                return []

            q_vec = self._embed_batch([query])[0]
            scores = _cosine_similarity(q_vec, self._embeddings)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    **self._chunks[idx],
                    "score": float(scores[idx]),
                })

            span.set_attribute("rag.documents_retrieved", len(results))
            if results:
                span.set_attribute("rag.top_score", results[0]["score"])

            return results


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm
