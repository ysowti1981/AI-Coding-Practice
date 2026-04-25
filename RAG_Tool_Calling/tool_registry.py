from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from RAG_Tool_Calling.tool import Tool


class ToolRegistry:
    """In-memory resgirty that supports semantic search over tools."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._tools: List[Tool] = []
        self._embeddings: Optional[np.ndarray] = None
        self.model = SentenceTransformer(model_name)

    def register(self, tool: Tool) -> None:
        """ Regirters a new tool and embeds its description."""
        if any(t.name == tool.name for t in self._tools):
            raise ValueError(f"Tool with name '{tool.name}' already exists.")

        embedding = self.model.encode(
            tool.embedding_text(),
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        self._tools.append(tool)
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

    def search(self, query: str, top_k: int = 5) -> List[Tool]:
        """rturnes the top_k tools most semantically similar to the query."""
        if not self._tools:
            return []

        query_vector = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # similarities = np.dot(self._embeddings, query_vector)
        # top_idx = np.argsort(similarities)[::-1][:top_k]

        scores = self._embeddings @ query_vector
        k = min(top_k, len(self._tools))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [self._tools[i] for i in top_idx]

    def __len__(self) -> int:
        return len(self._tools)
