from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        batch_embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        # Optional batch path: embedding once per document is N API calls, which is slow and can
        # blow through provider rate limits; when given, add_documents uses it instead.
        self._batch_embedding_fn = batch_embedding_fn
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        # ponytail: reference solution stays in-memory (Chroma is the bonus path); enough for <10k chunks.

    def _make_record(self, doc: Document, embedding: list[float] | None = None) -> dict[str, Any]:
        self._next_index += 1
        return {
            "id": f"{doc.id}#{self._next_index}",
            "content": doc.content,
            "metadata": {**doc.metadata, "doc_id": doc.metadata.get("doc_id", doc.id)},
            "embedding": self._embedding_fn(doc.content) if embedding is None else embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_vector = self._embedding_fn(query)
        scored = [
            {"id": r["id"], "content": r["content"], "metadata": r["metadata"], "score": _dot(query_vector, r["embedding"])}
            for r in records
        ]
        return sorted(scored, key=lambda r: r["score"], reverse=True)[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        With batch_embedding_fn the whole batch is embedded in a few API calls; otherwise
        fall back to one embedding_fn call per document.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return
        if self._batch_embedding_fn is None:
            self._store.extend(self._make_record(doc) for doc in docs)
            return
        vectors = self._batch_embedding_fn([doc.content for doc in docs])
        if len(vectors) != len(docs):
            raise ValueError(f"batch_embedding_fn trả về {len(vectors)} vector cho {len(docs)} tài liệu")
        self._store.extend(self._make_record(doc, vector) for doc, vector in zip(docs, vectors))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        wanted = metadata_filter or {}
        records = [r for r in self._store if all(r["metadata"].get(k) == v for k, v in wanted.items())]
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
