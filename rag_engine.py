"""Citation-first FAISS retrieval for Vayal Nanban."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "knowledge_base" / "tamil_nadu_agriculture.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class RetrievedPassage:
    """One source-labelled passage returned by vector similarity search."""

    document_id: str
    title: str
    content: str
    source_name: str
    source_url: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class RetrievalBundle:
    """Retrieved evidence plus prompt and display helpers."""

    query: str
    passages: tuple[RetrievedPassage, ...] = ()

    @classmethod
    def empty(cls, query: str = "") -> "RetrievalBundle":
        return cls(query=query, passages=())

    @property
    def document_ids(self) -> list[str]:
        return [passage.document_id for passage in self.passages]

    @property
    def source_domains(self) -> list[str]:
        domains: list[str] = []
        for passage in self.passages:
            domain = urlparse(passage.source_url).netloc
            if domain and domain not in domains:
                domains.append(domain)
        return domains

    def prompt_context(self) -> str:
        if not self.passages:
            return "No knowledge-base passage was retrieved. Say that the knowledge base is insufficient."

        blocks = []
        for index, passage in enumerate(self.passages, start=1):
            blocks.append(
                "\n".join(
                    (
                        f"[S{index}] {passage.title}",
                        f"Source: {passage.source_name}",
                        f"URL: {passage.source_url}",
                        f"Knowledge: {passage.content}",
                    )
                )
            )
        return "\n\n".join(blocks)

    def citations_markdown(self, language: str) -> str:
        if not self.passages:
            return ""
        heading = "### மீட்டெடுக்கப்பட்ட ஆதாரங்கள்" if language == "தமிழ்" else "### Retrieved sources"
        lines = [heading]
        for index, passage in enumerate(self.passages, start=1):
            lines.append(f"- [S{index}] [{passage.source_name}]({passage.source_url})")
        return "\n".join(lines)


def _validated_https_url(value: Any, document_id: str) -> str:
    url = str(value).strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Knowledge document {document_id!r} must use an HTTPS source URL")
    return url


def load_knowledge_documents(path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> list[Document]:
    """Load and validate the bundled, human-curated knowledge passages."""

    knowledge_path = Path(path)
    raw_items = json.loads(knowledge_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("The agriculture knowledge base must be a non-empty JSON list")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Every knowledge-base entry must be an object")

        document_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        source_name = str(item.get("source_name", "")).strip()
        if not document_id or not title or not content or not source_name:
            raise ValueError("Knowledge entries require id, title, content, and source_name")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate knowledge document id: {document_id}")
        seen_ids.add(document_id)

        source_url = _validated_https_url(item.get("source_url"), document_id)
        metadata = {
            "document_id": document_id,
            "title": title,
            "source_name": source_name,
            "source_url": source_url,
            "topics": list(item.get("topics", [])),
            "crops": list(item.get("crops", [])),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_retrieval_query(question: str, context: Any) -> str:
    """Enrich the farmer's question with the selected field context."""

    return (
        f"Farmer question: {question.strip()}\n"
        f"Language: {getattr(context, 'language', 'Not specified')}\n"
        f"Tamil Nadu district: {getattr(context, 'district', 'Not specified')}\n"
        f"Crop: {getattr(context, 'crop', 'Not specified')}\n"
        f"Crop stage: {getattr(context, 'stage', 'Not specified')}\n"
        f"Irrigation: {getattr(context, 'irrigation', 'Not specified')}"
    )


class RAGEngine:
    """Thin, testable wrapper around a LangChain FAISS vector store."""

    def __init__(self, vector_store: Any, knowledge_size: int):
        self._vector_store = vector_store
        self.knowledge_size = knowledge_size

    def retrieve(self, question: str, context: Any, k: int = 4) -> RetrievalBundle:
        if not question.strip():
            return RetrievalBundle.empty(question)

        retrieval_query = build_retrieval_query(question, context)
        results: Iterable[tuple[Document, float]] = self._vector_store.similarity_search_with_relevance_scores(
            retrieval_query,
            k=max(1, min(k, self.knowledge_size)),
        )

        passages: list[RetrievedPassage] = []
        seen_ids: set[str] = set()
        for document, raw_score in results:
            document_id = str(document.metadata.get("document_id", "")).strip()
            if not document_id or document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            passages.append(
                RetrievedPassage(
                    document_id=document_id,
                    title=str(document.metadata.get("title", document_id)),
                    content=document.page_content,
                    source_name=str(document.metadata.get("source_name", "Official source")),
                    source_url=str(document.metadata.get("source_url", "")),
                    score=max(0.0, min(float(raw_score), 1.0)),
                )
            )
        return RetrievalBundle(query=retrieval_query, passages=tuple(passages))


class LocalFAISSVectorStore:
    """Small, dependency-light FAISS store with cosine-similarity search."""

    def __init__(self, documents: list[Document], embeddings: Embeddings):
        if not documents:
            raise ValueError("At least one document is required for the FAISS index")

        vectors = np.asarray(
            embeddings.embed_documents([document.page_content for document in documents]),
            dtype="float32",
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(documents) or vectors.shape[1] == 0:
            raise ValueError("Embedding output has an invalid shape")

        faiss.normalize_L2(vectors)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._documents = list(documents)
        self._embeddings = embeddings

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        query_vector = np.asarray([self._embeddings.embed_query(query)], dtype="float32")
        if query_vector.ndim != 2 or query_vector.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension does not match the FAISS index")
        faiss.normalize_L2(query_vector)

        scores, indices = self._index.search(query_vector, min(max(1, k), len(self._documents)))
        results: list[tuple[Document, float]] = []
        for index, cosine_score in zip(indices[0], scores[0]):
            if index < 0:
                continue
            relevance = max(0.0, min((float(cosine_score) + 1.0) / 2.0, 1.0))
            results.append((self._documents[int(index)], relevance))
        return results


def build_rag_engine(
    api_key: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    knowledge_path: str | Path = DEFAULT_KNOWLEDGE_PATH,
) -> RAGEngine:
    """Embed the bundled corpus and build the FAISS index."""

    if not api_key.strip():
        raise ValueError("An OpenAI API key is required to build the vector index")
    documents = load_knowledge_documents(knowledge_path)
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=embedding_model,
        timeout=30,
        max_retries=1,
    )
    vector_store = LocalFAISSVectorStore(documents, embeddings)
    return RAGEngine(vector_store=vector_store, knowledge_size=len(documents))

