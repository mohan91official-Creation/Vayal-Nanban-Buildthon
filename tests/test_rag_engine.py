import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from farmer_assistant import FarmerContext
from rag_engine import (
    RAGEngine,
    RetrievalBundle,
    LocalFAISSVectorStore,
    build_retrieval_query,
    load_knowledge_documents,
)


class KeywordEmbeddings(Embeddings):
    @staticmethod
    def _vector(text):
        normalized = text.casefold()
        return [
            float(normalized.count("water") + normalized.count("irrigat")),
            float(normalized.count("pesticide") + normalized.count("pest")),
            float(normalized.count("scheme") + normalized.count("subsidy")),
            0.1,
        ]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_query = ""
        self.last_k = 0

    def similarity_search_with_relevance_scores(self, query, k):
        self.last_query = query
        self.last_k = k
        return self.results[:k]


class RAGEngineTests(unittest.TestCase):
    def setUp(self):
        self.context = FarmerContext(
            language="English",
            district="Thanjavur",
            crop="Paddy / நெல்",
            stage="Vegetative / வளர்ச்சி",
            irrigation="Canal / கால்வாய்",
        )

    def test_bundled_knowledge_base_is_source_labelled_and_substantial(self):
        documents = load_knowledge_documents()
        self.assertGreaterEqual(len(documents), 15)
        self.assertEqual(len({doc.metadata["document_id"] for doc in documents}), len(documents))
        for document in documents:
            self.assertTrue(document.page_content)
            self.assertTrue(document.metadata["source_name"])
            self.assertTrue(document.metadata["source_url"].startswith("https://"))

    def test_retrieval_query_contains_the_selected_field_context(self):
        query = build_retrieval_query("How should I irrigate?", self.context)
        self.assertIn("Thanjavur", query)
        self.assertIn("Paddy / நெல்", query)
        self.assertIn("Vegetative", query)
        self.assertIn("Canal", query)

    def test_vector_results_become_citable_passages(self):
        document = Document(
            page_content="Check root-zone moisture before changing irrigation.",
            metadata={
                "document_id": "water-1",
                "title": "Root-zone check",
                "source_name": "Official irrigation guide",
                "source_url": "https://example.gov.in/irrigation",
            },
        )
        store = FakeVectorStore([(document, 0.87)])
        engine = RAGEngine(store, knowledge_size=1)

        bundle = engine.retrieve("How should I irrigate?", self.context, k=4)

        self.assertEqual(store.last_k, 1)
        self.assertIn("Thanjavur", store.last_query)
        self.assertEqual(bundle.document_ids, ["water-1"])
        self.assertEqual(bundle.passages[0].score, 0.87)
        self.assertIn("[S1] Root-zone check", bundle.prompt_context())
        self.assertIn("https://example.gov.in/irrigation", bundle.citations_markdown("English"))

    def test_real_faiss_index_returns_the_semantically_matching_passage(self):
        documents = [
            Document(page_content="Check water and irrigation moisture.", metadata={"id": "water"}),
            Document(page_content="Use pesticide only after pest identification.", metadata={"id": "pest"}),
            Document(page_content="Verify scheme and subsidy eligibility.", metadata={"id": "scheme"}),
        ]
        store = LocalFAISSVectorStore(documents, KeywordEmbeddings())
        results = store.similarity_search_with_relevance_scores("irrigation water", k=2)
        self.assertEqual(results[0][0].metadata["id"], "water")
        self.assertGreater(results[0][1], results[1][1])

    def test_empty_question_skips_vector_search(self):
        store = FakeVectorStore([])
        engine = RAGEngine(store, knowledge_size=1)
        result = engine.retrieve("   ", self.context)
        self.assertEqual(result, RetrievalBundle.empty("   "))
        self.assertEqual(store.last_query, "")

    def test_invalid_or_non_https_source_is_rejected(self):
        entry = [
            {
                "id": "bad-source",
                "title": "Bad source",
                "content": "Content",
                "source_name": "Unknown",
                "source_url": "http://example.com",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(json.dumps(entry), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_knowledge_documents(path)


if __name__ == "__main__":
    unittest.main()

