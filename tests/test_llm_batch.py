"""Batch embedding + 429 retry tests — offline (no API key, no network, no Neo4j).

    pytest tests/test_llm_batch.py -v

These cover src/llm.py (embed_batch, _with_retry) and the optional batch path of
EmbeddingStore. They live in their own file so tests/test_base.py keeps exactly the
tests the lab ships with (41) — see SUBMISSION.md.
"""

import importlib
import os
import types
import unittest

PACKAGE_NAME = os.getenv("LAB_SOLUTION_PACKAGE", "src")
package = importlib.import_module(PACKAGE_NAME)
llm_mod = importlib.import_module(f"{PACKAGE_NAME}.llm")


class _FakeEmbeddings:
    """Stands in for openai client.embeddings: records inputs, returns one vector per input."""

    def __init__(self):
        self.inputs = []

    def create(self, model, input):
        texts = [input] if isinstance(input, str) else list(input)
        self.inputs.append(texts)
        data = [types.SimpleNamespace(index=i, embedding=[float(len(t)), float(i)])
                for i, t in enumerate(texts)]
        return types.SimpleNamespace(data=data,
                                     usage=types.SimpleNamespace(prompt_tokens=len(texts) * 7))


class _NoUsage(_FakeEmbeddings):
    """Gemini's OpenAI-compatible /embeddings returns usage=None."""

    def create(self, model, input):
        response = super().create(model, input)
        response.usage = None
        return response


class _NoIndex(_FakeEmbeddings):
    """Gemini returns data[index] = None and relies on array order, unlike OpenAI."""

    def create(self, model, input):
        response = super().create(model, input)
        for item in response.data:
            item.index = None
        return response


def _make_llm():
    """Build a MeteredLLM with fake clients — no API key, no network."""
    llm = llm_mod.MeteredLLM.__new__(llm_mod.MeteredLLM)
    llm.chat_provider = llm.embed_provider = "gemini"
    llm.chat_model_id = llm.embed_model_id = "fake-embedding"
    llm.usage = llm_mod.Usage()
    llm._embed_client = types.SimpleNamespace(embeddings=_FakeEmbeddings())
    return llm


def _rate_limit_error(message: str):
    """A real openai.RateLimitError: it needs an httpx response, not None."""
    import httpx
    from openai import RateLimitError

    request = httpx.Request("POST", "https://example.invalid/v1/embeddings")
    return RateLimitError(message, response=httpx.Response(429, request=request), body=None)


class TestBatchSizing(unittest.TestCase):
    """One API call per batch is the whole point: 176 chunks must not become 176 calls."""

    def setUp(self):
        self.llm = _make_llm()

    def _with_batch_size(self, size):
        original = llm_mod.EMBED_BATCH_SIZE
        llm_mod.EMBED_BATCH_SIZE = size
        self.addCleanup(lambda: setattr(llm_mod, "EMBED_BATCH_SIZE", original))

    def test_batch_sizes_are_capped(self):
        self._with_batch_size(10)
        self.llm.embed_batch([f"chunk {i}" for i in range(25)])
        sizes = [len(batch) for batch in self.llm._embed_client.embeddings.inputs]
        self.assertEqual(sizes, [10, 10, 5])
        self.assertEqual(self.llm.usage.calls, 3)

    def test_empty_input_makes_no_call(self):
        self.assertEqual(self.llm.embed_batch([]), [])
        self.assertEqual(self.llm.usage.calls, 0)

    def test_single_embed_still_one_call(self):
        self.assertEqual(self.llm.embed("hello"), [5.0, 0.0])
        self.assertEqual(self.llm.usage.calls, 1)


class TestVectorOrder(unittest.TestCase):
    """A batch response must map back to the input order, or the index is silently wrong."""

    def setUp(self):
        self.llm = _make_llm()

    def test_vectors_keep_input_order(self):
        self.assertEqual(self.llm.embed_batch(["a", "bb", "ccc"]),
                         [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])

    def test_vectors_keep_input_order_when_index_is_none(self):
        self.llm._embed_client.embeddings = _NoIndex()
        self.assertEqual(self.llm.embed_batch(["a", "bb", "ccc"]),
                         [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])


class TestTokenMetering(unittest.TestCase):
    """usage=None must not become in_tok=0, or Flat RAG indexing looks artificially cheap."""

    def setUp(self):
        self.llm = _make_llm()

    def test_real_usage_is_not_flagged_as_estimated(self):
        self.llm.embed_batch(["abc", "de"])
        self.assertEqual(self.llm.usage.est_input_tokens, 0)
        self.assertEqual(self.llm.usage.input_tokens, 14)   # 2 texts * 7 tokens from the fake

    def test_missing_usage_is_estimated_and_flagged(self):
        self.llm._embed_client.embeddings = _NoUsage()
        texts = ["mua bán trái phép chất ma túy", "điều 251"]
        self.llm.embed_batch(texts)
        self.assertEqual(self.llm.usage.input_tokens,
                         round(sum(len(t) for t in texts) / llm_mod.CHARS_PER_TOKEN))
        self.assertEqual(self.llm.usage.est_input_tokens, self.llm.usage.input_tokens)

    def test_estimated_tokens_sum_across_batches(self):
        self.llm._embed_client.embeddings = _NoUsage()
        original = llm_mod.EMBED_BATCH_SIZE
        llm_mod.EMBED_BATCH_SIZE = 1
        try:
            self.llm.embed_batch(["abcd", "efgh", "ijkl"])
        finally:
            llm_mod.EMBED_BATCH_SIZE = original
        self.assertEqual(self.llm.usage.calls, 3)
        # Rounded per batch, not once over the total: 3 x round(4/2.5) = 3 x 2, vs round(12/2.5) = 5.
        self.assertEqual(self.llm.usage.est_input_tokens,
                         3 * round(4 / llm_mod.CHARS_PER_TOKEN))


class TestRetryOn429(unittest.TestCase):
    """A per-minute quota resets in seconds (retry); a per-day quota does not (fail fast)."""

    def setUp(self):
        self.sleeps = []
        original = llm_mod.time.sleep
        llm_mod.time.sleep = self.sleeps.append
        self.addCleanup(lambda: setattr(llm_mod.time, "sleep", original))

    def test_retry_delay_uses_provider_hint(self):
        self.assertAlmostEqual(llm_mod._retry_delay(Exception("Please retry in 38.5s."), 0), 39.5)

    def test_retry_delay_falls_back_to_backoff(self):
        self.assertEqual(llm_mod._retry_delay(Exception("boom"), 2), 4.0)

    def test_per_minute_quota_is_retried_then_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise _rate_limit_error("Please retry in 5s.")
            return "ok"

        self.assertEqual(llm_mod._with_retry("t", flaky), "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(self.sleeps), 2)

    def test_per_day_quota_fails_fast_without_sleeping(self):
        def daily():
            raise _rate_limit_error("Please retry in 66405s.")

        with self.assertRaises(Exception):
            llm_mod._with_retry("t", daily)
        self.assertEqual(self.sleeps, [])          # 18h is not worth retrying inside a lab run


class TestEmbeddingStoreBatchPath(unittest.TestCase):
    """EmbeddingStore keeps its one-call-per-document contract when no batch fn is given."""

    DOCS = [package.Document(f"d{i}", f"nội dung {i}", {"doc_id": f"d{i}"}) for i in range(3)]

    def test_batch_fn_used_and_records_match(self):
        calls = []

        def batch_embedding_fn(texts):
            calls.append(list(texts))
            return [[float(len(t)), float(i)] for i, t in enumerate(texts)]

        store = package.EmbeddingStore("batch", batch_embedding_fn=batch_embedding_fn)
        store.add_documents(self.DOCS)
        self.assertEqual(len(calls), 1)                       # one batch, not one call per doc
        self.assertEqual(store.get_collection_size(), 3)
        for i, record in enumerate(store._store):
            self.assertEqual(record["embedding"], [float(len(record["content"])), float(i)])

    def test_falls_back_to_per_document_fn(self):
        store = package.EmbeddingStore("nobatch", embedding_fn=package._mock_embed)
        store.add_documents(self.DOCS)
        self.assertEqual(store.get_collection_size(), 3)
        self.assertEqual(len(store._store[0]["embedding"]), 64)

    def test_empty_batch_is_noop(self):
        store = package.EmbeddingStore("empty", batch_embedding_fn=lambda texts: [])
        store.add_documents([])
        self.assertEqual(store.get_collection_size(), 0)

    def test_mismatched_vector_count_raises(self):
        store = package.EmbeddingStore("bad", batch_embedding_fn=lambda texts: [[1.0]])
        with self.assertRaises(ValueError):
            store.add_documents(self.DOCS)


if __name__ == "__main__":
    unittest.main()