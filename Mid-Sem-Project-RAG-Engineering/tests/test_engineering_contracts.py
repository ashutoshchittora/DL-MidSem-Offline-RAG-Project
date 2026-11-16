"""
C14 — Engineering API Regression Tests

Fast, model-free tests using Python unittest.
"""

import importlib.util
import unittest
import uuid

from fastapi.testclient import TestClient

from api.main import app


class TestEngineeringAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health_endpoint(self):

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json().get("status"),
            "healthy"
        )

    def test_request_id_present(self):

        response = self.client.get("/health")

        request_id = response.headers.get("X-Request-ID")

        self.assertIsNotNone(request_id)
        uuid.UUID(request_id)

    def test_request_ids_unique(self):

        first = self.client.get("/health")
        second = self.client.get("/health")

        self.assertNotEqual(
            first.headers.get("X-Request-ID"),
            second.headers.get("X-Request-ID")
        )

    def test_latency_header(self):

        response = self.client.get("/health")

        elapsed = float(
            response.headers["X-Process-Time"]
        )

        self.assertGreaterEqual(elapsed, 0)

    def test_missing_question_rejected(self):

        response = self.client.post(
            "/answer",
            json={"top_k": 1}
        )

        self.assertEqual(response.status_code, 422)

    def test_empty_question_rejected(self):

        response = self.client.post(
            "/answer",
            json={"question": "", "top_k": 1}
        )

        self.assertEqual(response.status_code, 422)

    def test_whitespace_question_rejected(self):

        response = self.client.post(
            "/answer",
            json={"question": "   ", "top_k": 1}
        )

        self.assertEqual(response.status_code, 422)

    def test_invalid_retrieval_depth_rejected(self):

        for k in [0, -1, 2, 4, 6, True]:

            with self.subTest(top_k=k):

                response = self.client.post(
                    "/answer",
                    json={
                        "question": "Who is Diana Richburg?",
                        "top_k": k
                    }
                )

                self.assertEqual(
                    response.status_code,
                    422
                )

    def test_non_string_question_rejected(self):

        for question in [None, 123, [], {}]:

            with self.subTest(question=question):

                response = self.client.post(
                    "/answer",
                    json={
                        "question": question,
                        "top_k": 1
                    }
                )

                self.assertEqual(
                    response.status_code,
                    422
                )

    def test_malformed_json_rejected(self):

        response = self.client.post(
            "/answer",
            content='{"question":',
            headers={
                "Content-Type": "application/json"
            }
        )

        self.assertEqual(response.status_code, 422)

    def test_unknown_route(self):

        response = self.client.get(
            "/c14-nonexistent-route"
        )

        self.assertEqual(response.status_code, 404)

        self.assertTrue(
            response.headers.get("X-Request-ID")
        )

    def test_engineering_modules_available(self):

        modules = [
            "api.observability",
            "adapters.rag_adapter",
            "adapters.semantic_rag_adapter",
            "retrieval.semantic_retriever",
            "retrieval.build_chroma_index",
            "evaluation.answer_comparison",
            "evaluation.retrieval_comparison"
        ]

        for module in modules:

            with self.subTest(module=module):

                self.assertIsNotNone(
                    importlib.util.find_spec(module)
                )


if __name__ == "__main__":
    unittest.main()