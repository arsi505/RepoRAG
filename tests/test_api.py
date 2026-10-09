from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from reporag.api.app import app
from reporag.api.dependencies import APIRequestError, get_qa_executor
from reporag.qa.models import Evidence, QAResult
from reporag.qa.service import QAServiceError


def qa_result(*, citations_valid: bool = True) -> QAResult:
    evidence = (
        Evidence(
            source_id="S1",
            chunk_id="chunk-1",
            retrieval_rank=7,
            repository="ExampleRepo",
            file_path="src/auth.py",
            language="Python",
            chunk_type="function",
            symbol_name="authenticate",
            parent_symbol=None,
            start_line=12,
            end_line=24,
            content="def authenticate(token):\n    return verify(token)\n",
        ),
    )
    return QAResult(
        question="Where is authentication enforced?",
        answer_text="Authentication is enforced here [S1].",
        retrieval_method="hybrid",
        context_k=5,
        evidence_scope="all",
        cited_source_ids=("S1",),
        unknown_source_ids=(),
        citations_valid=citations_valid,
        evidence=evidence,
        context_text="context",
        qa_fingerprint="fingerprint",
        provider="deepseek",
        model="deepseek-flash",
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "total_tokens": 120,
        },
        generation_latency_seconds=1.25,
    )


class FakeExecutor:
    def __init__(self, result: QAResult | None = None, error: Exception | None = None):
        self.result = result or qa_result()
        self.error = error
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def payload() -> dict[str, object]:
    return {
        "repository": "D:\\Repositories\\ExampleRepo",
        "question": "Where is authentication enforced?",
        "retrieval_method": "hybrid",
        "evidence_scope": "all",
        "provider": "deepseek",
        "context_k": 5,
        "max_output_tokens": 2000,
    }


class RepoRAGAPITests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.client.close()

    def use_executor(self, executor: FakeExecutor) -> None:
        app.dependency_overrides[get_qa_executor] = lambda: executor

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "RepoRAG"})

    def test_valid_qa_request(self) -> None:
        self.use_executor(FakeExecutor())
        response = self.client.post("/api/qa", json=payload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Authentication is enforced here [S1].")
        self.assertEqual(body["repository"], "ExampleRepo")
        self.assertEqual((body["provider"], body["model"]), ("deepseek", "deepseek-flash"))
        self.assertEqual(body["generation_latency_seconds"], 1.25)
        self.assertEqual(body["usage"]["reasoning_tokens"], 5)

    def test_default_values(self) -> None:
        executor = FakeExecutor()
        self.use_executor(executor)
        response = self.client.post(
            "/api/qa",
            json={
                "repository": "D:\\Repositories\\ExampleRepo",
                "question": "Where is authentication enforced?",
            },
        )
        self.assertEqual(response.status_code, 200)
        request = executor.requests[0]
        self.assertEqual(request.retrieval_method, "hybrid")
        self.assertEqual(request.evidence_scope, "all")
        self.assertEqual(request.provider, "deepseek")
        self.assertEqual(request.context_k, 5)
        self.assertEqual(request.max_output_tokens, 2000)

    def test_invalid_retrieval_method(self) -> None:
        value = payload()
        value["retrieval_method"] = "unknown"
        response = self.client.post("/api/qa", json=value)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "invalid_request")

    def test_invalid_evidence_scope(self) -> None:
        value = payload()
        value["evidence_scope"] = "other"
        response = self.client.post("/api/qa", json=value)
        self.assertEqual(response.status_code, 422)

    def test_invalid_provider(self) -> None:
        value = payload()
        value["provider"] = "other"
        response = self.client.post("/api/qa", json=value)
        self.assertEqual(response.status_code, 422)

    def test_missing_repository(self) -> None:
        value = payload()
        value["repository"] = str(Path.cwd() / "definitely-missing-repository")
        response = self.client.post("/api/qa", json=value)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "repository_not_found")

    def test_missing_artifacts(self) -> None:
        test_root = Path.cwd() / ".test_tmp"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            value = payload()
            value["repository"] = directory
            response = self.client.post("/api/qa", json=value)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "repository_not_indexed")
        self.assertIn("index the repository first", response.json()["detail"])

    def test_missing_provider_key(self) -> None:
        executor = FakeExecutor(
            error=QAServiceError(
                "Generation failed: DEEPSEEK_API_KEY is not set; configure it"
            )
        )
        self.use_executor(executor)
        response = self.client.post("/api/qa", json=payload())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "provider_key_missing")

    def test_provider_error_is_sanitized(self) -> None:
        secret = "secret-provider-payload"
        self.use_executor(FakeExecutor(error=QAServiceError(secret)))
        response = self.client.post("/api/qa", json=payload())
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["code"], "generation_failed")
        self.assertNotIn(secret, response.text)

    def test_incomplete_generation_error(self) -> None:
        self.use_executor(
            FakeExecutor(
                error=QAServiceError(
                    "Generation failed: Generation reached the configured output-token limit."
                )
            )
        )
        response = self.client.post("/api/qa", json=payload())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "generation_output_limit")
        self.assertIn("larger max_output_tokens", response.json()["detail"])

    def test_source_metadata_serialization(self) -> None:
        self.use_executor(FakeExecutor())
        source = self.client.post("/api/qa", json=payload()).json()["sources"][0]
        self.assertEqual(
            source,
            {
                "id": "S1",
                "file_path": "src/auth.py",
                "symbol": "authenticate",
                "parent_symbol": None,
                "start_line": 12,
                "end_line": 24,
                "language": "Python",
                "chunk_type": "function",
                "retrieval_rank": 7,
                "content": "def authenticate(token):\n    return verify(token)\n",
            },
        )

    def test_citations_valid_serialization(self) -> None:
        self.use_executor(FakeExecutor(result=qa_result(citations_valid=False)))
        body = self.client.post("/api/qa", json=payload()).json()
        self.assertFalse(body["citations_valid"])
        self.assertEqual(body["cited_source_ids"], ["S1"])

    def test_api_key_or_authorization_data_is_not_serialized(self) -> None:
        secret = "Bearer never-return-this-value"
        self.use_executor(FakeExecutor())
        response = self.client.post(
            "/api/qa",
            json=payload(),
            headers={"Authorization": secret},
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(secret, response.text)
        self.assertNotIn("api_key", response.text.casefold())

    def test_no_evidence_error(self) -> None:
        self.use_executor(
            FakeExecutor(
                error=APIRequestError(
                    404,
                    "no_evidence",
                    "No repository evidence matched the request.",
                )
            )
        )
        response = self.client.post("/api/qa", json=payload())
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "no_evidence")

    def test_cors_is_restricted_to_local_frontend_origins(self) -> None:
        for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/api/qa",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], origin)

        denied = self.client.options(
            "/api/qa",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertNotIn("access-control-allow-origin", denied.headers)


if __name__ == "__main__":
    unittest.main()
