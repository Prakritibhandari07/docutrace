
import requests

from app.llm import generate_answer


def test_llm_returns_successful_response(monkeypatch):
    def mock_post(url, json, timeout):
        assert "localhost:11434/api/generate" in url
        assert json["model"] == "qwen2.5:3b"
        assert json["stream"] is False

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "response": "Artificial Intelligence enables machines to perform tasks."
                }

        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)

    answer = generate_answer(
        "What is Artificial Intelligence?",
        "Artificial Intelligence enables machines to perform tasks."
    )

    assert answer
    assert "Artificial Intelligence" in answer


def test_llm_handles_empty_response(monkeypatch):
    def mock_post(url, json, timeout):
        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": ""}

        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)

    answer = generate_answer("Explain AI", "No evidence available.")

    assert answer == (
        "The provided documents do not contain enough "
        "evidence to answer this question."
    )


def test_llm_handles_connection_error(monkeypatch):
    def mock_post(url, json, timeout):
        raise requests.exceptions.ConnectionError(
            "Ollama is unavailable"
        )

    monkeypatch.setattr(requests, "post", mock_post)

    answer = generate_answer("Explain AI", "Some context.")

    assert answer.startswith("Could not connect to Ollama:")


def test_llm_handles_http_error(monkeypatch):
    def mock_post(url, json, timeout):
        response = requests.Response()
        response.status_code = 500

        raise requests.exceptions.HTTPError(
            "Internal Server Error",
            response=response
        )

    monkeypatch.setattr(requests, "post", mock_post)

    answer = generate_answer("Explain AI", "Some context.")

    assert answer.startswith("An error occurred while contacting Ollama:")
