"""Tests for graph_gardener.llm — provider-agnostic LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from graph_gardener.llm import _chat_url, _validate_api_url, call


class TestChatUrl:
    def test_standard_deepseek_url(self):
        assert _chat_url("https://api.deepseek.com/v1") == \
            "https://api.deepseek.com/v1/chat/completions"

    def test_custom_base_url(self):
        assert _chat_url("https://api.groq.com/openai/v1") == \
            "https://api.groq.com/openai/v1/chat/completions"

    def test_no_trailing_slash(self):
        assert _chat_url("https://api.example.com/v1/") == \
            "https://api.example.com/v1/chat/completions"


class TestValidateApiUrl:
    def test_https_allowed(self):
        _validate_api_url("https://api.openai.com/v1")  # no raise

    def test_http_localhost_allowed(self):
        _validate_api_url("http://localhost:11434/v1")  # no raise

    def test_http_non_loopback_rejected(self):
        with pytest.raises(ValueError, match="HTTP is only allowed"):
            _validate_api_url("http://api.openai.com/v1")

    def test_ip_address_rejected(self):
        with pytest.raises(ValueError, match="IP addresses"):
            _validate_api_url("https://169.254.169.254/v1")

    def test_loopback_ip_allowed(self):
        _validate_api_url("http://127.0.0.1:11434/v1")  # no raise

    def test_no_hostname_rejected(self):
        with pytest.raises(ValueError, match="no hostname"):
            _validate_api_url("https:///path")

    def test_bad_scheme_rejected(self):
        with pytest.raises(ValueError, match="Unsupported scheme"):
            _validate_api_url("ftp://localhost/v1")


class TestCall:
    def test_missing_api_key_returns_none(self):
        result, metadata = call("system", "user prompt", api_key="", api_url="https://x.com/v1")
        assert result is None
        assert metadata is None

    def test_invalid_api_url_rejected(self):
        # HTTP on non-loopback host should be rejected
        result, metadata = call("system", "user", api_key="k", api_url="http://api.openai.com/v1")
        assert result is None
        assert metadata is None

    def test_successful_call_returns_parsed_json(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": '{"result": "success"}'},
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        with patch("graph_gardener.llm.requests.post", return_value=mock_resp):
            result, metadata = call(
                "system prompt", "user prompt",
                api_url="https://test.example.com/v1",
                api_key="test-key",
                model="test-model",
            )

        assert result is not None
        assert result["result"] == "success"
        # Metadata keys should NOT be in result
        assert "generated_at" not in result
        assert "model" not in result
        assert "tokens_in" not in result
        assert "tokens_out" not in result
        # They should be in metadata
        assert metadata is not None
        assert metadata["model"] == "test-model"
        assert metadata["tokens_in"] == 100
        assert metadata["tokens_out"] == 50
        assert "generated_at" in metadata

    def test_strips_code_fences(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": '```json\n{"key": "value"}\n```'},
            }],
            "usage": {},
        }

        with patch("graph_gardener.llm.requests.post", return_value=mock_resp):
            result, _ = call("system", "user", api_key="k", api_url="https://x.com/v1")

        assert result is not None
        assert result["key"] == "value"

    def test_http_error_returns_none(self):
        import requests as req_mod
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_mod.HTTPError("500")

        with patch("graph_gardener.llm.requests.post", return_value=mock_resp):
            result, metadata = call("s", "u", api_key="k", api_url="https://x.com/v1")

        assert result is None
        assert metadata is None
