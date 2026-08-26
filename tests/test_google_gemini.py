"""
PROJECT:
-------
LLMTool

TITLE:
------
test_google_gemini.py

MAIN OBJECTIVE:
---------------
Guard the Google Gemini integration: provider inference, credential resolution,
catalogue wiring, and the client's fatal-versus-transient error policy.

The offline tests are the ones that catch regressions in CI. The live tests are
skipped unless GOOGLE_API_KEY (or GEMINI_API_KEY) is set, so a contributor
without a key still gets a meaningful signal.

Author:
-------
Antoine Lemor
"""

import os

import pytest

from llm_tool.__main__ import infer_provider


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.6-flash", "google"),
        ("gemini-3.5-flash-lite", "google"),
        ("models/gemini-pro-latest", "google"),
        ("GEMINI-3.7-FLASH", "google"),
        ("gpt-4o-mini", "openai"),
        ("o3", "openai"),
        ("claude-sonnet-4-20250514", "anthropic"),
        ("llama3.2:3b", "ollama"),
        ("qwen2.5:7b-instruct", "ollama"),
        ("", "ollama"),
        (None, "ollama"),
    ],
)
def test_infer_provider(model, expected):
    """A model name alone has to route to the right provider.

    Without this, `--annotate data.csv --model gemini-3.6-flash` fell through to
    Ollama and died trying to pull "gemini-3.6-flash" from the Ollama registry.
    """
    assert infer_provider(model) == expected


# ---------------------------------------------------------------------------
# Catalogue wiring
# ---------------------------------------------------------------------------

def test_google_models_are_offered_to_the_picker():
    """The picker's bucket map must contain Gemini, or no menu ever shows it."""
    from llm_tool.cli.advanced_cli import LLMDetector

    buckets = LLMDetector.detect_all_llms()
    assert "google" in buckets

    models = LLMDetector.detect_google_models()
    assert models, "the Gemini catalogue must not be empty"
    assert all(m.provider == "google" for m in models)
    assert all(m.requires_api_key for m in models)
    # Schema-constrained JSON is what the annotator relies on for parseable output.
    assert all(m.supports_json for m in models)


def test_every_catalogued_model_has_a_description():
    """A missing description renders as a blank cell in the picker table."""
    from llm_tool.cli.advanced_cli import MODEL_DESCRIPTIONS, LLMDetector

    for model in LLMDetector.detect_google_models():
        assert model.name in MODEL_DESCRIPTIONS, f"no description for {model.name}"


def test_annotator_routes_google_to_the_api_client():
    """`google` must be in the provider set that builds an API client."""
    import inspect

    from llm_tool.annotators import llm_annotator

    source = inspect.getsource(llm_annotator)
    assert "'openai', 'anthropic', 'google'" in source


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

def test_gemini_api_key_alias_is_accepted(monkeypatch):
    """Google's own quickstarts set GEMINI_API_KEY, not GOOGLE_API_KEY."""
    from llm_tool.config.api_key_manager import APIKeyManager

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-alias-key")

    assert APIKeyManager().get_key("google") == "test-alias-key"


def test_google_api_key_wins_over_the_alias(monkeypatch):
    """The documented name takes precedence when both are set."""
    from llm_tool.config.api_key_manager import APIKeyManager

    monkeypatch.setenv("GOOGLE_API_KEY", "primary")
    monkeypatch.setenv("GEMINI_API_KEY", "alias")

    assert APIKeyManager().get_key("google") == "primary"


# ---------------------------------------------------------------------------
# Client error policy
# ---------------------------------------------------------------------------

def test_missing_key_fails_fast():
    """An empty credential must raise, not retry a corpus worth of rows."""
    pytest.importorskip("google.genai")
    from llm_tool.annotators.api_clients import GoogleClient, GoogleFatalError

    with pytest.raises(GoogleFatalError):
        GoogleClient(api_key="")


@pytest.mark.parametrize(
    "status,fatal",
    [(400, True), (401, True), (403, True), (404, True), (429, False), (500, False), (503, False)],
)
def test_fatal_classification(status, fatal):
    """429 and 5xx are worth retrying; 4xx auth/not-found are not.

    Gemini's `-latest` aliases answer 503 under load and succeed on the next
    attempt, so treating those as fatal would abandon a run for nothing.
    """
    pytest.importorskip("google.genai")
    from llm_tool.annotators.api_clients import GoogleClient

    error = Exception(f"{status} error")
    error.code = status
    assert GoogleClient._is_fatal(error) is fatal


def test_retired_model_message_is_fatal():
    """Google reports a retired generation in prose, without a usable status."""
    pytest.importorskip("google.genai")
    from llm_tool.annotators.api_clients import GoogleClient

    error = Exception("This model models/gemini-2.5-flash is no longer available to new users.")
    assert GoogleClient._is_fatal(error) is True


# ---------------------------------------------------------------------------
# Live tests (opt-in)
# ---------------------------------------------------------------------------

LIVE_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
live = pytest.mark.skipif(not LIVE_KEY, reason="set GOOGLE_API_KEY to run live Gemini tests")


@live
def test_live_generation_returns_text():
    pytest.importorskip("google.genai")
    from llm_tool.annotators.api_clients import create_api_client

    client = create_api_client("google", LIVE_KEY, model="gemini-3.6-flash")
    out = client.generate("Reply with exactly: OK", temperature=0)

    assert out and "OK" in out
    assert client.last_usage and client.last_usage["prompt_tokens"] > 0


@live
def test_live_json_schema_mode():
    """Schema-constrained output is what makes annotations parseable."""
    pytest.importorskip("google.genai")
    import json

    from llm_tool.annotators.api_clients import create_api_client

    client = create_api_client("google", LIVE_KEY, model="gemini-3.6-flash")
    out = client.generate(
        "Sentiment of: 'Ce film etait magnifique.'",
        temperature=0,
        json_mode=True,
        json_schema={
            "type": "object",
            "properties": {"sentiment": {"type": "string"}},
            "required": ["sentiment"],
        },
    )

    parsed = json.loads(out)
    assert parsed["sentiment"].lower() in {"positive", "positif"}


@live
def test_live_rejected_key_is_fatal():
    pytest.importorskip("google.genai")
    from llm_tool.annotators.api_clients import create_api_client, GoogleFatalError

    client = create_api_client("google", "AIzaNotARealKey", model="gemini-3.6-flash")
    with pytest.raises(GoogleFatalError):
        client.generate("hi")


@live
def test_live_model_listing():
    pytest.importorskip("google.genai")
    from llm_tool.annotators.api_clients import create_api_client

    names = create_api_client("google", LIVE_KEY).list_models()

    assert names, "a valid key should list at least one Gemini model"
    assert all(n.startswith("gemini") for n in names)
    # Image, TTS and embedding variants cannot serve a text annotation.
    assert not any(t in n for n in names for t in ("tts", "image", "embedding"))
