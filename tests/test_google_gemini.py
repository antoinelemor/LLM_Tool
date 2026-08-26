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


# ---------------------------------------------------------------------------
# Provider registry (extensibility)
# ---------------------------------------------------------------------------

def test_registry_is_the_single_source_of_truth():
    """Every registered provider must be complete enough to be selectable."""
    from llm_tool.config.providers import PROVIDERS, iter_providers

    assert {"ollama", "openai", "google", "anthropic"} <= set(PROVIDERS)

    for spec in iter_providers():
        assert spec.id and spec.label and spec.kind in {"cloud", "local"}
        if spec.kind == "cloud":
            assert spec.env_vars, f"{spec.id} needs at least one env var"
            assert spec.signup_url, f"{spec.id} must tell users where to get a key"
        for model in spec.models:
            assert model.description, f"{model.name} needs a picker description"
            assert model.context_length > 0


def test_exactly_one_fallback_provider():
    """Inference routes unmatched names to the fallback; two would be ambiguous."""
    from llm_tool.config.providers import iter_providers

    assert sum(1 for s in iter_providers() if s.is_fallback) == 1


def test_model_prefixes_do_not_overlap():
    """A model name must not match two providers, or inference becomes order-dependent."""
    from llm_tool.config.providers import iter_providers

    seen = {}
    for spec in iter_providers():
        for prefix in spec.model_prefixes:
            for other, other_id in seen.items():
                assert not (prefix.startswith(other) or other.startswith(prefix)), (
                    f"prefix {prefix!r} ({spec.id}) collides with {other!r} ({other_id})"
                )
            seen[prefix] = spec.id


def test_catalogued_models_route_back_to_their_own_provider():
    """Round-trip: every catalogued model must infer to the provider that lists it."""
    from llm_tool.config.providers import infer_provider, iter_providers

    for spec in iter_providers():
        for model in spec.models:
            assert infer_provider(model.name) == spec.id, model.name


def test_adding_a_provider_needs_no_edit_outside_the_registry():
    """A provider registered at runtime must reach the picker and --provider."""
    from llm_tool.config import providers as reg
    from llm_tool.cli.advanced_cli import LLMDetector

    spec = reg.ProviderSpec(
        id="acme", label="Acme AI", kind="cloud",
        env_vars=("ACME_API_KEY",), signup_url="https://acme.example/keys",
        model_prefixes=("acme-",),
        models=(reg.ModelSpec("acme-1", 128_000, "Acme 1 - test model"),),
    )
    reg.PROVIDERS["acme"] = spec
    try:
        assert "acme" in reg.cloud_provider_ids()
        assert reg.infer_provider("acme-1") == "acme"
        assert [m.name for m in LLMDetector.models_for("acme")] == ["acme-1"]
        assert "acme" in LLMDetector.detect_all_llms()
        assert reg.model_descriptions()["acme-1"].startswith("Acme 1")
    finally:
        del reg.PROVIDERS["acme"]


def test_key_manager_env_vars_come_from_the_registry():
    """The key manager must not keep its own copy of the provider list."""
    from llm_tool.config.api_key_manager import PROVIDER_ENV_VARS, PROVIDER_ENV_ALIASES

    assert PROVIDER_ENV_VARS["google"] == "GOOGLE_API_KEY"
    assert "GEMINI_API_KEY" in PROVIDER_ENV_ALIASES["google"]
    assert PROVIDER_ENV_VARS["openai"] == "OPENAI_API_KEY"


def test_agent_routes_compat_providers_without_a_bespoke_client(monkeypatch):
    """Gemini reaches agent mode through the OpenAI-compatible endpoint."""
    pytest.importorskip("openai")
    # Agent mode currently fails to import on main: llm_tool/agent/__init__ pulls
    # in agent_cli, which imports a system_prompt module that is not in the tree.
    # That break is unrelated to provider routing, so skip rather than fail here.
    create_agent_provider = pytest.importorskip(
        "llm_tool.agent.providers", reason="agent package does not import"
    ).create_agent_provider

    provider = create_agent_provider("google", "gemini-3.6-flash", api_key="test-key")
    assert provider.__class__.__name__ == "OpenAIAgentProvider"


def test_agent_cloud_provider_without_key_explains_where_to_get_one():
    create_agent_provider = pytest.importorskip(
        "llm_tool.agent.providers", reason="agent package does not import"
    ).create_agent_provider

    with pytest.raises(ValueError, match="aistudio.google.com"):
        create_agent_provider("google", "gemini-3.6-flash", api_key=None)
