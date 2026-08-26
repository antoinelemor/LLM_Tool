#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
providers.py

MAIN OBJECTIVE:
---------------
Single source of truth for the LLM providers the tool can annotate with.

Before this module, a provider was described by ~35 hardcoded literals spread
across the CLI, the agent, the key manager and the entry point --
``['openai', 'anthropic', 'google']`` here, ``{'openai', 'anthropic'}`` there,
a separate env-var map, a separate model catalogue, a separate picker branch.
Adding a provider meant finding every one of them, and missing one was silent:
Gemini was declared in three places and still never appeared in Agent mode,
where the ``else`` branch of a two-way ``if`` handed the user Anthropic's models
instead.

Everything a provider needs is now declared once, here, as data. To add a
provider, append one :class:`ProviderSpec` and implement its client; the CLI
picker, the model tables, provider inference, ``--provider``, the key manager
and the agent all pick it up without further edits.

Dependencies:
-------------
- dataclasses
- importlib.util
- os
- typing

MAIN FEATURES:
--------------
1) ProviderSpec / ModelSpec: declarative description of a provider and its models
2) PROVIDERS: the registry itself
3) infer_provider(): route a bare model name to the provider that serves it
4) provider_env_vars() / resolve_api_key(): credential lookup, aliases included
5) is_sdk_available(): whether the provider's optional SDK is importable
6) cloud_provider_ids() / iter_providers(): enumeration for menus and choices

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

__all__ = [
    "ModelSpec",
    "ProviderSpec",
    "PROVIDERS",
    "iter_providers",
    "get_provider",
    "provider_ids",
    "cloud_provider_ids",
    "infer_provider",
    "provider_env_vars",
    "resolve_api_key",
    "is_sdk_available",
    "model_descriptions",
]


@dataclass(frozen=True)
class ModelSpec:
    """
    One model offered in the pickers.

    Attributes
    ----------
    name : str
        Identifier passed to the provider's API.
    context_length : int
        Maximum input window, in tokens.
    description : str
        One line shown in the picker table.
    max_tokens : int
        Default output budget.
    prompt_cost_per_1k, completion_cost_per_1k : float, optional
        Cost per 1000 tokens. Left as None when the figure cannot be verified --
        the picker then shows "N/A", which is better than quoting a stale price.
    cached_prompt_cost_per_1k, batch_* : float, optional
        Discounted rates, where the provider offers them.
    """

    name: str
    context_length: int
    description: str = ""
    max_tokens: int = 8192
    supports_json: bool = True
    supports_streaming: bool = True
    prompt_cost_per_1k: Optional[float] = None
    completion_cost_per_1k: Optional[float] = None
    cached_prompt_cost_per_1k: Optional[float] = None
    batch_prompt_cost_per_1k: Optional[float] = None
    batch_cached_prompt_cost_per_1k: Optional[float] = None
    batch_completion_cost_per_1k: Optional[float] = None


@dataclass(frozen=True)
class ProviderSpec:
    """
    Everything the tool needs to know about one LLM provider.

    Attributes
    ----------
    id : str
        Internal identifier, as stored in run configs (``'google'``).
    label : str
        Human-readable name for menus (``'Google Gemini'``).
    kind : str
        ``'local'`` or ``'cloud'``. Local providers need no credential and are
        discovered at runtime rather than listed here.
    env_vars : tuple of str
        Environment variables consulted for the key, most canonical first. The
        extras matter: Google's own quickstarts set ``GEMINI_API_KEY``, so a user
        who followed Google's documentation would otherwise be told no key exists.
    sdk_module : str, optional
        Importable module proving the SDK is installed.
    install_extra : str, optional
        Extra that provides it, for the "how do I fix this" message.
    signup_url : str, optional
        Where to get a key.
    model_prefixes : tuple of str
        Lowercase prefixes that identify this provider from a bare model name.
    models : tuple of ModelSpec
        Static catalogue. Empty is legal: a provider can be wired up before its
        catalogue is curated, and free-text model entry still reaches it.
    is_fallback : bool
        The provider used when no prefix matches. Exactly one may set this.
    """

    id: str
    label: str
    kind: str
    env_vars: Tuple[str, ...] = ()
    sdk_module: Optional[str] = None
    install_extra: Optional[str] = None
    signup_url: Optional[str] = None
    model_prefixes: Tuple[str, ...] = ()
    models: Tuple[ModelSpec, ...] = field(default_factory=tuple)
    is_fallback: bool = False
    # Base URL of an OpenAI-compatible /chat/completions endpoint, when the
    # provider offers one. Agent mode drives tools through the OpenAI wire
    # format, so a provider with this set needs no bespoke agent client.
    openai_compat_base_url: Optional[str] = None
    default_agent_model: Optional[str] = None

    @property
    def requires_key(self) -> bool:
        """Whether a credential must be found before this provider can be used."""
        return self.kind == "cloud"


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_OLLAMA = ProviderSpec(
    id="ollama",
    label="Ollama",
    kind="local",
    # Only a remote or cloud Ollama endpoint authenticates; the local daemon
    # needs nothing, so "no key" is a valid final state for this provider.
    env_vars=("OLLAMA_API_KEY",),
    sdk_module="ollama",
    signup_url="https://ollama.com/download",
    # Ollama model names are free-form ("llama3.2:3b", "hf.co/user/repo"), so
    # they cannot be recognised by shape. This is the fallback instead.
    model_prefixes=(),
    models=(),  # discovered live from the daemon
    is_fallback=True,
    default_agent_model="llama3.2",
)

_OPENAI = ProviderSpec(
    id="openai",
    label="OpenAI",
    kind="cloud",
    env_vars=("OPENAI_API_KEY",),
    sdk_module="openai",
    install_extra=None,  # core dependency
    signup_url="https://platform.openai.com/api-keys",
    model_prefixes=("gpt-", "o1", "o3", "o4", "chatgpt", "text-davinci"),
    default_agent_model="gpt-4o",
    models=(
        ModelSpec(
            "gpt-4.1-2025-04-14", 1_047_576,
            "OpenAI GPT-4.1 - Smartest non-reasoning model, 1M context, tool calling",
            max_tokens=32768,
            prompt_cost_per_1k=0.001, completion_cost_per_1k=0.004,
            batch_prompt_cost_per_1k=0.001, batch_completion_cost_per_1k=0.004,
        ),
        ModelSpec(
            "gpt-5-2025-08-07", 200_000,
            "OpenAI GPT-5 - Flagship general-purpose model with enhanced reasoning",
            max_tokens=8000,
            prompt_cost_per_1k=0.00125, completion_cost_per_1k=0.01,
            cached_prompt_cost_per_1k=0.000125,
            batch_prompt_cost_per_1k=0.000625, batch_cached_prompt_cost_per_1k=6.25e-05,
            batch_completion_cost_per_1k=0.005,
        ),
        ModelSpec(
            "gpt-5-mini-2025-08-07", 200_000,
            "OpenAI GPT-5 Mini - Balanced GPT-5 variant, optimized for cost",
            max_tokens=4000,
            prompt_cost_per_1k=0.00025, completion_cost_per_1k=0.002,
            cached_prompt_cost_per_1k=2.5e-05,
            batch_prompt_cost_per_1k=0.000125, batch_cached_prompt_cost_per_1k=1.25e-05,
            batch_completion_cost_per_1k=0.001,
        ),
        ModelSpec(
            "gpt-5-nano-2025-08-07", 200_000,
            "OpenAI GPT-5 Nano - Ultra-fast GPT-5 tier for large batch workloads",
            max_tokens=4000,
            prompt_cost_per_1k=5e-05, completion_cost_per_1k=0.0004,
            cached_prompt_cost_per_1k=5e-06,
            batch_prompt_cost_per_1k=2.5e-05, batch_cached_prompt_cost_per_1k=2.5e-06,
            batch_completion_cost_per_1k=0.0002,
        ),
    ),
)

# Per-token costs are deliberately unset for Gemini. Google publishes pricing
# per million tokens at https://ai.google.dev/pricing and revises it per
# generation; a stale number here would be quoted by the cost estimator as if
# it were current, which is worse than the honest "N/A" the picker shows.
_GOOGLE = ProviderSpec(
    id="google",
    label="Google Gemini",
    kind="cloud",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    sdk_module="google.genai",
    install_extra="providers",
    signup_url="https://aistudio.google.com/apikey",
    model_prefixes=("gemini", "models/gemini"),
    # Google mirrors the OpenAI chat API, tool calling included (verified), so
    # agent mode reuses the OpenAI client rather than needing a Gemini one.
    openai_compat_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    default_agent_model="gemini-3.6-flash",
    models=(
        ModelSpec("gemini-3.6-flash", 1_048_576,
                  "Gemini 3.6 Flash - Fast, 1M context, native JSON schema; best default for annotation"),
        ModelSpec("gemini-3.7-flash", 1_048_576,
                  "Gemini 3.7 Flash - Newest Flash generation, 1M context"),
        ModelSpec("gemini-3.5-flash", 1_048_576,
                  "Gemini 3.5 Flash - Previous Flash generation, 1M context"),
        ModelSpec("gemini-3.5-flash-lite", 1_048_576,
                  "Gemini 3.5 Flash-Lite - Cheapest tier, for very large batches"),
        ModelSpec("gemini-3.1-pro-preview", 1_048_576,
                  "Gemini 3.1 Pro (preview) - Strongest reasoning, slower and pricier"),
        # Rolling aliases track the current generation but answer 503 under load
        # more often than a pinned id, so they are offered last.
        ModelSpec("gemini-flash-latest", 1_048_576,
                  "Gemini Flash (rolling alias) - Always the current Flash; can answer 503 under load"),
        ModelSpec("gemini-pro-latest", 1_048_576,
                  "Gemini Pro (rolling alias) - Always the current Pro; can answer 503 under load"),
    ),
)

_ANTHROPIC = ProviderSpec(
    id="anthropic",
    label="Anthropic",
    kind="cloud",
    env_vars=("ANTHROPIC_API_KEY",),
    sdk_module="anthropic",
    install_extra="providers",
    signup_url="https://console.anthropic.com/settings/keys",
    model_prefixes=("claude",),
    default_agent_model="claude-sonnet-4-20250514",
    # Client is implemented; the catalogue is intentionally empty until the
    # models have been exercised through the annotation pipeline. Free-text
    # entry still reaches this provider in the meantime.
    models=(),
)

PROVIDERS: Dict[str, ProviderSpec] = {
    p.id: p for p in (_OLLAMA, _OPENAI, _GOOGLE, _ANTHROPIC)
}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def iter_providers(kind: Optional[str] = None) -> Iterator[ProviderSpec]:
    """
    Yield registered providers, optionally filtered by kind.

    Parameters
    ----------
    kind : str, optional
        ``'cloud'`` or ``'local'``. All providers when omitted.

    Yields
    ------
    ProviderSpec
    """
    for spec in PROVIDERS.values():
        if kind is None or spec.kind == kind:
            yield spec


def get_provider(provider_id: str) -> Optional[ProviderSpec]:
    """Return the spec for `provider_id`, or None if it is not registered."""
    return PROVIDERS.get((provider_id or "").strip().lower())


def provider_ids() -> Tuple[str, ...]:
    """Every registered provider id, in registration order."""
    return tuple(PROVIDERS)


def cloud_provider_ids() -> Tuple[str, ...]:
    """Ids of the providers that need an API key."""
    return tuple(p.id for p in iter_providers(kind="cloud"))


def infer_provider(model_name: Optional[str]) -> str:
    """
    Guess which provider serves `model_name`.

    Parameters
    ----------
    model_name : str, optional
        A model identifier as typed on the command line.

    Returns
    -------
    str
        A registered provider id; the fallback provider when nothing matches.

    Notes
    -----
    Without this, ``--annotate data.csv --model gemini-3.6-flash`` fell through
    to Ollama and the run died trying to pull "gemini-3.6-flash" from the Ollama
    registry.

    Examples
    --------
    >>> infer_provider('gemini-3.6-flash')
    'google'
    >>> infer_provider('gpt-4o-mini')
    'openai'
    >>> infer_provider('llama3.2:3b')
    'ollama'
    """
    fallback = next((p.id for p in PROVIDERS.values() if p.is_fallback), "ollama")

    name = (model_name or "").strip().lower()
    if not name:
        return fallback

    for spec in PROVIDERS.values():
        if any(name.startswith(prefix) for prefix in spec.model_prefixes):
            return spec.id

    return fallback


def provider_env_vars(provider_id: str) -> Tuple[str, ...]:
    """Environment variables consulted for `provider_id`, canonical name first."""
    spec = get_provider(provider_id)
    return spec.env_vars if spec else ()


def resolve_api_key(provider_id: str) -> Optional[str]:
    """
    Read `provider_id`'s credential from the environment.

    Parameters
    ----------
    provider_id : str

    Returns
    -------
    Optional[str]
        The first non-empty value among the provider's environment variables.

    Notes
    -----
    Environment only -- the encrypted store is the key manager's job, and this
    module must stay importable without it to avoid an import cycle.
    """
    for var in provider_env_vars(provider_id):
        value = os.environ.get(var)
        if value:
            return value
    return None


def is_sdk_available(provider_id: str) -> bool:
    """
    Whether `provider_id`'s SDK can be imported.

    Uses :func:`importlib.util.find_spec` rather than a real import: the check
    runs while building menus, and importing an LLM SDK costs about a second.
    """
    spec = get_provider(provider_id)
    if not spec or not spec.sdk_module:
        return True
    try:
        return importlib.util.find_spec(spec.sdk_module) is not None
    except (ImportError, ValueError):
        return False


def model_descriptions() -> Dict[str, str]:
    """Every catalogued model's one-line description, keyed by model name."""
    return {
        model.name: model.description
        for spec in PROVIDERS.values()
        for model in spec.models
        if model.description
    }
