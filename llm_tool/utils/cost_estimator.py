#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
cost_estimator.py

MAIN OBJECTIVE:
---------------
Estimate API usage costs for annotation workflows. This module centralises
the logic so both CLI summaries and pipeline code can reuse the exact same
calculations and token analysis behaviour.

Dependencies:
-------------
- dataclasses
- typing
- pandas
- llm_tool.utils.token_analysis

MAIN FEATURES:
--------------
1) Define pricing information for models (ModelPricing)
2) Structured cost estimation output (CostEstimate)
3) OpenAI model pricing database
4) Resolve model pricing by provider/model combination
5) Estimate total annotation cost with token analysis
6) Format cost estimates for human-readable output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import math

import pandas as pd

from .token_analysis import analyse_text_tokens, TokenAnalysisResult


@dataclass
class ModelPricing:
    """Pricing information for a model (per 1K tokens by default)."""

    model: str
    provider: str
    prompt_cost_per_1k: float
    completion_cost_per_1k: float
    currency: str = "USD"
    last_updated: str = "2024-07-01"
    cached_prompt_cost_per_1k: Optional[float] = None
    batch_prompt_cost_per_1k: Optional[float] = None
    batch_cached_prompt_cost_per_1k: Optional[float] = None
    batch_completion_cost_per_1k: Optional[float] = None


@dataclass
class CostEstimate:
    """Structured cost estimation output."""

    model: str
    provider: str
    texts_count: int
    prompt_count: int
    request_count: int
    input_tokens: int
    output_tokens_estimated: int
    prompt_tokens_total: int
    header_tokens_total: int
    text_tokens_total: int
    input_cost: float
    output_cost: float
    total_cost: float
    per_request_input_tokens: float
    per_request_output_tokens: float
    per_request_cost: float
    pricing_currency: str
    pricing_last_updated: str
    assumptions: Dict[str, Any] = field(default_factory=dict)


DEFAULT_PROMPT_HEADER = "\n\nText to analyze:\n"
DEFAULT_OUTPUT_TOKENS_FALLBACK = 0.5  # Fallback ratio of max_tokens when we lack schema info
TOKENS_PER_JSON_FIELD = 12  # Heuristic for small JSON responses per key

OPENAI_BATCH_MAX_FILE_BYTES = 512 * 1024 * 1024  # 512 MB documented limit
OPENAI_BATCH_MAX_INPUT_TOKENS = 50_000_000  # documented soft limit for batch submissions
OPENAI_BATCH_APPROX_BYTES_PER_TOKEN = 4.5  # empirical factor based on JSONL payloads
OPENAI_BATCH_APPROX_JSON_OVERHEAD_BYTES = 200  # metadata per request (custom_id, method, etc.)
OPENAI_BATCH_APPROX_TOKEN_OVERHEAD = 32  # safety margin for system tokens


def _serialize_token_stats(result: TokenAnalysisResult) -> Dict[str, Any]:
    """Return dict-friendly token stats without heavy arrays."""
    stats = result.to_dict()
    stats.pop("token_lengths", None)
    stats.pop("char_lengths", None)
    return stats


def _build_component_summary(
    *,
    label: str,
    component_type: str,
    analysis: TokenAnalysisResult,
    tokens_per_request: float,
    tokens_total: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = {
        "label": label,
        "type": component_type,
        "stats": _serialize_token_stats(analysis),
        "tokens_per_request": tokens_per_request,
        "tokens_total": tokens_total,
        "samples": analysis.texts_analyzed,
    }
    if metadata:
        summary["metadata"] = metadata
    return summary

# Pricing reference (OpenAI public pricing, July 2024)
OPENAI_MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        provider="openai",
        prompt_cost_per_1k=0.005,
        completion_cost_per_1k=0.015,
        last_updated="2024-07-01",
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        provider="openai",
        prompt_cost_per_1k=0.00015,
        completion_cost_per_1k=0.0006,
        last_updated="2024-07-01",
    ),
    "gpt-4.1-2025-04-14": ModelPricing(
        model="gpt-4.1-2025-04-14",
        provider="openai",
        prompt_cost_per_1k=0.001,   # $1.00 per 1M input tokens
        completion_cost_per_1k=0.004,  # $4.00 per 1M output tokens
        last_updated="2025-04-14",
        batch_prompt_cost_per_1k=0.001,
        batch_completion_cost_per_1k=0.004,
    ),
    "gpt-5-2025-08-07": ModelPricing(
        model="gpt-5-2025-08-07",
        provider="openai",
        prompt_cost_per_1k=0.00125,  # $1.250 per 1M input tokens
        completion_cost_per_1k=0.01,  # $10.000 per 1M output tokens
        cached_prompt_cost_per_1k=0.000125,  # $0.125 per 1M cached input tokens
        batch_prompt_cost_per_1k=0.000625,  # $0.625 per 1M input tokens (batch)
        batch_cached_prompt_cost_per_1k=0.0000625,  # $0.0625 per 1M cached input tokens (batch)
        batch_completion_cost_per_1k=0.005,  # $5.000 per 1M output tokens (batch)
        last_updated="2025-08-07",
    ),
    "gpt-5-mini-2025-08-07": ModelPricing(
        model="gpt-5-mini-2025-08-07",
        provider="openai",
        prompt_cost_per_1k=0.00025,  # $0.250 per 1M input tokens
        completion_cost_per_1k=0.002,  # $2.000 per 1M output tokens
        cached_prompt_cost_per_1k=0.000025,  # $0.025 per 1M cached input tokens
        batch_prompt_cost_per_1k=0.000125,  # $0.125 per 1M input tokens (batch)
        batch_cached_prompt_cost_per_1k=0.0000125,  # $0.0125 per 1M cached input tokens (batch)
        batch_completion_cost_per_1k=0.001,  # $1.000 per 1M output tokens (batch)
        last_updated="2025-08-07",
    ),
    "gpt-5-nano-2025-08-07": ModelPricing(
        model="gpt-5-nano-2025-08-07",
        provider="openai",
        prompt_cost_per_1k=0.00005,  # $0.050 per 1M input tokens
        completion_cost_per_1k=0.0004,  # $0.400 per 1M output tokens
        cached_prompt_cost_per_1k=0.000005,  # $0.005 per 1M cached input tokens
        batch_prompt_cost_per_1k=0.000025,  # $0.025 per 1M input tokens (batch)
        batch_cached_prompt_cost_per_1k=0.0000025,  # $0.0025 per 1M cached input tokens (batch)
        batch_completion_cost_per_1k=0.0002,  # $0.200 per 1M output tokens (batch)
        last_updated="2025-08-07",
    ),
    "o1-mini": ModelPricing(
        model="o1-mini",
        provider="openai",
        prompt_cost_per_1k=0.0011,
        completion_cost_per_1k=0.0044,
        last_updated="2024-07-01",
    ),
}


def _build_joined_texts(df: pd.DataFrame, text_columns: List[str]) -> List[str]:
    """Reproduce annotator behaviour: join text columns with double new lines."""
    joined: List[str] = []
    for _, row in df[text_columns].iterrows():
        parts = [str(value) for value in row if pd.notna(value)]
        joined.append("\n\n".join(parts))
    return joined


def get_openai_pricing(model_name: str) -> Optional[ModelPricing]:
    """Return pricing entry for the given OpenAI model (case-insensitive)."""
    key = model_name.lower()
    for registered_name, pricing in OPENAI_MODEL_PRICING.items():
        if key == registered_name or key.startswith(registered_name):
            # Normalize model field to the exact name requested for reporting.
            return ModelPricing(
                model=model_name,
                provider=pricing.provider,
                prompt_cost_per_1k=pricing.prompt_cost_per_1k,
                completion_cost_per_1k=pricing.completion_cost_per_1k,
                currency=pricing.currency,
                last_updated=pricing.last_updated,
                cached_prompt_cost_per_1k=pricing.cached_prompt_cost_per_1k,
                batch_prompt_cost_per_1k=pricing.batch_prompt_cost_per_1k,
                batch_cached_prompt_cost_per_1k=pricing.batch_cached_prompt_cost_per_1k,
                batch_completion_cost_per_1k=pricing.batch_completion_cost_per_1k,
            )
    return None


def resolve_model_pricing(provider: str, model_name: str) -> Optional[ModelPricing]:
    """
    Return pricing information for the given provider/model combination.

    Currently only OpenAI models are bundled with verified pricing (sourced from
    https://openai.com/pricing, snapshot 2024-07-01). Additional providers can
    be added here as they are formally supported throughout the package.
    """
    provider_key = (provider or "").strip().lower()
    if provider_key == "openai":
        return get_openai_pricing(model_name)
    return None


def estimate_annotation_cost(
    *,
    df_subset: pd.DataFrame,
    text_columns: List[str],
    prompts: List[Dict[str, Any]],
    pricing: ModelPricing,
    max_output_tokens: int,
    prompt_header: str = DEFAULT_PROMPT_HEADER,
    output_tokens_ratio: float = DEFAULT_OUTPUT_TOKENS_FALLBACK,
) -> CostEstimate:
    """
    Estimate total cost for annotating ``df_subset`` using the provided prompts.

    Parameters
    ----------
    df_subset:
        DataFrame containing only the rows that will be annotated.
    text_columns:
        Column names used to build the text sent to the model.
    prompts:
        Prompt configuration list. Each entry should expose ``prompt`` (or ``content``)
        and optional ``expected_keys`` metadata.
    pricing:
        Model pricing information (per 1K tokens).
    max_output_tokens:
        Generation cap configured for the model.
    prompt_header:
        Static header appended before user text (mirrors annotator implementation).
    output_tokens_ratio:
        Fallback ratio applied to ``max_output_tokens`` when we cannot infer a better
        output length estimate from prompt metadata.
    """
    texts = _build_joined_texts(df_subset, text_columns)
    text_stats = analyse_text_tokens(texts)
    component_token_stats: List[Dict[str, Any]] = []
    prompt_summaries: List[Dict[str, Any]] = []

    # Prompt tokens
    prompt_tokens_per_prompt: List[int] = []
    output_tokens_per_prompt: List[int] = []
    for prompt_cfg in prompts:
        prompt_text = prompt_cfg.get("prompt") or prompt_cfg.get("template") or prompt_cfg.get("content") or ""
        prompt_stats = analyse_text_tokens([prompt_text])
        prompt_tokens_per_prompt.append(prompt_stats.token_total)

        expected_keys: Iterable[str] = prompt_cfg.get("expected_keys") or []
        expected_count = len(list(expected_keys))
        if expected_count:
            estimated_tokens = max(32, expected_count * TOKENS_PER_JSON_FIELD)
        else:
            estimated_tokens = int(max_output_tokens * output_tokens_ratio)

        output_tokens_per_prompt.append(min(max_output_tokens, estimated_tokens))

        prompt_index = len(prompt_summaries) + 1
        prompt_label = (
            prompt_cfg.get("name")
            or prompt_cfg.get("label")
            or prompt_cfg.get("id")
            or f"Prompt #{prompt_index}"
        )
        prompt_preview = (prompt_text.strip().replace("\n", " "))[:180]
        prompt_summaries.append(
            {
                "label": prompt_label,
                "analysis": prompt_stats,
                "tokens_single_prompt": prompt_stats.token_total,
                "metadata": {
                    "expected_keys": list(expected_keys),
                    "preview": prompt_preview,
                },
            }
        )

    header_stats = analyse_text_tokens([prompt_header])
    header_tokens = header_stats.token_total

    texts_count = len(texts)
    prompt_count = len(prompts) if prompts else 1
    request_count = texts_count * prompt_count

    text_tokens_total = text_stats.token_total
    text_tokens_total_effective = text_tokens_total * prompt_count
    prompt_tokens_total = sum(prompt_tokens_per_prompt) * texts_count
    header_tokens_total = header_tokens * request_count
    input_tokens = text_tokens_total_effective + prompt_tokens_total + header_tokens_total

    output_tokens_estimated = sum(output_tokens_per_prompt) * texts_count

    input_cost = (input_tokens / 1000.0) * pricing.prompt_cost_per_1k
    output_cost = (output_tokens_estimated / 1000.0) * pricing.completion_cost_per_1k
    total_cost = input_cost + output_cost

    per_request_input_tokens = input_tokens / request_count if request_count else 0.0
    per_request_output_tokens = output_tokens_estimated / request_count if request_count else 0.0
    per_request_cost = total_cost / request_count if request_count else 0.0

    batching_info: Dict[str, Any] = {
        "request_count": request_count,
        "approx_input_tokens": int(input_tokens),
        "approx_tokens_per_request": float(per_request_input_tokens),
    }

    provider_lower = pricing.provider.lower()
    if provider_lower == "openai" and request_count:
        approx_input_bytes = int(
            input_tokens * OPENAI_BATCH_APPROX_BYTES_PER_TOKEN
            + request_count * OPENAI_BATCH_APPROX_JSON_OVERHEAD_BYTES
        )
        approx_bytes_per_request = approx_input_bytes / request_count if request_count else 0.0
        batches_by_bytes = (
            math.ceil(approx_input_bytes / OPENAI_BATCH_MAX_FILE_BYTES)
            if approx_input_bytes > 0
            else 1
        )
        batches_by_tokens = (
            math.ceil(input_tokens / OPENAI_BATCH_MAX_INPUT_TOKENS)
            if input_tokens > 0
            else 1
        )
        recommended_batches = max(1, batches_by_bytes, batches_by_tokens)
        batching_info.update(
            {
                "approx_input_bytes": approx_input_bytes,
                "approx_bytes_per_request": approx_bytes_per_request,
                "max_file_bytes": OPENAI_BATCH_MAX_FILE_BYTES,
                "max_input_tokens": OPENAI_BATCH_MAX_INPUT_TOKENS,
                "recommended_batches": recommended_batches,
                "requests_per_batch": math.ceil(request_count / recommended_batches),
                "limit_bytes_triggered": approx_input_bytes > OPENAI_BATCH_MAX_FILE_BYTES,
                "limit_tokens_triggered": input_tokens > OPENAI_BATCH_MAX_INPUT_TOKENS,
            }
        )
    else:
        batching_info.update(
            {
                "approx_input_bytes": None,
                "approx_bytes_per_request": None,
                "max_file_bytes": None,
                "max_input_tokens": None,
                "recommended_batches": 1,
                "requests_per_batch": request_count,
                "limit_bytes_triggered": False,
                "limit_tokens_triggered": False,
            }
        )

    token_stats_summary = {
        "texts": text_stats.texts_analyzed,
        "min": int(text_stats.token_min),
        "max": int(text_stats.token_max),
        "mean": float(text_stats.token_mean),
        "median": float(text_stats.token_median),
        "p95": float(text_stats.token_p95),
        "requires_long_context": bool(text_stats.requires_long_document_model),
    }

    text_tokens_per_request = text_tokens_total_effective / request_count if request_count else 0.0
    prompt_tokens_per_request_total = sum(prompt_tokens_per_prompt)
    header_tokens_per_request = header_tokens

    component_token_stats.append(
        _build_component_summary(
            label="Dataset text",
            component_type="text",
            analysis=text_stats,
            tokens_per_request=text_tokens_per_request,
            tokens_total=int(text_tokens_total_effective),
            metadata={
                "text_columns": list(text_columns),
                "texts_count": texts_count,
            },
        )
    )

    for summary in prompt_summaries:
        tokens_single_prompt = summary["tokens_single_prompt"]
        component_token_stats.append(
            _build_component_summary(
                label=summary["label"],
                component_type="prompt",
                analysis=summary["analysis"],
                tokens_per_request=float(tokens_single_prompt),
                tokens_total=int(tokens_single_prompt * texts_count),
                metadata={
                    "expected_keys": summary["metadata"]["expected_keys"],
                    "preview": summary["metadata"]["preview"],
                    "applies_to_texts": texts_count,
                },
            )
        )

    component_token_stats.append(
        _build_component_summary(
            label="Prompt header",
            component_type="header",
            analysis=header_stats,
            tokens_per_request=float(header_tokens),
            tokens_total=int(header_tokens_total),
            metadata={"applies_to_requests": request_count},
        )
    )

    input_breakdown = {
        "text_tokens_total": int(text_tokens_total),
        "text_tokens_total_effective": int(text_tokens_total_effective),
        "prompt_tokens_total": int(prompt_tokens_total),
        "prompt_tokens_total_effective": int(prompt_tokens_total),
        "header_tokens_total": int(header_tokens_total),
        "header_tokens_total_effective": int(header_tokens_total),
        "prompt_tokens_per_prompt": [int(value) for value in prompt_tokens_per_prompt],
        "prompt_tokens_per_request_total": float(prompt_tokens_per_request_total),
        "text_tokens_per_request": float(text_tokens_per_request),
        "header_tokens_per_request": int(header_tokens_per_request),
    }

    pricing_extras: Dict[str, Optional[float]] = {
        "cached_prompt_cost_per_1k": pricing.cached_prompt_cost_per_1k,
        "batch_prompt_cost_per_1k": pricing.batch_prompt_cost_per_1k,
        "batch_cached_prompt_cost_per_1k": pricing.batch_cached_prompt_cost_per_1k,
        "batch_completion_cost_per_1k": pricing.batch_completion_cost_per_1k,
    }

    base_pricing = {
        "prompt_cost_per_1k": pricing.prompt_cost_per_1k,
        "completion_cost_per_1k": pricing.completion_cost_per_1k,
        "currency": pricing.currency,
        "last_updated": pricing.last_updated,
    }

    prompt_component_entries = [entry for entry in component_token_stats if entry["type"] == "prompt"]
    prompt_sub_components = [
        {
            "label": entry["label"],
            "tokens_total": entry["tokens_total"],
            "tokens_per_request": entry["tokens_per_request"],
            "cost": (entry["tokens_total"] / 1000.0) * pricing.prompt_cost_per_1k,
        }
        for entry in prompt_component_entries
    ]

    component_token_contributions = [
        {
            "label": "Input text (× prompts)",
            "category": "input-text",
            "tokens_total": int(text_tokens_total_effective),
            "tokens_per_request": float(text_tokens_per_request),
            "cost": (text_tokens_total_effective / 1000.0) * pricing.prompt_cost_per_1k,
        },
    ]

    if prompt_component_entries:
        component_token_contributions.append(
            {
                "label": "Prompt templates",
                "category": "input-prompts",
                "tokens_total": int(sum(entry["tokens_total"] for entry in prompt_component_entries)),
                "tokens_per_request": float(prompt_tokens_per_request_total),
                "cost": (sum(entry["tokens_total"] for entry in prompt_component_entries) / 1000.0)
                * pricing.prompt_cost_per_1k,
                "sub_components": prompt_sub_components,
            }
        )

    component_token_contributions.append(
        {
            "label": "Prompt header",
            "category": "input-header",
            "tokens_total": int(header_tokens_total),
            "tokens_per_request": float(header_tokens_per_request),
            "cost": (header_tokens_total / 1000.0) * pricing.prompt_cost_per_1k,
        }
    )

    component_token_contributions.append(
        {
            "label": "Model output (est.)",
            "category": "output",
            "tokens_total": int(output_tokens_estimated),
            "tokens_per_request": float(per_request_output_tokens),
            "cost": output_cost,
            "pricing_type": "completion",
        }
    )

    assumptions = {
        "output_tokens_per_prompt": output_tokens_per_prompt,
        "output_tokens_ratio_fallback": output_tokens_ratio,
        "tokens_per_json_field": TOKENS_PER_JSON_FIELD,
        "token_stats": token_stats_summary,
        "input_breakdown": input_breakdown,
        "pricing_extras": pricing_extras,
        "base_pricing": base_pricing,
        "batching": batching_info,
        "component_token_stats": component_token_stats,
        "component_token_contributions": component_token_contributions,
    }

    return CostEstimate(
        model=pricing.model,
        provider=pricing.provider,
        texts_count=texts_count,
        prompt_count=prompt_count,
        request_count=request_count,
        input_tokens=int(input_tokens),
        output_tokens_estimated=int(output_tokens_estimated),
        prompt_tokens_total=int(prompt_tokens_total),
        header_tokens_total=int(header_tokens_total),
        text_tokens_total=int(text_tokens_total),
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        per_request_input_tokens=per_request_input_tokens,
        per_request_output_tokens=per_request_output_tokens,
        per_request_cost=per_request_cost,
        pricing_currency=pricing.currency,
        pricing_last_updated=pricing.last_updated,
        assumptions=assumptions,
    )


def format_cost_estimate_lines(cost_estimate: CostEstimate, *, rich_markup: bool = True) -> List[str]:
    """
    Produce human-readable lines summarising the API cost estimate.

    Parameters
    ----------
    cost_estimate:
        Structured cost estimate returned by :func:`estimate_annotation_cost`.
    rich_markup:
        When True, include Rich markup tags suitable for colourful console panels.

    Returns
    -------
    list of str
        Ordered lines describing request volume, token usage, and monetary impact.
    """
    title_prefix = "Estimated API Cost"
    provider_name = cost_estimate.provider.capitalize()
    last_updated = cost_estimate.pricing_last_updated

    token_stats = cost_estimate.assumptions.get("token_stats") or {}
    input_breakdown = cost_estimate.assumptions.get("input_breakdown") or {}
    pricing_extras = cost_estimate.assumptions.get("pricing_extras") or {}
    has_cached_pricing = bool(pricing_extras.get("cached_prompt_cost_per_1k"))
    has_batch_pricing = any(
        pricing_extras.get(key)
        for key in (
            "batch_prompt_cost_per_1k",
            "batch_cached_prompt_cost_per_1k",
            "batch_completion_cost_per_1k",
        )
    )

    base_pricing = cost_estimate.assumptions.get("base_pricing") or {}
    prompt_cost_per_1k = base_pricing.get("prompt_cost_per_1k")
    completion_cost_per_1k = base_pricing.get("completion_cost_per_1k")
    effective_currency = base_pricing.get("currency", cost_estimate.pricing_currency)

    per_million_input = (
        (prompt_cost_per_1k or 0.0) * 1000 if prompt_cost_per_1k is not None else None
    )
    per_million_completion = (
        (completion_cost_per_1k or 0.0) * 1000 if completion_cost_per_1k is not None else None
    )
    per_million_cached = (
        pricing_extras.get("cached_prompt_cost_per_1k") * 1000
        if pricing_extras.get("cached_prompt_cost_per_1k") is not None
        else None
    )
    per_million_batch_in = (
        pricing_extras.get("batch_prompt_cost_per_1k") * 1000
        if pricing_extras.get("batch_prompt_cost_per_1k") is not None
        else None
    )
    per_million_batch_cached = (
        pricing_extras.get("batch_cached_prompt_cost_per_1k") * 1000
        if pricing_extras.get("batch_cached_prompt_cost_per_1k") is not None
        else None
    )
    per_million_batch_out = (
        pricing_extras.get("batch_completion_cost_per_1k") * 1000
        if pricing_extras.get("batch_completion_cost_per_1k") is not None
        else None
    )

    if rich_markup:
        header = f"[bold white]{title_prefix} — {provider_name} (pricing {last_updated}):[/bold white]"
        total_line = f"- Total cost ({cost_estimate.pricing_currency}): [bold cyan]${cost_estimate.total_cost:.4f}[/bold cyan]"
        footer = (
            "[dim]"
            "Assumes "
            f"{cost_estimate.assumptions.get('output_tokens_per_prompt')} "
            "output tokens/prompt; pricing fixed at package update."
            "[/dim]"
        )
    else:
        header = f"{title_prefix} — {provider_name} (pricing {last_updated}):"
        total_line = f"- Total cost ({cost_estimate.pricing_currency}): ${cost_estimate.total_cost:.4f}"
        footer = (
            "Assumes "
            f"{cost_estimate.assumptions.get('output_tokens_per_prompt')} "
            "output tokens/prompt; pricing fixed at package update."
        )

    breakdown_line = (
        f"- Input breakdown: text {input_breakdown.get('text_tokens_total', cost_estimate.text_tokens_total):,} "
        f"+ prompt {input_breakdown.get('prompt_tokens_total', cost_estimate.prompt_tokens_total):,} "
        f"+ header {input_breakdown.get('header_tokens_total', cost_estimate.header_tokens_total):,}"
    )

    per_request_line = (
        f"- Tokens/request (in/out): "
        f"{cost_estimate.per_request_input_tokens:.1f}/{cost_estimate.per_request_output_tokens:.1f}"
    )

    token_stats_line = None
    if token_stats:
        token_stats_line = (
            "- Token stats — "
            f"min: {int(token_stats.get('min', 0)):,}, "
            f"avg: {token_stats.get('mean', 0.0):.1f}, "
            f"p95: {int(token_stats.get('p95', 0)):,} "
            f"(texts: {int(token_stats.get('texts', cost_estimate.texts_count))})"
        )
        if token_stats.get("requires_long_context"):
            token_stats_line += " [long context recommended]" if rich_markup else " [long context recommended]"

    batching_lines: List[str] = []
    batching = cost_estimate.assumptions.get("batching") or {}
    if batching:
        approx_bytes = batching.get("approx_input_bytes")
        request_count = batching.get("request_count", cost_estimate.request_count)
        if approx_bytes:
            approx_mb = approx_bytes / (1024 ** 2)
            batching_lines.append(
                f"- Estimated batch payload: {approx_mb:.1f} MB for {int(request_count):,} request(s)"
            )

        recommended_batches = batching.get("recommended_batches", 1) or 1
        if recommended_batches > 1:
            requests_per_batch = batching.get("requests_per_batch") or math.ceil(
                request_count / recommended_batches
            )
            batching_lines.append(
                f"- Payload exceeds OpenAI limits → split into {recommended_batches} batch job(s)"
                f" (~{int(requests_per_batch):,} requests/job)."
            )
        elif approx_bytes and batching.get("max_file_bytes"):
            max_mb = batching["max_file_bytes"] / (1024 ** 2)
            batching_lines.append(
                f"- Batch payload within OpenAI limit ({max_mb:.0f} MB max/file)."
            )

    lines = [header]

    if per_million_input is not None or per_million_completion is not None:
        input_label = f"${per_million_input:.3f}/1M" if per_million_input is not None else "N/A"
        output_label = f"${per_million_completion:.3f}/1M" if per_million_completion is not None else "N/A"
        lines.append(
            f"- Pricing reference ({effective_currency}): input {input_label}, output {output_label}"
        )

    lines.extend([
        f"- Requests: {cost_estimate.request_count:,} ({cost_estimate.prompt_count} prompt(s) × {cost_estimate.texts_count:,} text(s))",
        f"- Input tokens: {cost_estimate.input_tokens:,} (~${cost_estimate.input_cost:.4f})",
        breakdown_line,
        per_request_line,
        f"- Output tokens (est.): {cost_estimate.output_tokens_estimated:,} (~${cost_estimate.output_cost:.4f})",
        total_line,
    ])

    if batching_lines:
        lines.extend(batching_lines)

    if token_stats_line:
        lines.append(token_stats_line)

    if per_million_cached is not None:
        lines.append(f"- Cached input pricing ({effective_currency}): ${per_million_cached:.3f}/1M tokens")
    if any(value is not None for value in (per_million_batch_in, per_million_batch_cached, per_million_batch_out)):
        batch_details = [
            f"in ${per_million_batch_in:.3f}/1M" if per_million_batch_in is not None else "",
            f"cache ${per_million_batch_cached:.4f}/1M" if per_million_batch_cached is not None else "",
            f"out ${per_million_batch_out:.3f}/1M" if per_million_batch_out is not None else "",
        ]
        batch_details = [part for part in batch_details if part]
        if batch_details:
            lines.append(f"- Batch pricing ({effective_currency}): " + ", ".join(batch_details))

    component_contributions = cost_estimate.assumptions.get("component_token_contributions") or []
    if component_contributions:
        lines.append("- Component cost breakdown:")
        for component in component_contributions:
            label = component.get("label", "Component")
            tokens_total = component.get("tokens_total", 0)
            tokens_per_request = component.get("tokens_per_request", 0.0)
            component_cost = component.get("cost", 0.0)
            contribution_line = (
                f"  • {label}: {tokens_total:,} tok "
                f"({tokens_per_request:.1f}/req) → ${component_cost:.4f}"
            )
            lines.append(contribution_line)

    lines.append(
        footer,
    )
    return lines
