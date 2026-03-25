#!/usr/bin/env python3
"""
Test de l'alignement du prompt LLM avec les annotations humaines du benchmark.

Ce script:
1. Charge N phrases du benchmark humain
2. Les annote avec un modèle Ollama local (gpt-oss:120b)
3. Compare les résultats avec le consensus humain
4. Génère un rapport d'alignement

Usage:
    python paper/scripts/evaluation/prompt_alignment_test.py [--n 20] [--model gpt-oss:120b] [--prompt prompts/prompt_EN_long.txt]
"""

import argparse
import json
import pandas as pd
import ast
import sys
import os
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm_tool.annotators.local_models import OllamaClient


def parse_list(val):
    """Safely parse a list from string."""
    if pd.isna(val) or val == '[]':
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []


def normalize_theme(theme: str) -> str:
    """Normalize theme name."""
    if not theme:
        return ''
    prefixes = ['theme_', 'consensus_themes_extra_large_', 'consensus_themes_']
    theme_lower = theme.lower().strip()
    for p in prefixes:
        if theme_lower.startswith(p):
            theme_lower = theme_lower[len(p):]
    return theme_lower


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    if not response:
        return {}

    # Try direct parse
    try:
        return json.loads(response)
    except:
        pass

    # Try to find JSON block
    import re
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass

    return {}


def annotate_text(client: OllamaClient, prompt_template: str, text: str) -> dict:
    """Annotate a single text with the LLM."""
    full_prompt = f"{prompt_template}\n\n**Sentence to annotate:**\n{text}"

    response = client.generate(
        prompt=full_prompt,
        temperature=0.1,  # Low temp for consistency
        max_tokens=500
    )

    if not response:
        return {}

    return extract_json_from_response(response)


def compare_annotations(llm_result: dict, consensus: dict) -> dict:
    """Compare LLM annotation with human consensus."""
    # Extract and normalize themes
    llm_themes = llm_result.get('themes_long', [])
    if llm_themes is None:
        llm_themes = []
    elif isinstance(llm_themes, str):
        llm_themes = [llm_themes] if llm_themes and llm_themes != 'null' else []
    llm_themes_norm = set(normalize_theme(t) for t in llm_themes if t and t != 'null')

    cons_themes = consensus.get('themes', [])
    cons_themes_norm = set(normalize_theme(t) for t in cons_themes if t)

    # Calculate overlap
    overlap = llm_themes_norm & cons_themes_norm
    only_llm = llm_themes_norm - cons_themes_norm
    only_cons = cons_themes_norm - llm_themes_norm

    # Exact match
    exact_match = llm_themes_norm == cons_themes_norm

    # Jaccard similarity
    union = llm_themes_norm | cons_themes_norm
    jaccard = len(overlap) / len(union) if union else 1.0

    # Sentiment
    llm_sentiment = llm_result.get('sentiment_long', '')
    if llm_sentiment:
        llm_sentiment = llm_sentiment.lower().replace('sentiment_', '')
    cons_sentiment = consensus.get('sentiment', '').replace('sentiment_', '')
    sentiment_match = llm_sentiment == cons_sentiment

    return {
        'exact_match': exact_match,
        'jaccard': jaccard,
        'overlap': list(overlap),
        'only_llm': list(only_llm),
        'only_cons': list(only_cons),
        'llm_themes': list(llm_themes_norm),
        'cons_themes': list(cons_themes_norm),
        'sentiment_match': sentiment_match,
        'llm_sentiment': llm_sentiment,
        'cons_sentiment': cons_sentiment
    }


def main():
    parser = argparse.ArgumentParser(description='Test prompt alignment with benchmark')
    parser.add_argument('-n', '--num_samples', type=int, default=20, help='Number of samples to test')
    parser.add_argument('-m', '--model', type=str, default='random', help='Ollama model name or "random" to pick randomly')
    parser.add_argument('-p', '--prompt', type=str, default='prompts/prompt_EN_long.txt', help='Prompt file')
    parser.add_argument('-b', '--benchmark', type=str, default='data/sets/benchmark_annotations_by_annotator.csv', help='Benchmark file')
    parser.add_argument('-o', '--output', type=str, default='paper/results/prompt_alignment', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None for random selection)')
    args = parser.parse_args()

    # Available models for random selection
    AVAILABLE_MODELS = ['nemotron:latest', 'gemma3:27b', 'gpt-oss:120b']

    # Select model
    import random
    if args.model == 'random':
        args.model = random.choice(AVAILABLE_MODELS)
        print(f"Randomly selected model: {args.model}")

    # Load prompt
    print(f"\n{'='*60}")
    print("PROMPT ALIGNMENT TEST")
    print(f"{'='*60}")

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {args.prompt}")
        return 1

    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    print(f"Prompt: {args.prompt} ({len(prompt_template)} chars)")

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        return 1

    df = pd.read_csv(benchmark_path)
    print(f"Benchmark: {args.benchmark} ({len(df)} samples)")

    # Filter samples with consensus themes (non-empty)
    df['consensus_parsed'] = df['consensus_themes'].apply(parse_list)
    df_with_cons = df[df['consensus_parsed'].apply(len) > 0].copy()
    print(f"Samples with consensus themes: {len(df_with_cons)}")

    # Sample N texts (random each time unless seed specified)
    n = min(args.num_samples, len(df_with_cons))
    random_state = args.seed if args.seed is not None else None
    sample_df = df_with_cons.sample(n=n, random_state=random_state)
    print(f"Testing on {n} samples (seed: {random_state if random_state else 'random'})")

    # Initialize Ollama client
    print(f"\nInitializing {args.model}...")
    try:
        client = OllamaClient(args.model, timeout=300)
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        return 1

    # Annotate samples
    print(f"\nAnnotating {n} samples...")
    results = []

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        text = row['text']
        consensus = {
            'themes': row['consensus_parsed'],
            'sentiment': row.get('consensus_sentiment', '')
        }

        print(f"\n[{idx+1}/{n}] {text[:60]}...")

        # Get LLM annotation
        llm_result = annotate_text(client, prompt_template, text)

        # Compare
        comparison = compare_annotations(llm_result, consensus)

        results.append({
            'id': row.get('id', idx),
            'text': text,
            'llm_raw': llm_result,
            **comparison
        })

        # Print result
        status = "EXACT" if comparison['exact_match'] else ("PARTIAL" if comparison['overlap'] else "NO MATCH")
        print(f"   {status} | LLM: {comparison['llm_themes']} | Consensus: {comparison['cons_themes']}")
        if comparison['only_llm']:
            print(f"   Sur-prédit: {comparison['only_llm']}")
        if comparison['only_cons']:
            print(f"   Manqué: {comparison['only_cons']}")

    # Summary
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")

    exact_matches = sum(1 for r in results if r['exact_match'])
    partial_matches = sum(1 for r in results if r['overlap'] and not r['exact_match'])
    no_matches = sum(1 for r in results if not r['overlap'])
    sentiment_matches = sum(1 for r in results if r['sentiment_match'])
    avg_jaccard = sum(r['jaccard'] for r in results) / len(results) if results else 0

    print(f"\nThèmes:")
    print(f"  Exact match: {exact_matches}/{n} ({100*exact_matches/n:.1f}%)")
    print(f"  Partial match: {partial_matches}/{n} ({100*partial_matches/n:.1f}%)")
    print(f"  No match: {no_matches}/{n} ({100*no_matches/n:.1f}%)")
    print(f"  Avg Jaccard: {avg_jaccard:.3f}")

    print(f"\nSentiment:")
    print(f"  Match: {sentiment_matches}/{n} ({100*sentiment_matches/n:.1f}%)")

    # Sur-prédiction analysis
    over_predicted = Counter()
    under_predicted = Counter()
    for r in results:
        over_predicted.update(r['only_llm'])
        under_predicted.update(r['only_cons'])

    print(f"\nThèmes sur-prédits (LLM mais pas consensus):")
    for theme, count in over_predicted.most_common(5):
        print(f"  {theme}: {count}")

    print(f"\nThèmes manqués (consensus mais pas LLM):")
    for theme, count in under_predicted.most_common(5):
        print(f"  {theme}: {count}")

    # Save results with timestamp
    os.makedirs(args.output, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(args.output) / f'alignment_test_{args.model.replace(":", "_")}_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'model': args.model,
                'prompt': args.prompt,
                'n_samples': n,
                'seed': random_state,
                'timestamp': timestamp
            },
            'summary': {
                'exact_match': exact_matches,
                'partial_match': partial_matches,
                'no_match': no_matches,
                'avg_jaccard': avg_jaccard,
                'sentiment_match': sentiment_matches
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés: {output_file}")

    return 0


if __name__ == '__main__':
    exit(main())
