#!/usr/bin/env python3
"""
Quick Start Example: LLM Annotation with Ollama

This example demonstrates how to use LLM Tool for zero-shot annotation
using a local Ollama model.

Prerequisites:
- Ollama installed (https://ollama.ai)
- A model pulled (e.g., ollama pull llama3.2)
- LLM Tool installed (pip install -e .)

Author: Antoine Lemor
"""

from llm_tool.annotators.llm_annotator import LLMAnnotator
from llm_tool.annotators.local_models import OllamaClient
import json


def main():
    print("=" * 70)
    print("LLM TOOL - Quick Start: Sentiment Analysis with Ollama")
    print("=" * 70)
    print()

    # Sample data
    texts = [
        "This product is amazing! Best purchase I've made this year.",
        "Terrible experience, would not recommend to anyone.",
        "It's okay, nothing special but gets the job done.",
        "Absolutely love it! Five stars all the way!",
        "Waste of money. Very disappointed with the quality.",
    ]

    # Define annotation schema
    schema = {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the text"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score (0-1)"
        }
    }

    # Initialize Ollama client
    print("Initializing Ollama client...")
    client = OllamaClient(model="llama3.2")

    # Check if model is available
    try:
        available_models = client.list_models()
        if "llama3.2" not in available_models:
            print("❌ Error: llama3.2 model not found.")
            print("   Please run: ollama pull llama3.2")
            return
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("   Please ensure Ollama is running: ollama serve")
        return

    print("✓ Ollama connected successfully")
    print()

    # Create annotator
    annotator = LLMAnnotator(
        llm_client=client,
        schema=schema,
        temperature=0.3  # Lower temperature for more consistent results
    )

    # Annotate texts
    print("Annotating texts...")
    print("-" * 70)

    results = []
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}...")

        try:
            annotation = annotator.annotate(text)
            results.append({
                "text": text,
                "annotation": annotation
            })

            # Display result
            sentiment = annotation.get("sentiment", "unknown")
            confidence = annotation.get("confidence", 0)
            print(f"    → Sentiment: {sentiment.upper()} (confidence: {confidence:.2f})")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "text": text,
                "annotation": None,
                "error": str(e)
            })

    print()
    print("-" * 70)

    # Summary
    successful = sum(1 for r in results if r.get("annotation") is not None)
    print(f"\nSummary: {successful}/{len(texts)} texts annotated successfully")

    # Save results
    output_file = "annotation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Results saved to: {output_file}")
    print()

    # Display sentiment distribution
    sentiments = [
        r["annotation"]["sentiment"]
        for r in results
        if r.get("annotation")
    ]

    if sentiments:
        print("\nSentiment Distribution:")
        for sentiment in ["positive", "negative", "neutral"]:
            count = sentiments.count(sentiment)
            percentage = (count / len(sentiments)) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {sentiment.capitalize():8s}: {bar} {count} ({percentage:.1f}%)")

    print()
    print("=" * 70)
    print("Annotation complete! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    main()
