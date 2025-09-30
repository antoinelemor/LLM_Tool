#!/usr/bin/env python3
"""
Test the annotation pipeline with real data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm_tool.pipelines.pipeline_controller import PipelineController
import json

def test_annotation():
    """Test annotation with small dataset"""
    print("=== Testing Annotation Pipeline ===\n")

    # Configuration for annotation
    config = {
        'mode': 'file',
        'data_source': 'csv',
        'file_path': 'data/test_data.csv',
        'text_column': 'text',

        # Annotation settings
        'run_annotation': True,
        'annotation_mode': 'local',
        'annotation_provider': 'ollama',
        'annotation_model': 'gemma3:27b',  # Using available model
        'prompts': [{
            'template': 'Classify the sentiment of this text as positive or negative. Answer with only one word: positive or negative.\n\nText: {text}\n\nSentiment:',
            'label': 'sentiment',
            'type': 'classification'
        }],
        'batch_size': 5,
        'max_workers': 2,
        'output_path': 'data/test_annotations.json',

        # Skip other phases for now
        'run_validation': False,
        'run_training': False,
        'run_deployment': False
    }

    try:
        # Initialize controller
        controller = PipelineController()

        # Run annotation
        print("Starting annotation...")
        print(f"Using model: {config['annotation_model']}")
        print(f"Processing file: {config['file_path']}\n")

        results = controller.run_annotation(config)

        if results:
            print("\n✓ Annotation completed successfully!")
            print(f"  Total annotated: {results.get('total_annotated', 0)}")
            print(f"  Output file: {results.get('output_file', 'unknown')}")

            # Show sample results
            if results.get('output_file'):
                with open(results['output_file'], 'r') as f:
                    annotations = json.load(f)

                print(f"\nSample annotations (first 3):")
                for item in annotations[:3]:
                    text = item.get('text', '')[:50] + '...' if len(item.get('text', '')) > 50 else item.get('text', '')
                    label = item.get('annotation', 'N/A')
                    print(f"  • \"{text}\" → {label}")
        else:
            print("✗ No results returned from annotation")

    except Exception as e:
        print(f"✗ Annotation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_annotation()