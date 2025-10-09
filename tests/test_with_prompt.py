#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_with_prompt.py

MAIN OBJECTIVE:
---------------
Test the annotation pipeline with a complex prompt file for Canadian public policy
annotation including themes, political parties, and sentiment analysis.

Dependencies:
-------------
- sys
- os
- json
- llm_tool.pipelines.pipeline_controller

MAIN FEATURES:
--------------
1) Load and use external prompt file (prompt_EN_long.txt)
2) Test annotation with complex multi-field schema
3) Configure Ollama model with temperature and token settings
4) Verify annotation with expected keys validation
5) Display sample annotations with extracted fields

Author:
-------
Antoine Lemor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm_tool.pipelines.pipeline_controller import PipelineController
import json

def test_with_prompt_file():
    """Test annotation with prompt file"""
    print("=== Testing Annotation with Prompt File ===\n")

    # Read the prompt
    with open('data/prompt_EN_long.txt', 'r') as f:
        prompt_text = f.read()

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
            'template': prompt_text,
            'label': 'policy_annotation',
            'type': 'classification',
            'expected_keys': ['themes_long', 'political_parties_long', 'specific_themes_long', 'sentiment_long']
        }],
        'batch_size': 2,
        'max_workers': 1,
        'output_path': 'data/test_annotations_prompt.json',
        'temperature': 0.3,  # Lower temperature for more consistent annotations
        'max_tokens': 500,

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
        print(f"Processing file: {config['file_path']}")
        print(f"Prompt: Canadian public policy annotation\n")

        results = controller.run_annotation(config)

        if results:
            print("\n✓ Annotation completed successfully!")
            print(f"  Total annotated: {results.get('total_annotated', 0)}")
            print(f"  Output file: {results.get('output_file', results.get('output_path', 'unknown'))}")

            # Show sample results
            output_path = results.get('output_file') or results.get('output_path')
            if output_path and os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    annotations = json.load(f)

                print(f"\nSample annotations (first 3):")
                for item in annotations[:3]:
                    text = item.get('text', '')[:60] + '...' if len(item.get('text', '')) > 60 else item.get('text', '')
                    annotation = item.get('annotation', {})
                    if isinstance(annotation, str):
                        try:
                            annotation = json.loads(annotation)
                        except:
                            pass

                    print(f"\n  Text: \"{text}\"")
                    if isinstance(annotation, dict):
                        print(f"  Themes: {annotation.get('themes_long', 'N/A')}")
                        print(f"  Sentiment: {annotation.get('sentiment_long', 'N/A')}")
                    else:
                        print(f"  Annotation: {annotation}")
        else:
            print("✗ No results returned from annotation")

    except Exception as e:
        print(f"✗ Annotation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_prompt_file()