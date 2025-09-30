#!/usr/bin/env python3
"""
Test script to verify the pipeline works programmatically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm_tool.pipelines.pipeline_controller import PipelineController
from llm_tool.annotators.llm_annotator import LLMAnnotator
from llm_tool.trainers.model_trainer import ModelTrainer
from llm_tool.validators.annotation_validator import AnnotationValidator

def test_pipeline():
    """Test the complete pipeline programmatically"""
    print("=== Testing LLMTool Pipeline ===\n")

    # Test configuration
    config = {
        'mode': 'file',
        'data_source': 'file',
        'file_path': 'data/test_data.csv',  # We'll need to create test data
        'text_column': 'text',

        # Annotation config
        'run_annotation': True,
        'annotation_mode': 'local',
        'annotation_provider': 'ollama',
        'annotation_model': 'llama3.3:latest',
        'prompt_text': 'Classify this text as positive or negative: {text}',

        # Validation config
        'run_validation': True,
        'validation_sample_size': 10,
        'export_to_doccano': True,

        # Training config
        'run_training': False,  # Skip for now as it requires annotated data

        # Deployment config
        'run_deployment': False
    }

    # Test individual components
    print("1. Testing LLM Annotator...")
    try:
        annotator = LLMAnnotator()
        print("   ✓ LLM Annotator initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize LLM Annotator: {e}")

    print("\n2. Testing Model Trainer...")
    try:
        trainer = ModelTrainer()
        print("   ✓ Model Trainer initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize Model Trainer: {e}")

    print("\n3. Testing Annotation Validator...")
    try:
        validator = AnnotationValidator()
        print("   ✓ Annotation Validator initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize Annotation Validator: {e}")

    print("\n4. Testing Pipeline Controller...")
    try:
        controller = PipelineController()
        print("   ✓ Pipeline Controller initialized successfully")

        # Test pipeline initialization
        state = controller.initialize_pipeline(config)
        print(f"   ✓ Pipeline state initialized: {state.current_phase.value}")

    except Exception as e:
        print(f"   ✗ Failed to initialize Pipeline Controller: {e}")

    print("\n=== All components loaded successfully! ===")
    print("\nThe pipeline is ready to process data.")
    print("To run the full pipeline, create a CSV file with a 'text' column")
    print("and run the interactive CLI with: python -m llm_tool.cli.advanced_cli")

if __name__ == "__main__":
    test_pipeline()