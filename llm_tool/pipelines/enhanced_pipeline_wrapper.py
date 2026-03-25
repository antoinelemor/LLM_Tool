#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
enhanced_pipeline_wrapper.py

MAIN OBJECTIVE:
---------------
Wrapper for PipelineController that intercepts callbacks to provide
JSON samples and error information to the EnhancedProgressManager.

Author:
-------
Antoine Lemor
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AnnotationTracker:
    """Tracks annotation data for display"""
    count: int = 0
    last_json: Optional[Dict] = None
    last_error: Optional[str] = None
    json_buffer: list = None

    def __post_init__(self):
        if self.json_buffer is None:
            self.json_buffer = []


class EnhancedPipelineWrapper:
    """Wraps PipelineController to provide enhanced progress with JSON samples"""

    def __init__(self, pipeline_controller, enhanced_progress_manager):
        """Initialize the wrapper

        Args:
            pipeline_controller: The PipelineController instance
            enhanced_progress_manager: The EnhancedProgressManager instance
        """
        self.pipeline = pipeline_controller
        self.progress_manager = enhanced_progress_manager
        self.tracker = AnnotationTracker()
        self.logger = logging.getLogger(__name__)

        # Store progress manager reference in pipeline for passing to components
        self.pipeline.progress_manager = enhanced_progress_manager

        # Store original callback
        self.original_callback = pipeline_controller.progress_callback

        # Replace with our enhanced callback
        pipeline_controller.progress_callback = self._enhanced_callback

    def _enhanced_callback(self, phase: str, progress: float, message: str,
                          subtask: Optional[Dict[str, Any]] = None):
        """Enhanced callback that captures JSON samples and errors"""

        # Detect when training is about to start (progress 70) and pause Rich
        if phase == 'training' and progress >= 70 and not hasattr(self, '_training_paused'):
            self._training_paused = True
            # Show the training start message first
            self.progress_manager.update_progress(phase, progress, message, subtask)
            # Then pause Rich to avoid mutex conflicts
            self.progress_manager.pause_for_training()
            self.logger.info("Rich progress paused for training phase")
            return

        # During training phase, don't update Rich (it's paused)
        if hasattr(self, '_training_paused') and phase == 'training':
            return

        # Resume Rich progress when deployment phase starts or when we reach complete
        if (phase in ['deployment', 'complete'] or progress >= 96) and hasattr(self, '_training_paused'):
            del self._training_paused
            self.progress_manager.resume_after_training()
            self.logger.info("Rich progress resumed after training")

        # Check if this is an annotation update
        if phase == 'annotation' and subtask:
            current = subtask.get('current', 0)
            total = subtask.get('total', 0)

            # Always try to capture the latest annotation for live display
            sample_json = self._try_get_last_annotation_json()
            if sample_json:
                subtask['json_data'] = sample_json

        # Call the enhanced progress manager's update only if not paused
        if not hasattr(self, '_training_paused'):
            self.progress_manager.update_progress(
                phase=phase,
                progress=progress,
                message=message,
                subtask=subtask
            )

    def _try_get_last_annotation_json(self) -> Optional[Dict]:
        """Try to get the last annotation JSON from the pipeline state"""
        try:
            # First try to get from annotator directly
            if hasattr(self.pipeline, 'annotator') and hasattr(self.pipeline.annotator, 'last_annotation'):
                if self.pipeline.annotator.last_annotation:
                    self.logger.debug("Retrieved last annotation from annotator")
                    return self.pipeline.annotator.last_annotation

            # Fallback: Access pipeline state if available
            if hasattr(self.pipeline, 'state') and self.pipeline.state:
                if self.pipeline.state.annotation_results:
                    # Try to get last annotation
                    results = self.pipeline.state.annotation_results

                    # If it's a DataFrame, get last row with annotation
                    if 'data' in results and hasattr(results['data'], 'iloc'):
                        df = results['data']
                        if 'annotation' in df.columns:
                            # Get last non-null annotation
                            annotations = df[df['annotation'].notna()]['annotation']
                            if not annotations.empty:
                                last_annotation = annotations.iloc[-1]
                                if isinstance(last_annotation, str):
                                    return json.loads(last_annotation)
                                elif isinstance(last_annotation, dict):
                                    return last_annotation
        except Exception as e:
            self.logger.debug(f"Could not get annotation JSON: {e}")

        # Return None if we can't get real data - don't show fake examples
        return None

    def capture_annotation_data(self, data: Dict):
        """Method to be called by annotator to capture JSON data"""
        self.tracker.count += 1
        self.tracker.last_json = data
        self.tracker.json_buffer.append(data)

        # Keep only last 10 items in buffer
        if len(self.tracker.json_buffer) > 10:
            self.tracker.json_buffer = self.tracker.json_buffer[-10:]

    def capture_error(self, error: str, item_info: Optional[str] = None):
        """Method to be called when an error occurs"""
        self.tracker.last_error = error
        self.progress_manager.show_error(error, item_info)

    def run_pipeline(self, config: Dict[str, Any]):
        """Run the pipeline with enhanced progress tracking"""
        try:
            # The progress manager should already be started via context manager
            return self.pipeline.run_pipeline(config)
        except Exception as e:
            # Capture any pipeline errors
            self.capture_error(str(e), "Pipeline execution")
            raise
        finally:
            # Restore original callback
            if self.original_callback:
                self.pipeline.progress_callback = self.original_callback