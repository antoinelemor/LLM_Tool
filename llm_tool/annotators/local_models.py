#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
local_models.py

MAIN OBJECTIVE:
---------------
This script provides client implementations for local LLM models including
Ollama and LlamaCPP with comprehensive error handling and retry mechanisms.

Dependencies:
-------------
- sys
- subprocess
- logging
- typing
- time
- json

MAIN FEATURES:
--------------
1) Ollama client with model management
2) LlamaCPP client for GGUF models
3) Model listing and availability checking
4) Automatic retry mechanisms
5) JSON response handling

Author:
-------
Antoine Lemor
"""

import logging
import time
import json
import subprocess
import signal
import threading
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import os
from pathlib import Path

# Try to import Ollama
try:
    from ollama import generate, chat
    # DO NOT import 'list' at module level - it causes mutex locks
    # Import it dynamically when needed instead
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    generate = None
    chat = None

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False
    Llama = None


class BaseLocalClient(ABC):
    """Base class for local model clients"""

    def __init__(self, model_name: str, **kwargs):
        """Initialize the local client"""
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response from the model"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available"""
        pass


class OllamaClient(BaseLocalClient):
    """Ollama client implementation"""

    def __init__(self, model_name: str, **kwargs):
        """Initialize Ollama client"""
        super().__init__(model_name, **kwargs)

        self.logger.info(f"[1/4] Initializing OllamaClient for {model_name}")

        if not HAS_OLLAMA:
            raise ImportError("Ollama library not installed. Install with: pip install ollama")

        self.options = kwargs.get('options', {})
        # Default timeout of 5 minutes per request (can be overridden via kwargs or per-call)
        self.default_timeout = kwargs.get('timeout', 300)

        # Generation counter for periodic model reload
        self._generation_count = 0
        # Reload model every N generations to prevent memory accumulation (0 = disabled)
        self._reload_every = kwargs.get('reload_every', 3000)
        # Short timeout for health check test requests
        self._health_check_timeout = kwargs.get('health_check_timeout', 30)

        self.logger.info(f"[2/4] Checking Ollama service...")
        self._check_ollama_service()

        self.logger.info(f"[3/4] Checking if model {model_name} is available...")
        # Check if model is available
        if not self.is_available():
            self.logger.warning(f"Model {model_name} not found in Ollama. Attempting to pull...")
            self._pull_model()
        else:
            self.logger.info(f"[4/4] Model {model_name} is available ✓")

    def _check_ollama_service(self):
        """Check if Ollama service is running and healthy"""
        try:
            # Try to list models to check if service is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama service not running. Start with: ollama serve")

            # Check for stuck models
            self._check_and_recover_stuck_models()

        except FileNotFoundError:
            raise RuntimeError("Ollama not installed. Install from: https://ollama.ai")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama service not responding")

    def _get_running_models(self) -> List[Dict[str, str]]:
        """Get list of currently running/loaded models with their status"""
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return []

            lines = result.stdout.strip().splitlines()
            models = []

            for line in lines:
                # Skip header
                if "NAME" in line and "SIZE" in line:
                    continue
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 6:
                    model_info = {
                        'name': parts[0],
                        'id': parts[1],
                        'size': parts[2],
                        'status': parts[-1] if parts[-1] in ['Stopping...', 'Running'] else 'Running'
                    }
                    models.append(model_info)

            return models
        except Exception as e:
            self.logger.warning(f"Could not get running models: {e}")
            return []

    def _check_and_recover_stuck_models(self, max_wait: int = 30) -> bool:
        """
        Check for stuck models (in 'Stopping...' state) and wait/recover if needed.

        Parameters
        ----------
        max_wait : int
            Maximum seconds to wait for a stopping model before forcing recovery

        Returns
        -------
        bool
            True if recovery was needed and performed, False otherwise
        """
        running = self._get_running_models()

        stuck_models = [m for m in running if m.get('status') == 'Stopping...']
        if not stuck_models:
            return False

        self.logger.warning(f"Found {len(stuck_models)} model(s) in 'Stopping' state: {[m['name'] for m in stuck_models]}")
        self.logger.info(f"Waiting up to {max_wait}s for models to finish stopping...")

        # Wait for models to finish stopping
        start_time = time.time()
        while time.time() - start_time < max_wait:
            time.sleep(2)
            running = self._get_running_models()
            stuck_models = [m for m in running if m.get('status') == 'Stopping...']

            if not stuck_models:
                self.logger.info("All stuck models have finished stopping")
                return True

            elapsed = int(time.time() - start_time)
            self.logger.info(f"Still waiting... ({elapsed}s/{max_wait}s)")

        # If still stuck after waiting, try to force stop
        self.logger.warning(f"Models still stuck after {max_wait}s, attempting force recovery...")
        return self._force_stop_stuck_models(stuck_models)

    def _force_stop_stuck_models(self, stuck_models: List[Dict[str, str]]) -> bool:
        """
        Force stop stuck models by stopping Ollama service.

        Parameters
        ----------
        stuck_models : list
            List of stuck model info dicts

        Returns
        -------
        bool
            True if recovery succeeded
        """
        try:
            self.logger.warning("Attempting to stop stuck Ollama models...")

            # Try using ollama stop command first (if available)
            for model in stuck_models:
                try:
                    result = subprocess.run(
                        ["ollama", "stop", model['name']],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        self.logger.info(f"Successfully stopped {model['name']}")
                except Exception as e:
                    self.logger.warning(f"Could not stop {model['name']}: {e}")

            # Wait and check if it worked
            time.sleep(3)
            running = self._get_running_models()
            stuck = [m for m in running if m.get('status') == 'Stopping...']

            if not stuck:
                self.logger.info("Successfully recovered from stuck state")
                return True

            # Last resort: suggest manual intervention
            self.logger.error(
                "Could not automatically recover stuck Ollama models. "
                "Please manually restart Ollama service: "
                "pkill ollama && ollama serve"
            )
            return False

        except Exception as e:
            self.logger.error(f"Error during force stop: {e}")
            return False

    def health_check(self, active_test: bool = False) -> Dict[str, Any]:
        """
        Perform a health check on Ollama service.

        Parameters
        ----------
        active_test : bool
            If True, perform an actual generation test to verify model responsiveness.
            This is slower but more reliable for detecting stuck models.

        Returns
        -------
        dict
            Health status with keys: 'healthy', 'service_running', 'model_loaded',
            'stuck_models', 'responds_to_requests', 'error'
        """
        result = {
            'healthy': False,
            'service_running': False,
            'model_loaded': False,
            'stuck_models': [],
            'responds_to_requests': None,  # None = not tested, True/False = tested
            'error': None
        }

        try:
            # Check service
            check = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            result['service_running'] = (check.returncode == 0)

            if not result['service_running']:
                result['error'] = "Ollama service not running"
                return result

            # Check for stuck models
            running = self._get_running_models()
            result['stuck_models'] = [m['name'] for m in running if m.get('status') == 'Stopping...']

            # Check if our model is loaded
            for m in running:
                if m['name'].startswith(self.model_name.split(':')[0]):
                    result['model_loaded'] = True
                    break

            # Basic health check without active test
            basic_healthy = result['service_running'] and len(result['stuck_models']) == 0

            # If active test requested, verify model actually responds
            if active_test and basic_healthy:
                result['responds_to_requests'] = self._test_model_responsiveness()
                result['healthy'] = basic_healthy and result['responds_to_requests']
            else:
                result['healthy'] = basic_healthy

        except Exception as e:
            result['error'] = str(e)

        return result

    def _test_model_responsiveness(self) -> bool:
        """
        Test if the model actually responds to a simple request.

        Returns
        -------
        bool
            True if model responds within timeout, False otherwise
        """
        self.logger.info(f"Testing model responsiveness with short request (timeout={self._health_check_timeout}s)...")

        try:
            # Simple test prompt that should generate a quick response
            test_prompt = "Reply with only the word 'OK'."

            response = self._generate_with_timeout(
                prompt=test_prompt,
                options={'temperature': 0, 'num_predict': 10},
                timeout=self._health_check_timeout
            )

            if response and len(response.strip()) > 0:
                self.logger.info(f"Model responsive: got '{response.strip()[:20]}...'")
                return True
            else:
                self.logger.warning("Model did not respond to test request")
                return False

        except Exception as e:
            self.logger.warning(f"Model responsiveness test failed: {e}")
            return False

    def _hard_reset_model(self) -> bool:
        """
        Perform a hard reset of the model by unloading and reloading it.

        This is used when the model appears loaded but doesn't respond to requests.

        Returns
        -------
        bool
            True if reset was successful, False otherwise
        """
        self.logger.warning(f"Performing hard reset of model {self.model_name}...")

        try:
            # Step 1: Stop/unload the model
            self.logger.info("Step 1/3: Unloading model...")
            stop_result = subprocess.run(
                ["ollama", "stop", self.model_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            if stop_result.returncode != 0:
                # Try with base name if full name failed
                base_name = self.model_name.split(':')[0]
                subprocess.run(
                    ["ollama", "stop", base_name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            # Step 2: Wait for model to fully unload
            self.logger.info("Step 2/3: Waiting for unload to complete...")
            max_wait = 60
            start_time = time.time()

            while time.time() - start_time < max_wait:
                running = self._get_running_models()
                our_model_running = any(
                    m['name'].startswith(self.model_name.split(':')[0])
                    for m in running
                )

                if not our_model_running:
                    self.logger.info("Model unloaded successfully")
                    break

                time.sleep(2)
            else:
                self.logger.warning(f"Model still running after {max_wait}s wait")

            # Step 3: Trigger reload by sending a simple request
            self.logger.info("Step 3/3: Reloading model with test request...")
            time.sleep(3)  # Brief pause before reload

            # Reset generation counter
            self._generation_count = 0

            # Test if model responds after reset
            if self._test_model_responsiveness():
                self.logger.info("Hard reset successful - model is responsive")
                return True
            else:
                self.logger.error("Hard reset failed - model still unresponsive")
                return False

        except Exception as e:
            self.logger.error(f"Hard reset failed with error: {e}")
            return False

    def _maybe_reload_model(self) -> bool:
        """
        Check if periodic model reload is needed and perform it.

        Returns
        -------
        bool
            True if reload was performed, False otherwise
        """
        if self._reload_every <= 0:
            return False

        if self._generation_count >= self._reload_every:
            self.logger.info(
                f"Periodic reload triggered after {self._generation_count} generations "
                f"(threshold: {self._reload_every})"
            )
            return self._hard_reset_model()

        return False

    def _pull_model(self):
        """Pull model from Ollama registry"""
        try:
            self.logger.info(f"Pulling model {self.model_name}...")
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for download
            )
            if result.returncode == 0:
                self.logger.info(f"Successfully pulled model {self.model_name}")
            else:
                raise RuntimeError(f"Failed to pull model: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
            raise

    def is_available(self) -> bool:
        """Check if the model is available in Ollama

        Handles model names with and without tags (e.g., 'nemotron' matches 'nemotron:latest')
        """
        try:
            models = self.list_models()

            # Exact match
            if self.model_name in models:
                return True

            # Check if model_name without tag matches any model
            # e.g., 'nemotron' should match 'nemotron:latest'
            base_name = self.model_name.split(':')[0]
            for model in models:
                model_base = model.split(':')[0]
                if base_name == model_base:
                    self.logger.info(f"Model '{self.model_name}' matched to '{model}'")
                    return True

            return False
        except Exception as e:
            self.logger.warning(f"Error checking model availability: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return []
            
            lines = result.stdout.strip().splitlines()
            models = []
            
            for line in lines:
                # Skip header
                if "NAME" in line and "MODIFIED" in line:
                    continue
                if not line.strip():
                    continue

                parts = line.split()
                if parts:
                    model_name = parts[0]
                    # Keep the full model name with tag (e.g., gpt-oss:120b)
                    models.append(model_name)
            
            return models
        except:
            return []

    def _generate_with_timeout(
        self,
        prompt: str,
        options: Dict[str, Any],
        timeout: float
    ) -> Optional[str]:
        """
        Execute Ollama generate with a timeout.

        Uses ThreadPoolExecutor to wrap the blocking call and enforce timeout.
        """
        def _do_generate():
            response = generate(
                model=self.model_name,
                prompt=prompt,
                options=options
            )
            # Extract response
            if isinstance(response, dict):
                return response.get('response', '').strip()
            elif hasattr(response, 'get'):
                return response.get('response', '').strip()
            else:
                return str(response)

        # Use a thread pool to enforce timeout on the blocking Ollama call
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_generate)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                self.logger.error(f"Ollama generation timed out after {timeout}s")
                # Try to cancel the future (may not actually stop the underlying call)
                future.cancel()
                return None

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        format: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response from Ollama model.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model
        temperature : float
            Temperature parameter (0-2)
        max_tokens : int
            Maximum tokens in response
        format : str, optional
            Response format ('json' for JSON mode)
        timeout : float, optional
            Timeout in seconds for each generation attempt.
            Defaults to self.default_timeout (300s = 5 minutes).

        Returns
        -------
        str or None
            Generated response or None on error
        """
        # Check for periodic model reload to prevent memory accumulation
        self._maybe_reload_model()

        # Build options
        options = self.options.copy()
        options['temperature'] = temperature
        options['num_predict'] = max_tokens

        # Add any additional options from kwargs
        for key in ['seed', 'top_p', 'top_k', 'num_thread']:
            if key in kwargs:
                options[key] = kwargs[key]

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self.default_timeout

        # Track consecutive timeouts for stall detection
        consecutive_timeouts = 0
        max_consecutive_timeouts = 2  # After 2 timeouts, check Ollama health
        hard_reset_attempted = False

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Generating response with {self.model_name} (attempt {attempt + 1}/{self.max_retries}, timeout={request_timeout}s, gen_count={self._generation_count})")

                # Use timeout-wrapped generate
                content = self._generate_with_timeout(prompt, options, request_timeout)

                if not content:
                    consecutive_timeouts += 1
                    self.logger.warning(f"Empty response from Ollama (attempt {attempt + 1}, consecutive timeouts: {consecutive_timeouts})")

                    # Check if Ollama might be stuck
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        self.logger.warning("Multiple consecutive failures detected, performing ACTIVE health check...")

                        # Use ACTIVE health check - actually test if model responds
                        health = self.health_check(active_test=True)

                        if health.get('stuck_models'):
                            self.logger.warning(f"Found stuck models: {health['stuck_models']}, attempting recovery...")
                            self._check_and_recover_stuck_models(max_wait=60)
                            consecutive_timeouts = 0  # Reset after recovery attempt

                        elif health.get('model_loaded') and not health.get('responds_to_requests'):
                            # Model is loaded but doesn't respond - need hard reset
                            if not hard_reset_attempted:
                                self.logger.warning(
                                    "Model loaded but unresponsive - attempting HARD RESET..."
                                )
                                hard_reset_attempted = True
                                if self._hard_reset_model():
                                    consecutive_timeouts = 0  # Reset after successful hard reset
                                    # Continue to retry with the same attempt number
                                    continue
                                else:
                                    self.logger.error("Hard reset failed - model may need manual restart")

                        elif not health.get('healthy'):
                            self.logger.error(f"Ollama unhealthy: {health.get('error', 'unknown error')}")

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return None

                # Success - increment generation counter and reset timeout counter
                self._generation_count += 1
                consecutive_timeouts = 0
                self.logger.info(f"Successfully generated response ({len(content)} chars)")
                return content

            except Exception as e:
                consecutive_timeouts += 1
                self.logger.error(f"Ollama generation failed (attempt {attempt + 1}): {e}")

                # Check for stuck models on repeated failures
                if consecutive_timeouts >= max_consecutive_timeouts:
                    self.logger.warning("Checking Ollama health after repeated failures...")
                    try:
                        # Use active health check
                        health = self.health_check(active_test=True)

                        if health.get('stuck_models'):
                            self._check_and_recover_stuck_models(max_wait=30)
                        elif health.get('model_loaded') and not health.get('responds_to_requests'):
                            # Try hard reset if not already attempted
                            if not hard_reset_attempted:
                                hard_reset_attempted = True
                                if self._hard_reset_model():
                                    consecutive_timeouts = 0
                                    continue  # Retry after successful reset

                    except Exception as recovery_error:
                        self.logger.error(f"Recovery attempt failed: {recovery_error}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return None

        return None


class LlamaCPPClient(BaseLocalClient):
    """LlamaCPP client implementation for GGUF models"""

    def __init__(self, model_path: str, **kwargs):
        """Initialize LlamaCPP client"""
        if not HAS_LLAMACPP:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python"
            )
        
        # Use model_path as model_name for consistency
        super().__init__(model_path, **kwargs)
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Model parameters
        self.n_ctx = kwargs.get('n_ctx', 2048)  # Context size
        self.n_threads = kwargs.get('n_threads', 4)  # CPU threads
        self.n_gpu_layers = kwargs.get('n_gpu_layers', 0)  # GPU layers
        self.seed = kwargs.get('seed', -1)
        self.f16_kv = kwargs.get('f16_kv', True)
        self.logits_all = kwargs.get('logits_all', False)
        self.vocab_only = kwargs.get('vocab_only', False)
        self.use_mlock = kwargs.get('use_mlock', False)
        self.embedding = kwargs.get('embedding', False)
        
        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize the LlamaCPP model"""
        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                f16_kv=self.f16_kv,
                logits_all=self.logits_all,
                vocab_only=self.vocab_only,
                use_mlock=self.use_mlock,
                embedding=self.embedding
            )
            self.logger.info(f"Successfully loaded model: {self.model_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def is_available(self) -> bool:
        """Check if the model is loaded and available"""
        return hasattr(self, 'model') and self.model is not None

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response from LlamaCPP model.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model
        temperature : float
            Temperature for sampling
        max_tokens : int
            Maximum tokens to generate
        top_p : float
            Top-p sampling parameter
        top_k : int
            Top-k sampling parameter
        repeat_penalty : float
            Repetition penalty
        
        Returns
        -------
        str or None
            Generated response or None on error
        """
        if not self.is_available():
            self.logger.error("Model not available")
            return None
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                # Generate response
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    echo=False  # Don't include prompt in output
                )
                
                # Extract text from response
                if isinstance(output, dict):
                    text = output.get('choices', [{}])[0].get('text', '').strip()
                else:
                    text = str(output).strip()
                
                if not text:
                    self.logger.warning(f"Empty response (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return None
                
                # If JSON mode requested, validate
                if kwargs.get('json_mode', False):
                    try:
                        json.loads(text)
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
                        if json_match:
                            text = json_match.group(0)
                            try:
                                json.loads(text)
                            except:
                                self.logger.warning(f"Could not extract valid JSON (attempt {attempt + 1})")
                                if attempt < self.max_retries - 1:
                                    time.sleep(self.retry_delay * (attempt + 1))
                                    continue
                                return None
                        else:
                            self.logger.warning(f"No JSON found in response (attempt {attempt + 1})")
                            if attempt < self.max_retries - 1:
                                time.sleep(self.retry_delay * (attempt + 1))
                                continue
                            return None
                
                return text
                
            except Exception as e:
                self.logger.error(f"Generation failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return None
        
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_available():
            return {}
        
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_path.name,
            'context_size': self.n_ctx,
            'threads': self.n_threads,
            'gpu_layers': self.n_gpu_layers
        }


def list_ollama_models() -> List[str]:
    """
    List all available Ollama models.

    Returns
    -------
    list
        List of available model names
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().splitlines()
        models = []

        for line in lines:
            # Skip header
            if "NAME" in line and "MODIFIED" in line:
                continue
            if not line.strip():
                continue

            parts = line.split()
            if parts:
                models.append(parts[0])

        return models
    except Exception:
        return []


def find_gguf_models(directory: str = None) -> List[str]:
    """
    Find GGUF model files in a directory.
    
    Parameters
    ----------
    directory : str, optional
        Directory to search. If None, searches common locations
    
    Returns
    -------
    list
        List of GGUF model file paths
    """
    models = []
    
    # Default search directories
    if directory:
        search_dirs = [Path(directory)]
    else:
        search_dirs = [
            Path.home() / 'models',
            Path.home() / '.cache' / 'llama-cpp',
            Path.home() / '.local' / 'share' / 'models',
            Path('/usr/local/models'),
            Path.cwd() / 'models'
        ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            # Search for GGUF files
            for model_file in search_dir.glob('**/*.gguf'):
                models.append(str(model_file))
    
    return models


def create_local_client(provider: str, model_or_path: str, **kwargs) -> BaseLocalClient:
    """
    Factory function to create appropriate local model client.
    
    Parameters
    ----------
    provider : str
        Local provider ('ollama' or 'llamacpp')
    model_or_path : str
        Model name (for Ollama) or path to model file (for LlamaCPP)
    **kwargs
        Additional configuration options
    
    Returns
    -------
    BaseLocalClient
        Appropriate local client instance
    
    Raises
    ------
    ValueError
        If provider is not supported
    """
    provider = provider.lower()
    
    if provider == 'ollama':
        return OllamaClient(model_or_path, **kwargs)
    elif provider == 'llamacpp':
        return LlamaCPPClient(model_or_path, **kwargs)
    else:
        raise ValueError(f"Unsupported local provider: {provider}")
