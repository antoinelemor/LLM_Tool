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
    from ollama import Client as OllamaSDKClient
    from ollama import ResponseError as OllamaResponseError
    # DO NOT import the module-level `generate`/`list` helpers: they bind a client
    # to the ambient OLLAMA_HOST at import time, which makes the endpoint
    # impossible to change per instance. Build an explicit Client instead.
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    OllamaSDKClient = None

    class OllamaResponseError(Exception):
        """Placeholder so `except OllamaResponseError` stays valid without the SDK."""
        status_code = None


# Default endpoints. The local daemon needs no credentials; the hosted service at
# ollama.com serves the same REST API but requires a Bearer token for inference
# (model listing via /api/tags is public).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CLOUD_HOST = "https://ollama.com"


class OllamaFatalError(RuntimeError):
    """
    Raised for a condition that every subsequent row would hit identically.

    A rejected credential or a model the endpoint does not serve will not resolve
    itself on retry, so callers should abandon the run rather than replay the
    failure for every row — which against a metered endpoint is also billable.
    """


def _is_local_host(host: str) -> bool:
    """True when `host` points at a daemon on this machine."""
    if not host:
        return True
    lowered = host.lower()
    return any(
        marker in lowered
        for marker in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]")
    )


class OllamaEndpoint:
    """
    An Ollama server address plus the credential needed to reach it.

    Ollama Cloud and a local `ollama serve` expose the same REST API, so the only
    thing that varies between them is the base URL and whether an Authorization
    header is required. Keeping both in one object means every call site takes a
    single argument instead of threading two.
    """

    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None):
        self.host = (host or DEFAULT_OLLAMA_HOST).rstrip('/')
        self.api_key = (api_key or '').strip() or None

    @property
    def is_local(self) -> bool:
        """True when the endpoint is a daemon on this machine."""
        return _is_local_host(self.host)

    @property
    def is_cloud(self) -> bool:
        """True when the endpoint is remote (cloud or a networked server)."""
        return not self.is_local

    @property
    def label(self) -> str:
        """Short human-readable name for menus and logs."""
        if self.is_local:
            return "Local Ollama"
        if self.host.rstrip('/') == OLLAMA_CLOUD_HOST:
            return "Ollama Cloud"
        return f"Ollama @ {self.host}"

    def headers(self) -> Dict[str, str]:
        """Auth headers for this endpoint (empty for an unauthenticated local daemon)."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def client(self, timeout: Optional[float] = None):
        """Build an `ollama.Client` bound to this endpoint."""
        if not HAS_OLLAMA:
            raise ImportError("Ollama library not installed. Install with: pip install ollama")
        return OllamaSDKClient(host=self.host, headers=self.headers(), timeout=timeout)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for run configs (the key is kept so resumed runs stay authenticated)."""
        return {'host': self.host, 'api_key': self.api_key}

    def __repr__(self) -> str:
        return f"OllamaEndpoint(host={self.host!r}, authenticated={bool(self.api_key)})"


def resolve_ollama_endpoint(
    host: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OllamaEndpoint:
    """
    Resolve an Ollama endpoint from explicit values, then the environment.

    Precedence: explicit argument > OLLAMA_HOST / OLLAMA_API_KEY > stored key >
    local default. A remote host with no key still resolves — listing models is
    public, so the caller can show a picker and only then ask for a credential.
    """
    resolved_host = host or os.environ.get('OLLAMA_HOST') or DEFAULT_OLLAMA_HOST
    resolved_key = api_key or os.environ.get('OLLAMA_API_KEY')

    if not resolved_key and not _is_local_host(resolved_host):
        # Only reach for the vault when a credential can actually be needed.
        try:
            from llm_tool.config.api_key_manager import APIKeyManager
            resolved_key = APIKeyManager().get_key('ollama')
        except Exception:
            resolved_key = None

    return OllamaEndpoint(host=resolved_host, api_key=resolved_key)


def probe_ollama(
    endpoint: Optional[OllamaEndpoint] = None,
    model: Optional[str] = None,
    timeout: float = 15.0,
    generation_timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Check that an Ollama endpoint is reachable, authenticated, and serving a model.

    This is the pre-flight the UI runs before committing to a long annotation job,
    so it reports *why* it failed rather than raising: a wrong host, a missing key
    and a missing model all need different fixes.

    Parameters
    ----------
    endpoint : OllamaEndpoint, optional
        Endpoint to test. Defaults to the resolved ambient endpoint.
    model : str, optional
        If given, also verify the model is served and answers a trivial prompt.
    timeout : float
        Timeout for the reachability/listing call.
    generation_timeout : float
        Timeout for the one-token test generation.

    Returns
    -------
    dict
        Keys: 'reachable', 'authenticated', 'models', 'model_available',
        'responds', 'latency_ms', 'error', 'hint', 'endpoint'.
    """
    endpoint = endpoint or resolve_ollama_endpoint()
    result: Dict[str, Any] = {
        'reachable': False,
        'authenticated': None,     # None = not applicable/untested
        'models': [],
        'model_available': None,
        'responds': None,
        'latency_ms': None,
        'error': None,
        'hint': None,
        'endpoint': endpoint,
    }

    if not HAS_OLLAMA:
        result['error'] = "Ollama library not installed"
        result['hint'] = "pip install ollama"
        return result

    # 1. Reachability + model listing (public on ollama.com, unauthenticated locally).
    try:
        client = endpoint.client(timeout=timeout)
        listing = client.list()
        result['models'] = [m.model for m in getattr(listing, 'models', []) if getattr(m, 'model', None)]
        result['reachable'] = True
    except ConnectionError as e:
        result['error'] = f"Cannot reach {endpoint.host}: {e}"
        result['hint'] = (
            "Start the daemon with: ollama serve"
            if endpoint.is_local else
            f"Check the host is correct and reachable: {endpoint.host}"
        )
        return result
    except OllamaResponseError as e:
        status = getattr(e, 'status_code', None)
        result['error'] = f"{endpoint.host} returned {status or 'an error'}: {e}"
        if status in (401, 403):
            result['authenticated'] = False
            result['hint'] = "This endpoint needs an API key. Add one for the 'ollama' provider."
        return result
    except Exception as e:
        result['error'] = f"Unexpected error contacting {endpoint.host}: {e}"
        return result

    if model is None:
        return result

    # 2. Is the requested model served here?
    base = model.split(':')[0]
    result['model_available'] = any(
        m == model or m.split(':')[0] == base for m in result['models']
    )

    # 3. Does it actually answer? This is what catches a bad key: listing is public
    #    on ollama.com but inference is not, so only a real generation proves auth.
    try:
        start = time.time()
        client = endpoint.client(timeout=generation_timeout)
        response = client.generate(
            model=model,
            prompt="Reply with only the word 'OK'.",
            options={'temperature': 0, 'num_predict': 5},
        )
        text = (getattr(response, 'response', '') or '').strip()
        result['latency_ms'] = int((time.time() - start) * 1000)
        result['responds'] = bool(text)
        result['authenticated'] = True
        if not text:
            result['error'] = f"Model {model} returned an empty response"
    except OllamaResponseError as e:
        status = getattr(e, 'status_code', None)
        result['responds'] = False
        if status in (401, 403):
            result['authenticated'] = False
            result['error'] = f"Authentication failed for {endpoint.host} (HTTP {status})"
            result['hint'] = (
                "Ollama Cloud requires a valid API key. Set OLLAMA_API_KEY or store "
                "one for the 'ollama' provider."
            )
        elif status == 404:
            result['error'] = f"Model '{model}' not found on {endpoint.host}"
            result['hint'] = (
                f"Pull it first: ollama pull {model}"
                if endpoint.is_local else
                "Pick a model from this endpoint's catalogue."
            )
        else:
            result['error'] = f"Generation failed (HTTP {status}): {e}"
        return result
    except Exception as e:
        result['responds'] = False
        result['error'] = f"Generation failed: {e}"
        return result

    return result

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
        """
        Initialize Ollama client.

        Parameters
        ----------
        model_name : str
            Model to generate with (e.g. 'gemma4:31b').
        host : str, optional
            Base URL of the Ollama server. Defaults to OLLAMA_HOST or localhost.
            Pass OLLAMA_CLOUD_HOST to use Ollama Cloud.
        api_key : str, optional
            Bearer token, required by Ollama Cloud. Falls back to OLLAMA_API_KEY
            or the stored 'ollama' key.
        endpoint : OllamaEndpoint, optional
            Pre-resolved endpoint; takes precedence over host/api_key.
        """
        super().__init__(model_name, **kwargs)

        self.logger.info(f"[1/4] Initializing OllamaClient for {model_name}")

        if not HAS_OLLAMA:
            raise ImportError("Ollama library not installed. Install with: pip install ollama")

        self.endpoint = kwargs.get('endpoint') or resolve_ollama_endpoint(
            host=kwargs.get('host'),
            api_key=kwargs.get('api_key'),
        )

        self.options = kwargs.get('options', {})
        # Default timeout of 5 minutes per request (can be overridden via kwargs or per-call)
        self.default_timeout = kwargs.get('timeout', 300)

        # Generation counter for periodic model reload
        self._generation_count = 0
        # Reload model every N generations to prevent memory accumulation (0 = disabled)
        self._reload_every = kwargs.get('reload_every', 3000)
        # Short timeout for health check test requests
        self._health_check_timeout = kwargs.get('health_check_timeout', 30)

        # One client per instance, bound to this endpoint.
        self._client = self.endpoint.client(timeout=self.default_timeout)

        self.logger.info(f"[2/4] Checking Ollama service at {self.endpoint.host}...")
        self._check_ollama_service()

        self.logger.info(f"[3/4] Checking if model {model_name} is available...")
        # Check if model is available
        if not self.is_available():
            if self.endpoint.is_local:
                # Several hosted-only models share a name with a public registry entry,
                # so an absent-minded pick would start a multi-gigabyte download of a
                # model the user meant to run in the cloud. Check before pulling.
                if model_name in list_ollama_models(host=OLLAMA_CLOUD_HOST):
                    raise RuntimeError(
                        f"Model '{model_name}' is served by Ollama Cloud, not by "
                        f"{self.endpoint.label} ({self.endpoint.host}). Select it from the "
                        f"Ollama Cloud catalogue, or pull it locally first if you meant to."
                    )
                self.logger.warning(f"Model {model_name} not found in Ollama. Attempting to pull...")
                self._pull_model()
            else:
                # Remote catalogues are fixed — a missing model is a selection error,
                # and pulling would target the wrong machine.
                raise RuntimeError(
                    f"Model '{model_name}' is not served by {self.endpoint.label} "
                    f"({self.endpoint.host}). Choose a model from that endpoint's catalogue."
                )
        else:
            self.logger.info(f"[4/4] Model {model_name} is available ✓")

    @property
    def is_local(self) -> bool:
        """True when this client talks to a daemon on this machine."""
        return self.endpoint.is_local

    def _check_ollama_service(self):
        """Check that the configured Ollama endpoint is reachable and healthy"""
        try:
            # Listing is the cheapest call that proves the server is up. It runs
            # over HTTP against the configured host, so it works for a remote
            # endpoint on a machine with no ollama CLI installed.
            self._client.list()
        except ConnectionError as e:
            if self.endpoint.is_local:
                raise RuntimeError(
                    f"Ollama service not running at {self.endpoint.host}. "
                    f"Start with: ollama serve"
                ) from e
            raise RuntimeError(
                f"Cannot reach Ollama endpoint {self.endpoint.host}: {e}"
            ) from e
        except OllamaResponseError as e:
            status = getattr(e, 'status_code', None)
            if status in (401, 403):
                raise RuntimeError(
                    f"Authentication failed for {self.endpoint.host} (HTTP {status}). "
                    f"Set OLLAMA_API_KEY or store a key for the 'ollama' provider."
                ) from e
            raise RuntimeError(f"Ollama endpoint {self.endpoint.host} error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama service not responding at {self.endpoint.host}: {e}") from e

        # Stuck-model recovery unloads models, which is only ours to do locally.
        if self.endpoint.is_local:
            self._check_and_recover_stuck_models()

    def _get_running_models(self) -> List[Dict[str, str]]:
        """Get list of currently running/loaded models with their status"""
        # Only a local daemon has a meaningful notion of "loaded models" we may act on.
        if not self.endpoint.is_local:
            return []
        try:
            running = self._client.ps()
            models = []
            for m in getattr(running, 'models', []) or []:
                name = getattr(m, 'model', None) or getattr(m, 'name', None)
                if not name:
                    continue
                models.append({
                    'name': name,
                    'id': getattr(m, 'digest', '') or '',
                    'size': str(getattr(m, 'size', '') or ''),
                    'status': 'Running',
                })
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

            # Unloading models is only ours to do on a daemon we own.
            if not self.endpoint.is_local:
                self.logger.warning(
                    f"Skipping stuck-model recovery: {self.endpoint.host} is a remote endpoint."
                )
                return False

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
            # Check service over HTTP against the configured endpoint.
            try:
                self._client.list()
                result['service_running'] = True
            except Exception as e:
                result['service_running'] = False
                result['error'] = f"Ollama endpoint {self.endpoint.host} not responding: {e}"
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
        # A hard reset unloads the model from the server. Against a shared or
        # hosted endpoint that is not ours to do, and the unresponsiveness is
        # almost certainly network- or quota-related rather than a stuck model.
        if not self.endpoint.is_local:
            self.logger.warning(
                f"Skipping hard reset: {self.endpoint.host} is a remote endpoint."
            )
            return False

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
        """Pull model into the local Ollama server"""
        if not self.endpoint.is_local:
            # Pulling targets the server's own disk; for a remote endpoint that
            # is both futile and not ours to trigger.
            raise RuntimeError(
                f"Cannot pull '{self.model_name}' into remote endpoint {self.endpoint.host}."
            )
        try:
            self.logger.info(f"Pulling model {self.model_name}...")
            # Pull over the API rather than the CLI so a machine without the
            # ollama binary can still populate its own daemon.
            self._client.pull(self.model_name)
            self.logger.info(f"Successfully pulled model {self.model_name}")
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

            # An untagged name is a request for whatever tag the server has, so
            # 'nemotron' may resolve to 'nemotron:latest'. An explicit tag is not
            # negotiable though — matching 'gpt-oss:20b' to a local 'gpt-oss:120b'
            # would pass this check and then 404 on every single generation.
            if ':' not in self.model_name:
                for model in models:
                    if model.split(':')[0] == self.model_name:
                        self.logger.info(f"Model '{self.model_name}' matched to '{model}'")
                        return True

            return False
        except Exception as e:
            self.logger.warning(f"Error checking model availability: {e}")
            return False

    def list_models(self) -> List[str]:
        """List models served by this client's endpoint"""
        try:
            listing = self._client.list()
            # Keep the full model name with tag (e.g., gpt-oss:120b)
            return [m.model for m in getattr(listing, 'models', []) if getattr(m, 'model', None)]
        except Exception as e:
            self.logger.warning(f"Could not list models from {self.endpoint.host}: {e}")
            return []

    def get_context_length(self) -> Optional[int]:
        """
        Ask the server for this model's real context window.

        Returns None when the endpoint does not report one, so callers can fall
        back to their own estimate rather than treating 0 as a valid window.
        """
        try:
            info = self._client.show(self.model_name)
            model_info = getattr(info, 'modelinfo', None) or {}
            for key, value in model_info.items():
                if key.endswith('.context_length') and isinstance(value, int):
                    return value
        except Exception as e:
            self.logger.debug(f"Could not read context length for {self.model_name}: {e}")
        return None

    def _generate_with_timeout(
        self,
        prompt: str,
        options: Dict[str, Any],
        timeout: float,
        format: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Execute Ollama generate with a timeout.

        Uses ThreadPoolExecutor to wrap the blocking call and enforce timeout.
        """
        def _do_generate():
            kwargs: Dict[str, Any] = {
                'model': self.model_name,
                'prompt': prompt,
                'options': options,
            }
            # 'json' or a JSON-schema dict; omitted entirely when unset so we
            # don't constrain models that were not asked to produce JSON.
            if format:
                kwargs['format'] = format

            response = self._client.generate(**kwargs)
            # Extract response
            if isinstance(response, dict):
                return response.get('response', '').strip()
            text = getattr(response, 'response', None)
            if text is not None:
                return text.strip()
            return str(response)

        # Enforce the timeout on the blocking Ollama call. Note we deliberately
        # avoid `with ThreadPoolExecutor(...)`: its __exit__ waits for the worker,
        # so a "timed out" call would still block the caller until the server
        # finished. Shut down without waiting and let the orphan thread die.
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_generate)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            self.logger.error(f"Ollama generation timed out after {timeout}s")
            future.cancel()
            return None
        finally:
            executor.shutdown(wait=False)

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
        format : str or dict, optional
            Response format ('json' for JSON mode, or a JSON schema dict)
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

        # Build options. Precedence: instance defaults < explicit temperature /
        # max_tokens < an `options` dict passed by the caller. That last case is
        # how the annotator forwards the sampling settings the user configured,
        # so it has to win.
        options = self.options.copy()
        options['temperature'] = temperature
        options['num_predict'] = max_tokens

        # Add any additional options from kwargs
        for key in ['seed', 'top_p', 'top_k', 'num_thread']:
            if key in kwargs:
                options[key] = kwargs[key]

        caller_options = kwargs.get('options')
        if isinstance(caller_options, dict):
            options.update(caller_options)

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
                content = self._generate_with_timeout(prompt, options, request_timeout, format=format)

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

            except OllamaResponseError as e:
                status = getattr(e, 'status_code', None)
                # A rejected credential or a missing model will not fix itself on
                # retry — and against a metered endpoint each retry costs quota.
                if status in (401, 403):
                    raise OllamaFatalError(
                        f"Authentication rejected by {self.endpoint.host} (HTTP {status}). "
                        f"Check the API key for the 'ollama' provider."
                    ) from e
                if status == 404:
                    raise OllamaFatalError(
                        f"Model '{self.model_name}' not found on {self.endpoint.host}."
                    ) from e
                self.logger.error(f"Ollama returned HTTP {status} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return None

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


def list_ollama_models(
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[OllamaEndpoint] = None,
    timeout: float = 10.0,
) -> List[str]:
    """
    List all models served by an Ollama endpoint.

    Parameters
    ----------
    host : str, optional
        Base URL of the server. Defaults to OLLAMA_HOST or localhost.
    api_key : str, optional
        Bearer token for an authenticated endpoint.
    endpoint : OllamaEndpoint, optional
        Pre-resolved endpoint; takes precedence over host/api_key.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    list
        List of available model names, or [] if the endpoint is unreachable.
    """
    if not HAS_OLLAMA:
        return []
    try:
        endpoint = endpoint or resolve_ollama_endpoint(host=host, api_key=api_key)
        listing = endpoint.client(timeout=timeout).list()
        return [m.model for m in getattr(listing, 'models', []) if getattr(m, 'model', None)]
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
