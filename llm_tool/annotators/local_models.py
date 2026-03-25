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
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
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
        """Check if Ollama service is running"""
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
        except FileNotFoundError:
            raise RuntimeError("Ollama not installed. Install from: https://ollama.ai")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama service not responding")

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
        """Check if the model is available in Ollama"""
        try:
            models = self.list_models()
            return self.model_name in models
        except:
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

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        format: Optional[str] = None,
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
        
        Returns
        -------
        str or None
            Generated response or None on error
        """
        # Build options
        options = self.options.copy()
        options['temperature'] = temperature
        options['num_predict'] = max_tokens
        
        # Add any additional options from kwargs
        for key in ['seed', 'top_p', 'top_k', 'num_thread']:
            if key in kwargs:
                options[key] = kwargs[key]
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Generating response with {self.model_name} (attempt {attempt + 1}/{self.max_retries})")

                # Always use generate() instead of chat() to avoid locking issues
                # The chat() function with format parameter can cause mutex locks
                response = generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options
                )

                # Extract response
                if isinstance(response, dict):
                    content = response.get('response', '').strip()
                elif hasattr(response, 'get'):
                    content = response.get('response', '').strip()
                else:
                    content = str(response)

                if not content:
                    self.logger.warning(f"Empty response from Ollama (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return None

                self.logger.info(f"Successfully generated response ({len(content)} chars)")
                return content
                    
            except Exception as e:
                self.logger.error(f"Ollama generation failed (attempt {attempt + 1}): {e}")
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
    if not HAS_OLLAMA:
        return []
    
    try:
        client = OllamaClient('dummy')  # Just to access list_models method
        return client.list_models()
    except:
        # Fallback to command line
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
            
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            
            return models
        except:
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
