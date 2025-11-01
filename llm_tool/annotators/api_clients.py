#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
api_clients.py

MAIN OBJECTIVE:
---------------
This script provides client implementations for various LLM API providers
including OpenAI, Anthropic, and Google, with comprehensive error handling
and retry mechanisms.

Dependencies:
-------------
- sys
- openai
- anthropic
- google.generativeai
- logging
- typing
- time

MAIN FEATURES:
--------------
1) OpenAI API client with GPT-4, o1, o3 support
2) Anthropic Claude API client
3) Google Gemini API client
4) Automatic retry mechanisms
5) Error handling and logging

Author:
-------
Antoine Lemor
"""

import logging
import time
import json
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# OpenAI import for ChatGPT API usage (>=1.0.0)
try:
    from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, APIError
    HAS_NEW_OPENAI = True
except ImportError:
    HAS_NEW_OPENAI = False
    OpenAI = None

# Anthropic import
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

# Google Generative AI import
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False
    genai = None


class BaseAPIClient(ABC):
    """Base class for API clients"""

    def __init__(self, api_key: str, **kwargs):
        """Initialize the API client"""
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout = kwargs.get('timeout', 60)
        self.progress_manager = kwargs.get('progress_manager', None)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response from the API"""
        pass


class OpenAIClient(BaseAPIClient):
    """OpenAI API client implementation"""

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenAI client"""
        super().__init__(api_key, **kwargs)

        # If we have a progress manager, disable console logging to avoid conflicts
        if self.progress_manager:
            # Remove all console handlers to prevent duplicate output
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    self.logger.removeHandler(handler)
            # Also prevent propagation to root logger
            self.logger.propagate = False

        if not HAS_NEW_OPENAI:
            raise ImportError("OpenAI library not installed. Install with: pip install openai>=1.0.0")

        self.client = OpenAI(
            api_key=api_key,
            max_retries=self.max_retries,
            timeout=self.timeout
        )
        self.model = kwargs.get('model', 'gpt-3.5-turbo')

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response from OpenAI API.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model
        model : str, optional
            Model to use (overrides default)
        temperature : float
            Temperature parameter (0-2)
        max_tokens : int
            Maximum tokens in response
        response_format : dict, optional
            Response format specification

        Returns
        -------
        str or None
            Generated response or None on error
        """
        model = model or self.model
        model_name_lower = model.lower()

        # Detect model type for parameter adaptation
        # o-series models: o1, o1-, o3-, o4- (reasoning models with fixed parameters)
        is_o_series = (
            model_name_lower == 'o1' or
            model_name_lower.startswith('o1-') or
            model_name_lower.startswith('o3-') or
            model_name_lower.startswith('o4-')
        )
        # Recent models with different parameter handling (2025+ models, GPT-5 variants)
        is_2025_model = (
            any(x in model_name_lower for x in ['2025', 'gpt-5', 'gpt5'])
            and not model_name_lower.startswith('gpt-4.1')
        )

        try:
            # Build base arguments
            common_kw = {
                'model': model,
                'messages': [{"role": "user", "content": prompt}],
            }

            # Use max_completion_tokens for recent models, max_tokens for older
            if is_o_series or is_2025_model:
                common_kw["max_completion_tokens"] = max_tokens
            else:
                common_kw["max_tokens"] = max_tokens

            # Add temperature and top_p based on model type
            if is_o_series:
                # o1/o3 models: fixed parameters (reasoning models)
                common_kw["temperature"] = 1.0
                common_kw["top_p"] = 1.0
                self.logger.info(f"o-series model detected ({model}): temperature and top_p fixed to 1.0")
            elif is_2025_model:
                # 2025/gpt-5 models: use default parameters (temperature=1.0)
                # These models often have restricted parameter ranges
                common_kw["temperature"] = 1.0
                common_kw["top_p"] = 1.0
                self.logger.info(f"Recent model detected ({model}): using default parameters (temperature=1.0, top_p=1.0)")
            else:
                # Classic models: allow custom parameters
                common_kw["temperature"] = temperature
                common_kw["top_p"] = kwargs.get('top_p', 1.0)

            # Add response format if specified
            if response_format:
                common_kw["response_format"] = response_format
            elif kwargs.get('json_mode', False):
                common_kw["response_format"] = {"type": "json_object"}

            # Make API call
            completion = self.client.chat.completions.create(**common_kw)
            raw = completion.choices[0].message.content

            # Check if model returned None or empty response
            if raw is None or not raw.strip():
                warning_msg = f"Model '{model}' returned empty response, retrying..."

                # Only show via progress manager if available, otherwise log
                if self.progress_manager:
                    self.progress_manager.show_warning(warning_msg)
                else:
                    self.logger.warning(warning_msg)

                # Retry with more explicit prompt
                enhanced_prompt = prompt + "\n\nIMPORTANT: You MUST provide a complete response. Do not return an empty response."

                retry_kw = common_kw.copy()
                retry_kw['messages'] = [{"role": "user", "content": enhanced_prompt}]

                retry_completion = self.client.chat.completions.create(**retry_kw)
                raw = retry_completion.choices[0].message.content

                # If still empty after retry, return None
                if raw is None or not raw.strip():
                    error_msg = f"Model '{model}' returned empty response even after retry"

                    # Only show via progress manager if available, otherwise log
                    if self.progress_manager:
                        self.progress_manager.show_error(error_msg)
                    else:
                        self.logger.error(error_msg)
                    return None

            raw = raw.strip()

            # Check for empty JSON response
            if response_format and response_format.get("type") == "json_object":
                try:
                    parsed_json = json.loads(raw)
                    if not parsed_json or parsed_json == {}:
                        self.logger.warning("OpenAI returned empty JSON, retrying with explicit prompt...")

                        # Retry with more explicit prompt
                        enhanced_prompt = prompt + "\n\nIMPORTANT: You MUST provide values for ALL applicable fields. If a field is not applicable, use null as the value, but include the field in your response."

                        retry_kw = common_kw.copy()
                        retry_kw['messages'] = [{"role": "user", "content": enhanced_prompt}]

                        retry_completion = self.client.chat.completions.create(**retry_kw)
                        raw = retry_completion.choices[0].message.content.strip()
                except json.JSONDecodeError:
                    pass  # Continue with normal validation

            return raw

        except (APIConnectionError, APIStatusError, APITimeoutError, APIError) as e:
            self.logger.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error calling OpenAI: {e}")
            return None


class AnthropicClient(BaseAPIClient):
    """Anthropic Claude API client implementation"""

    def __init__(self, api_key: str, **kwargs):
        """Initialize Anthropic client"""
        super().__init__(api_key, **kwargs)

        if not HAS_ANTHROPIC:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = kwargs.get('model', 'claude-3-sonnet-20240229')

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response from Anthropic API.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model
        model : str, optional
            Model to use (overrides default)
        temperature : float
            Temperature parameter
        max_tokens : int
            Maximum tokens in response

        Returns
        -------
        str or None
            Generated response or None on error
        """
        model = model or self.model

        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract text from response
            if hasattr(message, 'content'):
                if isinstance(message.content, list) and len(message.content) > 0:
                    return message.content[0].text
                elif isinstance(message.content, str):
                    return message.content

            return None

        except anthropic.APIConnectionError as e:
            self.logger.error(f"Anthropic connection error: {e}")
            return None
        except anthropic.RateLimitError as e:
            self.logger.error(f"Anthropic rate limit error: {e}")
            return None
        except anthropic.APIStatusError as e:
            self.logger.error(f"Anthropic API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error calling Anthropic: {e}")
            return None


class GoogleClient(BaseAPIClient):
    """Google Generative AI client implementation"""

    def __init__(self, api_key: str, **kwargs):
        """Initialize Google client"""
        super().__init__(api_key, **kwargs)

        if not HAS_GOOGLE:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")

        genai.configure(api_key=api_key)
        self.model_name = kwargs.get('model', 'gemini-pro')
        self.model = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response from Google Gemini API.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model
        temperature : float
            Temperature parameter
        max_tokens : int
            Maximum tokens in response

        Returns
        -------
        str or None
            Generated response or None on error
        """
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 40)
            )

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Extract text from response
            if response.text:
                return response.text

            # Check for safety filters
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                return part.text

            self.logger.warning("No text found in Google response")
            return None

        except Exception as e:
            self.logger.error(f"Error calling Google Gemini: {e}")
            return None


def create_api_client(provider: str, api_key: str, **kwargs) -> BaseAPIClient:
    """
    Factory function to create appropriate API client.

    Parameters
    ----------
    provider : str
        API provider ('openai', 'anthropic', 'google')
    api_key : str
        API key for the provider
    **kwargs
        Additional configuration options

    Returns
    -------
    BaseAPIClient
        Appropriate API client instance

    Raises
    ------
    ValueError
        If provider is not supported
    """
    provider = provider.lower()

    if provider == 'openai':
        return OpenAIClient(api_key, **kwargs)
    elif provider == 'anthropic':
        return AnthropicClient(api_key, **kwargs)
    elif provider == 'google':
        return GoogleClient(api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported API provider: {provider}")
