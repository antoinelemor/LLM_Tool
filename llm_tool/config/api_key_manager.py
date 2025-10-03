#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
api_key_manager.py

MAIN OBJECTIVE:
---------------
This script provides secure storage and management of API keys using encryption.
Keys are stored encrypted and can be automatically loaded when needed.

Dependencies:
-------------
- cryptography
- json
- pathlib
- os
- getpass

MAIN FEATURES:
--------------
1) Secure encryption/decryption of API keys
2) Persistent storage in user's home directory
3) Multiple provider support (OpenAI, Anthropic, Google)
4) Automatic key retrieval
5) Key validation and testing

Author:
-------
Antoine Lemor
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from getpass import getpass

# Try to import cryptography for secure key storage
try:
    from cryptography.fernet import Fernet
    import base64
    HAS_CRYPTOGRAPHY = True
except ImportError as e:
    HAS_CRYPTOGRAPHY = False
    logging.warning(f"cryptography library not installed ({e}). API keys will be stored in plain text (NOT RECOMMENDED). Install with: pip install cryptography")
except Exception as e:
    HAS_CRYPTOGRAPHY = False
    logging.error(f"Error importing cryptography: {e}")


class APIKeyManager:
    """Secure API key storage and management"""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the API key manager.

        Parameters
        ----------
        config_dir : Path, optional
            Directory for configuration files. Defaults to ~/.llm_tool
        """
        self.config_dir = config_dir or Path.home() / ".llm_tool"
        self.config_dir.mkdir(exist_ok=True, mode=0o700)  # Secure permissions

        self.keys_file = self.config_dir / "api_keys.enc"
        self.config_file = self.config_dir / "key_config.json"
        self.master_key_file = self.config_dir / ".master_key"

        self.logger = logging.getLogger(__name__)
        self._cipher = None
        self._keys = {}

        # Load existing keys if available
        self._initialize_encryption()
        self.load_keys()

    def _initialize_encryption(self):
        """Initialize encryption system"""
        if not HAS_CRYPTOGRAPHY:
            self.logger.warning("Encryption not available. Keys will be stored in plain text.")
            return

        # Check if master key exists
        if self.master_key_file.exists():
            try:
                with open(self.master_key_file, 'rb') as f:
                    key = f.read()
                self._cipher = Fernet(key)
                self.logger.debug("Loaded existing master key")
            except Exception as e:
                self.logger.error(f"Failed to load master key: {e}")
                self._generate_master_key()
        else:
            self._generate_master_key()

    def _generate_master_key(self):
        """Generate a new master encryption key"""
        if not HAS_CRYPTOGRAPHY:
            return

        try:
            # Generate a secure key
            key = Fernet.generate_key()
            self._cipher = Fernet(key)

            # Save it securely
            self.master_key_file.touch(mode=0o600)  # Secure file permissions
            with open(self.master_key_file, 'wb') as f:
                f.write(key)

            self.logger.info("Generated new master encryption key")
        except Exception as e:
            self.logger.error(f"Failed to generate master key: {e}")

    def _encrypt(self, data: str) -> str:
        """Encrypt data"""
        if not HAS_CRYPTOGRAPHY or self._cipher is None:
            return data  # Return plain text if encryption not available

        try:
            encrypted = self._cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data

    def _decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        if not HAS_CRYPTOGRAPHY or self._cipher is None:
            return encrypted_data  # Return as-is if encryption not available

        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self._cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return encrypted_data

    def save_key(self, provider: str, api_key: str, model_name: Optional[str] = None):
        """
        Save an API key securely.

        Parameters
        ----------
        provider : str
            Provider name (openai, anthropic, google, etc.)
        api_key : str
            The API key to save
        model_name : str, optional
            Preferred model name for this provider
        """
        provider = provider.lower()

        # Store key info
        self._keys[provider] = {
            'api_key': self._encrypt(api_key),
            'model_name': model_name,
            'encrypted': HAS_CRYPTOGRAPHY
        }

        # Save to disk
        self._save_keys_to_disk()

        self.logger.info(f"API key for {provider} saved {'securely' if HAS_CRYPTOGRAPHY else 'in plain text'}")

    def get_key(self, provider: str) -> Optional[str]:
        """
        Get an API key for a provider.

        Parameters
        ----------
        provider : str
            Provider name

        Returns
        -------
        str or None
            The API key if available, None otherwise
        """
        provider = provider.lower()

        # First check environment variables
        env_var_names = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'huggingface': 'HF_TOKEN'
        }

        env_var = env_var_names.get(provider)
        if env_var and env_var in os.environ:
            return os.environ[env_var]

        # Check stored keys
        if provider in self._keys:
            encrypted_key = self._keys[provider]['api_key']
            return self._decrypt(encrypted_key)

        return None

    def get_model_name(self, provider: str) -> Optional[str]:
        """
        Get the preferred model name for a provider.

        Parameters
        ----------
        provider : str
            Provider name

        Returns
        -------
        str or None
            The model name if saved, None otherwise
        """
        provider = provider.lower()
        if provider in self._keys:
            return self._keys[provider].get('model_name')
        return None

    def has_key(self, provider: str) -> bool:
        """
        Check if a key exists for a provider.

        Parameters
        ----------
        provider : str
            Provider name

        Returns
        -------
        bool
            True if key exists, False otherwise
        """
        return self.get_key(provider) is not None

    def delete_key(self, provider: str):
        """
        Delete an API key.

        Parameters
        ----------
        provider : str
            Provider name
        """
        provider = provider.lower()
        if provider in self._keys:
            del self._keys[provider]
            self._save_keys_to_disk()
            self.logger.info(f"API key for {provider} deleted")

    def list_providers(self) -> list:
        """
        List all providers with saved keys.

        Returns
        -------
        list
            List of provider names
        """
        # Include both stored and environment keys
        providers = set(self._keys.keys())

        # Check environment variables
        env_vars = {
            'OPENAI_API_KEY': 'openai',
            'ANTHROPIC_API_KEY': 'anthropic',
            'GOOGLE_API_KEY': 'google',
            'HF_TOKEN': 'huggingface'
        }

        for env_var, provider in env_vars.items():
            if env_var in os.environ:
                providers.add(provider)

        return sorted(list(providers))

    def load_keys(self):
        """Load saved keys from disk"""
        if not self.keys_file.exists():
            return

        try:
            with open(self.keys_file, 'r') as f:
                self._keys = json.load(f)
            self.logger.debug(f"Loaded {len(self._keys)} API keys from disk")
        except Exception as e:
            self.logger.error(f"Failed to load API keys: {e}")
            self._keys = {}

    def _save_keys_to_disk(self):
        """Save keys to disk"""
        try:
            # Ensure secure permissions
            self.keys_file.touch(mode=0o600)

            with open(self.keys_file, 'w') as f:
                json.dump(self._keys, f, indent=2)

            self.logger.debug(f"Saved {len(self._keys)} API keys to disk")
        except Exception as e:
            self.logger.error(f"Failed to save API keys: {e}")

    def prompt_and_save_key(self, provider: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Prompt user for an API key and save it.

        Parameters
        ----------
        provider : str
            Provider name
        model_name : str, optional
            Preferred model name

        Returns
        -------
        str or None
            The API key entered by user
        """
        try:
            print(f"\n🔑 Enter API key for {provider}")
            if HAS_CRYPTOGRAPHY:
                print("   (Will be stored securely using encryption)")
            else:
                print("   ⚠️  WARNING: Will be stored in plain text. Install 'cryptography' for secure storage.")

            api_key = getpass(f"{provider.upper()} API Key: ")

            if api_key:
                self.save_key(provider, api_key, model_name)
                return api_key

            return None
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled")
            return None

    def get_or_prompt_key(self, provider: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Get API key from storage or prompt user if not found.

        Parameters
        ----------
        provider : str
            Provider name
        model_name : str, optional
            Preferred model name

        Returns
        -------
        str or None
            The API key
        """
        # Try to get existing key
        api_key = self.get_key(provider)
        if api_key:
            return api_key

        # Prompt for new key
        return self.prompt_and_save_key(provider, model_name)

    def export_config(self) -> Dict[str, Any]:
        """
        Export configuration (without sensitive keys).

        Returns
        -------
        dict
            Configuration with model preferences but no API keys
        """
        config = {}
        for provider, data in self._keys.items():
            config[provider] = {
                'model_name': data.get('model_name'),
                'has_key': True,
                'encrypted': data.get('encrypted', False)
            }
        return config

    def import_config(self, config: Dict[str, Any]):
        """
        Import configuration (model preferences only, not keys).

        Parameters
        ----------
        config : dict
            Configuration to import
        """
        for provider, data in config.items():
            if provider in self._keys and 'model_name' in data:
                self._keys[provider]['model_name'] = data['model_name']

        self._save_keys_to_disk()


# Global instance
_key_manager = None


def get_key_manager() -> APIKeyManager:
    """Get global API key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager
