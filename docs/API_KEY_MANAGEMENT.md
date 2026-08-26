# 🔑 Secure API Key Management

## Overview

LLM Tool has a built-in secure API key management system that allows you to:

- **Store keys in encrypted format** on your machine
- **Automatically reuse** saved keys
- **Save your preferred models** for each provider
- **Avoid re-entering** your keys every time

## 🔐 Security

### Encryption

API keys are encrypted using the `cryptography` library with Fernet algorithm (AES-128 in CBC mode).

- A unique master key is automatically generated on first use
- This master key is stored in the config directory, with restricted permissions
- Encrypted API keys are stored alongside it in `api_keys.enc`

The config directory is:

| Platform | Location |
|----------|----------|
| **Windows** | `%USERPROFILE%\.llm_tool\` (e.g. `C:\Users\you\.llm_tool\`) |
| **macOS / Linux** | `~/.llm_tool/` |

### File Permissions

On **macOS and Linux** the system sets POSIX modes:
- The config directory is `0700` (accessible only by you)
- The master key and the encrypted keys file are `0600` (read/write only by you)

On **Windows** those mode bits do not apply — NTFS uses ACLs, and the files
inherit the permissions of your user profile directory. In practice that already
restricts them to your account and to local administrators. If your machine has
several user accounts and you want to be explicit, tighten the folder from an
elevated PowerShell:

```powershell
icacls "$env:USERPROFILE\.llm_tool" /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F"
```

Either way, the keys are **encrypted at rest**, so the file permissions are a
second line of defence rather than the only one.

### Without cryptography library

If you don't have `cryptography` installed, the system will still work but:
- ⚠️ Keys will be stored in **plain text** (not recommended)
- A warning will be displayed on each save
- It is strongly recommended to install cryptography: `pip install cryptography`

## 📖 Usage

### Via the Provider Center

**Mode 6 → Resume Center → LLM providers** is the single place to manage cloud
credentials. For every registered provider it shows:

| Column | Meaning |
|--------|---------|
| SDK | whether the provider's Python package is installed, and the `pip` command if not |
| API key | `set via <VAR>` (environment), `stored encrypted`, or `not set` |
| Models | how many models the catalogue offers |
| Status | `ready`, or exactly what is missing |

From there you can store a key (encrypted, after a connectivity test), re-test an
existing one, or delete a stored key. An environment variable always takes
precedence over the encrypted store, and the screen says so when both are set.

Providers come from `llm_tool/config/providers.py`; registering one there makes
it appear in this screen and in the model pickers automatically.

### Via the model picker

When you select an API model (OpenAI, Anthropic, Google), the system:

1. **Checks** if a key already exists for this provider
2. If yes, **offers to use it** automatically
3. If no, **asks for the key** and offers to save it

Example:
```
✓ Selected LLM: gpt-4o

🔑 API Key Required for openai
Your key will be stored securely using encryption
API Key: ********
Save this API key for future use? [Y/n]: y
✓ API key saved securely
```

### Via Python Code

```python
from llm_tool.config.settings import Settings

settings = Settings()

# Save an API key
settings.set_api_key('openai', 'sk-...', model_name='gpt-4o')

# Retrieve an API key
api_key = settings.get_api_key('openai')

# Get a key (or prompt user if not available)
api_key = settings.get_or_prompt_api_key('openai', model_name='gpt-4o')

# List providers with saved keys
providers = settings.list_saved_providers()
print(f"Saved keys for: {providers}")
```

### Direct Key Manager Usage

```python
from llm_tool.config.api_key_manager import get_key_manager

key_manager = get_key_manager()

# Save a key with preferred model
key_manager.save_key('openai', 'sk-...', model_name='gpt-4o')

# Retrieve a key
api_key = key_manager.get_key('openai')

# Retrieve preferred model
model = key_manager.get_model_name('openai')

# Check if a key exists
if key_manager.has_key('openai'):
    print("OpenAI key found!")

# Delete a key
key_manager.delete_key('openai')

# List all providers
providers = key_manager.list_providers()
```

## 🗂️ File Structure

```
~/.llm_tool/                     (Windows: %USERPROFILE%\.llm_tool\)
├── .master_key           # Master encryption key
├── api_keys.enc          # Encrypted API keys
├── config.json           # General configuration
└── key_config.json       # Metadata (no sensitive keys)
```

### Stored Keys Format

The `api_keys.enc` file contains a structured JSON:

```json
{
  "openai": {
    "api_key": "base64_encoded_encrypted_key",
    "model_name": "gpt-4o",
    "encrypted": true
  },
  "anthropic": {
    "api_key": "base64_encoded_encrypted_key",
    "model_name": "claude-3-5-sonnet-20241022",
    "encrypted": true
  }
}
```

## 🔄 Key Search Priority

The system searches for API keys in this order:

1. **Environment variables** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
2. **Encrypted storage** (`~/.llm_tool/api_keys.enc`)
3. **Legacy configuration file** (`~/.llm_tool/config.json`)

This allows you to:
- Use environment variables for CI/CD environments
- Have a fallback to encrypted storage for local use
- Maintain compatibility with older configurations

## 🔧 Maintenance

### Regenerate Master Key

If you want to regenerate the encryption key:

```bash
rm ~/.llm_tool/.master_key ~/.llm_tool/api_keys.enc          # macOS / Linux
```
```powershell
Remove-Item "$env:USERPROFILE\.llm_tool\.master_key", "$env:USERPROFILE\.llm_tool\api_keys.enc"
```

On next use, a new master key will be generated and you'll need to re-enter your API keys.

### Export Configuration (without keys)

```python
from llm_tool.config.api_key_manager import get_key_manager

key_manager = get_key_manager()
config = key_manager.export_config()

# Returns model preferences without sensitive keys
print(config)
# {
#   "openai": {
#     "model_name": "gpt-4o",
#     "has_key": true,
#     "encrypted": true
#   }
# }
```

### Transfer to Another Machine

To transfer your keys to another machine:

1. **Copy the files** (securely!):
   ```bash
   scp ~/.llm_tool/.master_key  other-machine:~/.llm_tool/     # macOS / Linux
   scp ~/.llm_tool/api_keys.enc other-machine:~/.llm_tool/
   ```
   ```powershell
   # Windows (OpenSSH client ships with Windows 10/11)
   scp "$env:USERPROFILE\.llm_tool\.master_key"  user@host:.llm_tool/
   scp "$env:USERPROFILE\.llm_tool\api_keys.enc" user@host:.llm_tool/
   ```
   Note the quoting: an unquoted `C:\Users\...` path makes `scp` read `C` as a
   hostname. Copying to a Windows machine works the same way in reverse.

2. Ensure **correct permissions** (macOS / Linux only — see File Permissions above):
   ```bash
   chmod 700 ~/.llm_tool
   chmod 600 ~/.llm_tool/.master_key
   chmod 600 ~/.llm_tool/api_keys.enc
   ```

## 🛡️ Security Best Practices

1. ✅ **Install cryptography**: `pip install cryptography`
2. ✅ **Never share** your master key or API keys
3. ✅ **Backup** your configuration files regularly (securely)
4. ✅ **Use environment variables** for production servers
5. ✅ **Revoke and rotate** your API keys regularly
6. ❌ **Never include** key files in Git
7. ❌ **Don't send** keys via email or unencrypted messages

## 🆘 Troubleshooting

### "cryptography library not installed"

```bash
pip install cryptography
```

### "Permission denied" when reading keys

```bash
chmod 600 ~/.llm_tool/.master_key ~/.llm_tool/api_keys.enc    # macOS / Linux
```
On Windows, this usually means another process holds the file open, or the
folder sits in a OneDrive-synced profile. Close other LLM Tool windows and retry.

### Keys are not being saved

Check that the directory exists and is writable:
```bash
ls -la ~/.llm_tool/                                            # macOS / Linux
```
```powershell
Get-ChildItem -Force "$env:USERPROFILE\.llm_tool"             # Windows
```

If the directory doesn't exist:
```bash
mkdir -p ~/.llm_tool && chmod 700 ~/.llm_tool                  # macOS / Linux
```
```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.llm_tool"
```

### Complete Reset

To reset everything:
```bash
rm -rf ~/.llm_tool                                             # macOS / Linux
```
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.llm_tool"      # Windows
```

Next time you use LLM Tool, everything will be recreated automatically.

## 📚 Supported Providers

The system currently supports:

- **OpenAI** (GPT-3.5, GPT-4, GPT-4o, o1, o3-mini, GPT-5, etc.)
- **Anthropic** (Claude 3 Opus, Sonnet, Haiku, etc.)
- **Google Gemini** (Gemini 3.x Flash / Pro) — free key at https://aistudio.google.com/apikey
- **Google** (Gemini Pro, etc.)
- **HuggingFace** (models via API)

Keys are automatically associated with standard environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `HF_TOKEN`
- `OLLAMA_API_KEY` (only needed for a remote or cloud Ollama endpoint)

### Setting them in your shell

```bash
export OPENAI_API_KEY="sk-..."             # macOS / Linux, this shell only
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc    # persistent
```
```powershell
$env:OPENAI_API_KEY = "sk-..."             # Windows PowerShell, this shell only
setx OPENAI_API_KEY "sk-..."               # persistent — applies to NEW terminals
```
```bat
set OPENAI_API_KEY=sk-...                  :: Windows Command Prompt, this shell only
```

`export` is the macOS/Linux form and does nothing in PowerShell. Note that
`setx` does **not** affect the terminal you run it in — open a new one.
