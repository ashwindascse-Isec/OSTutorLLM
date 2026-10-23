import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
LIBRARY_DIR = DATA_DIR / "library"
PROGRESS_FILE = DATA_DIR / "progress_records.json"
MANIFEST_FILE = DATA_DIR / "library_manifest.json"

for d in [DATA_DIR, LIBRARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_environment() -> Dict[str, str]:
    """
    Robustly loads environment variables from .env supporting both standard
    KEY=VALUE format and custom 'groq api key: value' formats.
    """
    keys = {}
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Match KEY=VALUE
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    keys[k] = v
                # Match colon separated: "groq api key: xxx"
                elif ":" in line:
                    parts = line.split(":", 1)
                    k_raw, v = parts[0].strip().lower(), parts[1].strip().strip("'\"")
                    if "groq" in k_raw:
                        keys["GROQ_API_KEY"] = v
                    elif "gemini" in k_raw or "google" in k_raw:
                        keys["GEMINI_API_KEY"] = v
                        keys["GOOGLE_API_KEY"] = v

    for k, v in keys.items():
        if k not in os.environ or not os.environ[k]:
            os.environ[k] = v

    return keys

# Run loader on import
_LOADED_KEYS = load_environment()

def get_groq_key() -> Optional[str]:
    return os.environ.get("GROQ_API_KEY")

def get_gemini_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

def get_masked_key(key: Optional[str]) -> str:
    if not key:
        return "Not Set"
    if len(key) <= 8:
        return "****"
    return key[:4] + "..." + key[-4:]

# Default generation models
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
FALLBACK_GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
FALLBACK_GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.5-flash-lite"]

DEFAULT_PROVIDER = "Groq" if get_groq_key() else ("Gemini" if get_gemini_key() else "None")
