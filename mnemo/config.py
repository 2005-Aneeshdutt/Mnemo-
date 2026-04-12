import os
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from project root (folder that contains `mnemo/`), not only CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()


def _get_env(primary: str, legacy: str | None = None) -> str | None:
    v = os.environ.get(primary, "").strip()
    if v:
        return v
    if legacy:
        v = os.environ.get(legacy, "").strip()
        if v:
            return v
    return None


def _env_int(primary: str, default: int, legacy: str | None = None) -> int:
    raw = _get_env(primary, legacy)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(primary: str, default: float, legacy: str | None = None) -> float:
    raw = _get_env(primary, legacy)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_str(primary: str, default: str, legacy: str | None = None) -> str:
    raw = _get_env(primary, legacy)
    return raw if raw is not None else default


def _env_bool(primary: str, default: bool, legacy: str | None = None) -> bool:
    raw = _get_env(primary, legacy)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
CHAT_MODEL = _env_str("GROQ_MODEL", "llama-3.3-70b-versatile")
EXTRACT_MODEL = _env_str("GROQ_EXTRACT_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = _env_str("GROQ_EMBED_MODEL", "nomic-embed-text-v1_5")


def embedding_model_candidates() -> list[str]:
    """Primary model first, then comma-separated fallbacks from GROQ_EMBED_MODEL_FALLBACKS."""
    primary = EMBEDDING_MODEL
    extra = os.environ.get("GROQ_EMBED_MODEL_FALLBACKS", "")
    out: list[str] = [primary]
    if extra.strip():
        out.extend(x.strip() for x in extra.split(",") if x.strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"
DB_PATH = Path(_env_str("MNEMO_DB_PATH", str(DEFAULT_DB_PATH), "MEMORI_DB_PATH"))

TOP_K = _env_int("MNEMO_TOP_K", 12, "MEMORI_TOP_K")
RECENT_MESSAGES = _env_int("MNEMO_RECENT_MSG", 12, "MEMORI_RECENT_MSG")

HYBRID_W_DENSE = _env_float("MNEMO_W_DENSE", 0.5, "MEMORI_W_DENSE")
HYBRID_W_LEX = _env_float("MNEMO_W_LEX", 0.35, "MEMORI_W_LEX")
HYBRID_W_REC = _env_float("MNEMO_W_REC", 0.15, "MEMORI_W_REC")

ANN_ENABLED = _env_bool("MNEMO_ANN_ENABLED", True, "MEMORI_ANN_ENABLED")
ANN_MIN_ROWS = _env_int("MNEMO_ANN_MIN_ROWS", 32, "MEMORI_ANN_MIN_ROWS")
ANN_CANDIDATE_MULT = _env_int("MNEMO_ANN_CANDIDATE_MULT", 8, "MEMORI_ANN_CANDIDATE_MULT")
ANN_MIN_CANDIDATES = _env_int("MNEMO_ANN_MIN_CANDIDATES", 64, "MEMORI_ANN_MIN_CANDIDATES")
ANN_MAX_CANDIDATES = _env_int("MNEMO_ANN_MAX_CANDIDATES", 512, "MEMORI_ANN_MAX_CANDIDATES")

API_HOST = _env_str("MNEMO_API_HOST", "127.0.0.1", "MEMORI_API_HOST")
API_PORT = _env_int("MNEMO_API_PORT", 8765, "MEMORI_API_PORT")

# Public API auth (optional). Legacy MEMORI_API_KEY still read if MNEMO_API_KEY unset.
MNEMO_API_KEY = (_get_env("MNEMO_API_KEY", "MEMORI_API_KEY") or "").strip()

RATE_LIMIT_DEFAULT = _env_str("MNEMO_RATE_LIMIT", "120/minute", "MEMORI_RATE_LIMIT")
RATE_LIMIT_CHAT = _env_str("MNEMO_RATE_LIMIT_CHAT", "60/minute", "MEMORI_RATE_LIMIT_CHAT")

EMBEDDINGS_DISABLED = _env_bool("MNEMO_NO_EMBEDDINGS", False, "MEMORI_NO_EMBEDDINGS")

