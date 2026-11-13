"""Optional runtime configuration & `.env` resolver for `st_ds`.

The library NEVER requires a `.env` file. If a `.env` file is present in the
working directory or repository root (or passed via `STDS_ENV_FILE`), its keys
serve as fallbacks under the strict precedence rule:

    explicit parameter > OS environment variable > `.env` file > built-in defaults (`tr`)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LangCode = Literal["tr", "en"]
SUPPORTED_LANGS: tuple[LangCode, ...] = ("tr", "en")

_LANG_ALIASES: dict[str, LangCode] = {
    "tr": "tr",
    "tr-tr": "tr",
    "tr_tr": "tr",
    "tur": "tr",
    "turkish": "tr",
    "türkçe": "tr",
    "en": "en",
    "en-us": "en",
    "en_us": "en",
    "en-gb": "en",
    "en_gb": "en",
    "eng": "en",
    "english": "en",
}

_TRUTHY = {"1", "true", "yes", "y", "on", "evet"}
_FALSY = {"0", "false", "no", "n", "off", "hayir", "hayır"}


def normalize_lang(lang: str | None) -> LangCode:
    """Normalize a language string into `'tr'` or `'en'`, raising on invalid values."""
    if lang is None or not str(lang).strip():
        return "tr"
    key = str(lang).strip().lower()
    if key in _LANG_ALIASES:
        return _LANG_ALIASES[key]
    raise ValueError(
        f"Unsupported language '{lang}'. Supported values: 'tr' (Turkish) or 'en' (English)."
    )


def _find_env_file(env_file: str | Path | None = None) -> Path | None:
    if env_file is not None:
        candidate = Path(env_file)
        return candidate if candidate.is_file() else None

    explicit_env = os.environ.get("STDS_ENV_FILE")
    if explicit_env:
        candidate = Path(explicit_env)
        return candidate if candidate.is_file() else None

    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return cwd_env

    pkg_env = Path(__file__).resolve().parents[1] / ".env"
    if pkg_env.is_file():
        return pkg_env
    return None


def _strip_unquoted_comment(raw_val: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(raw_val):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or raw_val[idx - 1].isspace():
                return raw_val[:idx].strip()
    return raw_val.strip()


def load_dotenv_dict(env_file: str | Path | None = None) -> tuple[dict[str, str], Path | None]:
    """Parse an optional `.env` file without mutating `os.environ` or failing if missing."""
    path = _find_env_file(env_file)
    if path is None:
        return {}, None

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        cleaned = _strip_unquoted_comment(raw_val)
        if len(cleaned) >= 2 and (
            (cleaned[0] == '"' and cleaned[-1] == '"')
            or (cleaned[0] == "'" and cleaned[-1] == "'")
        ):
            cleaned = cleaned[1:-1]
        values[key] = cleaned
    return values, path


def _read_setting(
    key: str,
    dotenv_map: dict[str, str],
    default: str,
) -> str:
    env_val = os.environ.get(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    dot_val = dotenv_map.get(key)
    if dot_val is not None and dot_val.strip():
        return dot_val.strip()
    return default


def _parse_bool(raw: str, *, default: bool = True) -> bool:
    low = raw.strip().lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    return default


def resolve_lang(
    explicit_lang: str | None = None,
    *,
    env_file: str | Path | None = None,
) -> LangCode:
    """Resolve effective language: explicit arg > OS env > optional `.env` > `'tr'`."""
    if explicit_lang is not None and str(explicit_lang).strip():
        return normalize_lang(explicit_lang)
    dotenv_map, _ = load_dotenv_dict(env_file)
    raw = _read_setting("STDS_LANG", dotenv_map, "tr")
    return normalize_lang(raw)


@dataclass(frozen=True)
class STDSRuntimeConfig:
    """Resolved optional runtime defaults from `.env` and process environment."""

    lang: LangCode = "tr"
    runs_dir: str = "runs"
    dpi: int = 160
    pdf_export: bool = True
    default_profile: str = "visnir_24ch_v1"
    env_file_used: str | None = None


def get_runtime_config(
    *,
    lang: str | None = None,
    env_file: str | Path | None = None,
) -> STDSRuntimeConfig:
    """Resolve `STDSRuntimeConfig` cleanly whether `.env` exists or not."""
    dotenv_map, env_path = load_dotenv_dict(env_file)
    resolved_lang = resolve_lang(lang, env_file=env_file)

    runs_dir = _read_setting("STDS_RUNS_DIR", dotenv_map, "runs")
    dpi_raw = _read_setting("STDS_DPI", dotenv_map, "160")
    try:
        dpi = max(72, int(dpi_raw))
    except ValueError:
        dpi = 160

    pdf_export = _parse_bool(_read_setting("STDS_PDF_EXPORT", dotenv_map, "true"), default=True)
    default_profile = _read_setting("STDS_DEFAULT_PROFILE", dotenv_map, "visnir_24ch_v1")

    return STDSRuntimeConfig(
        lang=resolved_lang,
        runs_dir=runs_dir,
        dpi=dpi,
        pdf_export=pdf_export,
        default_profile=default_profile,
        env_file_used=str(env_path) if env_path is not None else None,
    )
