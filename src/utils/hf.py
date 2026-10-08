from __future__ import annotations

import os
from typing import Any


def apply_hf_environment(config: dict[str, Any]) -> None:
    """
    Applies Hugging Face offline/cache-related environment variables.

    This does not download anything; it only configures behavior.
    """
    offline = bool(config.get("hf_offline", False))
    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Optional: user-provided cache dir. Prefer passing cache_dir to from_pretrained,
    # but also allow setting HF_HOME to consolidate caches across tools.
    cache_dir = config.get("hf_cache_dir")
    if cache_dir:
        os.environ.setdefault("HF_HOME", str(cache_dir))


def hf_from_pretrained_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """
    Standard kwargs for Hugging Face `from_pretrained` calls.

    Supported config keys:
      - hf_cache_dir: str | None
      - hf_local_files_only: bool | None
      - hf_offline: bool (implies local_files_only)
    """
    offline = bool(config.get("hf_offline", False))
    local_only = config.get("hf_local_files_only")
    if local_only is None:
        local_only = offline

    out: dict[str, Any] = {
        "local_files_only": bool(local_only),
    }
    cache_dir = config.get("hf_cache_dir")
    if cache_dir:
        out["cache_dir"] = str(cache_dir)
    return out


def hf_set_dtype_arg(kwargs: dict[str, Any], dtype: Any) -> dict[str, Any]:
    """
    Sets the preferred model dtype argument for `from_pretrained`.

    Newer transformers versions prefer `dtype` over `torch_dtype`.
    """
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs
