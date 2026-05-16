from __future__ import annotations

import hashlib
import json
from typing import Any

from diskcache import Cache

from .settings import settings


def make_cache() -> Cache:
    return Cache(settings.cache_dir)


def cache_key(prefix: str, payload: Any) -> str:
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(b).hexdigest()
    return f"{prefix}:{digest}"
