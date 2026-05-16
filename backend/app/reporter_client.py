from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .cache import cache_key, make_cache
from .settings import settings


class ReporterClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=settings.reporter_api_base_url, timeout=60.0)
        self._cache = make_cache()
        self._lock = asyncio.Lock()
        self._last_request_monotonic: float | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _rate_limit(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            if self._last_request_monotonic is None:
                self._last_request_monotonic = now
                return
            elapsed = now - self._last_request_monotonic
            to_sleep = settings.reporter_rate_limit_seconds - elapsed
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
            self._last_request_monotonic = asyncio.get_running_loop().time()

    async def post_json_cached(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = cache_key(f"POST:{path}", payload)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        await self._rate_limit()
        resp = await self._client.post(path, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._cache.set(key, data, expire=settings.cache_ttl_seconds)
        return data

    async def search_projects(
        self,
        criteria: dict[str, Any],
        *,
        include_fields: list[str] | None = None,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "criteria": criteria,
            "offset": offset,
            "limit": limit,
        }
        if include_fields:
            payload["include_fields"] = include_fields

        return await self.post_json_cached("/projects/search", payload)

    async def fetch_all_projects(
        self,
        criteria: dict[str, Any],
        *,
        include_fields: list[str] | None = None,
        page_size: int = 500,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        offset = 0
        page = 0
        all_projects: list[dict[str, Any]] = []

        while True:
            if max_pages is not None and page >= max_pages:
                break

            data = await self.search_projects(
                criteria,
                include_fields=include_fields,
                offset=offset,
                limit=page_size,
            )

            projects = data.get("results") or []
            if not isinstance(projects, list):
                break

            all_projects.extend(projects)

            total = data.get("meta", {}).get("total")
            if total is None:
                # fallback: stop when page is short
                if len(projects) < page_size:
                    break
            else:
                if len(all_projects) >= int(total):
                    break

            if not projects:
                break

            offset += page_size
            page += 1

        return all_projects
