from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal


RunState = Literal["queued", "running", "completed", "failed"]


@dataclass
class RunRecord:
    run_id: str
    status: RunState
    created_at: float
    updated_at: float
    config_yaml: str
    max_pages: int | None = None
    message: str | None = None
    progress: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    results: list[dict[str, Any]] | None = None
    keyword_expansions: dict[str, list[str]] | None = None
    expansion_trace: dict[str, Any] | None = None


class RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}

    def create(self, config_yaml: str, *, max_pages: int | None = None) -> RunRecord:
        run_id = str(uuid.uuid4())
        now = time.time()
        rec = RunRecord(
            run_id=run_id,
            status="queued",
            created_at=now,
            updated_at=now,
            config_yaml=config_yaml,
            max_pages=max_pages,
            progress={},
        )
        with self._lock:
            self._runs[run_id] = rec
        return rec

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def update(self, run_id: str, **kwargs: Any) -> RunRecord | None:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                return None
            for k, v in kwargs.items():
                setattr(rec, k, v)
            rec.updated_at = time.time()
            return rec


run_store = RunStore()
