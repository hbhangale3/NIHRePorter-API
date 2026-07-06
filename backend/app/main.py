from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .concept_suggester import ConceptSuggester
from .csv_export import rows_to_csv_bytes
from .models import RunRequest, RunStatus
from .run_store import run_store
from .runner import run_pipeline_async
from .keyword_suggester import KeywordSuggester
from pydantic import BaseModel


app = FastAPI(title="NIH RePORTER Outreach List Builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _execute_run(run_id: str) -> None:
    rec = run_store.get(run_id)
    if rec is None:
        return

    run_store.update(run_id, status="running", message=None, progress={"stage": "querying"})
    try:
        results, summary, keyword_expansions, expansion_trace = await run_pipeline_async(
            rec.config_yaml,
            max_pages=rec.max_pages,
        )
        run_store.update(
            run_id,
            status="completed",
            results=results,
            summary=summary,
            keyword_expansions=keyword_expansions,
            expansion_trace=expansion_trace,
            progress={"stage": "completed"},
        )
    except Exception as e:
        run_store.update(run_id, status="failed", message=str(e), progress={"stage": "failed"})


@app.post("/api/runs", response_model=RunStatus)
async def create_run(req: RunRequest, background_tasks: BackgroundTasks) -> RunStatus:
    rec = run_store.create(req.config_yaml, max_pages=req.max_pages)
    background_tasks.add_task(_execute_run, rec.run_id)
    return RunStatus(
        run_id=rec.run_id, 
        status=rec.status, 
        message=rec.message, 
        progress=rec.progress or {},
        keyword_expansions=None,
        expansion_trace=None,
    )


@app.get("/api/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str) -> RunStatus:
    rec = run_store.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunStatus(
        run_id=rec.run_id, 
        status=rec.status, 
        message=rec.message, 
        progress=rec.progress or {},
        keyword_expansions=rec.keyword_expansions,
        expansion_trace=rec.expansion_trace,
    )


@app.get("/api/runs/{run_id}/results")
def get_results(run_id: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    rec = run_store.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    if rec.status != "completed" or rec.results is None:
        raise HTTPException(status_code=409, detail="results not available")

    total = len(rec.results)
    items = rec.results[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": items, "summary": rec.summary or {}}


@app.get("/api/runs/{run_id}/export.csv")
def export_csv(run_id: str) -> Response:
    rec = run_store.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    if rec.status != "completed" or rec.results is None:
        raise HTTPException(status_code=409, detail="results not available")

    from .models import PIOutreachRow

    rows = [PIOutreachRow.model_validate(r) for r in rec.results]
    csv_bytes = rows_to_csv_bytes(rows)

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=outreach_{run_id}.csv"},
    )


class KeywordSuggestionRequest(BaseModel):
    topic_description: str
    openai_api_key: str | None = None
    model: str = "gpt-4o-mini"
    max_broad_keywords: int = 3
    max_topic_terms: int = 10
    context: str = "biomedical research and health disparities"


@app.post("/api/suggest-keywords")
def suggest_keywords(req: KeywordSuggestionRequest) -> dict[str, Any]:
    suggester = KeywordSuggester(api_key=req.openai_api_key, model=req.model)
    config = suggester.suggest_config(
        topic_description=req.topic_description,
        context=req.context,
        max_broad_keywords=req.max_broad_keywords,
        max_topic_terms=req.max_topic_terms
    )
    return config


class ConceptSuggestionRequest(BaseModel):
    question: str
    top_k: int = 8


@app.post("/api/concepts/suggest")
def suggest_concepts(req: ConceptSuggestionRequest) -> dict[str, Any]:
    suggester = ConceptSuggester()
    return suggester.suggest(req.question, top_k=req.top_k)
