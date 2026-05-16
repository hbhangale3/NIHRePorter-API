# NIH RePORTER PI outreach list builder

React website + FastAPI backend to build an updatable outreach list for NIH projects on technology and health disparities.

## What it does

- Queries NIH RePORTER API (backend only)
- **🤖 AI-powered keyword expansion** (optional) - automatically expands search terms with synonyms, acronyms, and variations
- Pagination handling
- Rate limiting (~1 request/sec) + disk caching
- YAML-configured two-stage filtering
  - Stage 1: broad NIH query (FY + broad keywords)
  - Stage 2: strict local topic matching per named topics
- PI + institution deduplication and aggregation
- CSV export with traceability fields

## Project structure

- `backend/` FastAPI + CLI runner
- `frontend/` React + Vite UI

## Quickstart

### 1) Backend

```
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173). The UI calls the backend at `/api/*` via proxy.

## Config format (YAML)

See `backend/config.example.yaml` for a complete example.

Each topic supports:

- `include_any` (at least one term must match)
- `include_all` (all terms must match; optional)
- `exclude_any` (any term match disqualifies)
- `co_require_groups` (for each group, at least one term must match)

A project may match multiple topics; `matched_topics` is retained for traceability.

## AI Keyword Expansion

To improve search recall, enable AI-powered keyword expansion in your config. See **[AI_EXPANSION.md](./AI_EXPANSION.md)** for:
- Setup instructions
- Cost estimates
- Configuration options
- Best practices
