# Backend (FastAPI)

## Setup

1. Create a virtualenv
2. Install deps:

`pip install -r requirements.txt`

## Run API

`uvicorn app.main:app --reload --port 8000`

## Example config

`config.example.yaml`

## CLI yearly rerun

`python -m app.cli --config path/to/config.yaml --out-dir out/2026 --max-pages 50`

## API runs

POST `/api/runs` accepts:

- `config_yaml` (string)
- `max_pages` (int|null) optional interactive safety limit for pagination

## MeSH Concept Expansion

The backend now supports an optional `query.mesh_expansion` block that expands `broad_keywords` with NLM MeSH terminology before the existing NIH RePORTER search.

Before:
`YAML Keywords → Optional AI Expansion → NIH RePORTER → Filtering → PI CSV`

After:
`YAML Keywords → Optional MeSH Expansion → Optional AI Expansion → NIH RePORTER → Filtering → PI CSV`

If `mesh_expansion` is omitted, behavior stays exactly the same as before. If live MeSH lookup fails, the pipeline logs the issue and falls back to the original keywords so runs still complete.

## Rebuild MeSH Knowledge Base

From the `backend/` directory:

`python scripts/download_mesh_data.py`

`python scripts/build_mesh_kb.py`

From the project root:

`python backend/scripts/download_mesh_data.py`

`python backend/scripts/build_mesh_kb.py`

Raw XML files are stored in `backend/knowledge/mesh/`. The rebuild step writes the ignored processed artifacts to `backend/knowledge/processed/`:

- `mesh_descriptors.json`
- `mesh_graph.json`
- `mesh_lookup.pkl`
