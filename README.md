# NIH RePORTER PI Finder

A full-stack tool for discovering NIH-funded Principal Investigators (PIs) by research topic. It combines a broad NIH RePORTER API query with strict local topic-matching rules to produce a clean, deduplicated outreach list — exportable as CSV.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (React + Vite)                                         │
│  • YAML editor  • Status polling  • Paginated results table     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ /api/* (Vite proxy → :8000)
┌───────────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend (:8000)                                        │
│                                                                 │
│  POST /api/runs ──► Background task                             │
│                         │                                       │
│              ┌──────────▼──────────┐                           │
│              │  Stage 1: NIH Query │  httpx + diskcache        │
│              │  fiscal_years +     │  30-day cache             │
│              │  broad_keywords     │  ~1 req/sec rate limit    │
│              └──────────┬──────────┘                           │
│                         │ ~500–5000 raw projects                │
│              ┌──────────▼──────────┐                           │
│              │  Stage 2: Topic     │  Pure Python              │
│              │  Matching (local)   │  include_any / all /      │
│              │                     │  exclude_any /            │
│              │                     │  co_require_groups        │
│              └──────────┬──────────┘                           │
│                         │ filtered & deduplicated PIs           │
│              ┌──────────▼──────────┐                           │
│              │  Aggregation        │  Group by core_project_num│
│              │  & CSV export       │  Pandas → CSV             │
│              └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Why two stages?

The NIH RePORTER API only supports broad keyword search. A single keyword like `"AI"` returns thousands of unrelated grants (lab equipment, AI acronyms in other fields, etc.). Stage 2 applies precise, configurable topic rules **locally** — no extra API calls — to filter down to exactly what you want.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, js-yaml |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| HTTP client | httpx (async) |
| Caching | diskcache (SQLite-backed, 30-day TTL) |
| Data export | Pandas → CSV |
| AI features | OpenAI API (gpt-4o-mini default) |
| Config format | YAML |

---

## Features

- **Two-stage filtering** — broad NIH API query + strict local topic matching
- **MeSH concept expansion** — expands search terms with National Library of Medicine biomedical terminology before NIH search (optional, with graceful fallback)
- **AI keyword expansion** — automatically expands search terms with synonyms, acronyms, and variations after MeSH expansion (optional, requires OpenAI API key)
- **AI config suggestion** — describe a research topic in plain English; the app generates `broad_keywords` and `topic_terms` for you
- **PI deduplication** — groups projects by `core_project_num` (year-invariant), derives PI info from the most recent fiscal year
- **Multi-year aggregation** — collects funding, dates, and abstracts across all fiscal years per project
- **CSV export** — 22-column export with full traceability (abstracts, terms, project URLs)
- **30-day disk cache** — avoids re-querying the NIH API for repeated searches
- **Rate limiting** — ~1 req/sec to stay within NIH API limits

---

## Project Structure

```
.
├── backend/
│   ├── requirements.txt
│   ├── config.example.yaml          # Annotated config template
│   └── app/
│       ├── main.py                  # FastAPI routes
│       ├── runner.py                # Pipeline orchestrator
│       ├── reporter_client.py       # NIH API client (async, cached)
│       ├── processor.py             # Topic matching + PI aggregation
│       ├── topic_matcher.py         # include/exclude/co_require logic
│       ├── mesh_expander.py         # NLM MeSH concept expansion
│       ├── keyword_expander.py      # OpenAI keyword expansion
│       ├── keyword_suggester.py     # OpenAI config suggestion
│       ├── models.py                # Pydantic data models
│       ├── run_store.py             # In-memory run state store
│       ├── csv_export.py            # Pandas CSV generation
│       ├── cache.py                 # diskcache wrapper
│       ├── settings.py              # Env-var settings (pydantic-settings)
│       ├── config_loader.py         # YAML → AppConfig parser
│       ├── utils.py                 # Text normalization helpers
│       └── cli.py                   # CLI batch runner
└── frontend/
    ├── vite.config.js               # Proxies /api → localhost:8000
    └── src/
        ├── App.jsx                  # Main UI component
        ├── main.jsx                 # React entry point
        └── styles.css               # Dark theme
```

---

## Quickstart

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Build the reproducible MeSH knowledge-base artifacts used by optional MeSH expansion:

```bash
python scripts/download_mesh_data.py
python scripts/build_mesh_kb.py
```

You can also run the same two steps from the project root:

```bash
python backend/scripts/download_mesh_data.py
python backend/scripts/build_mesh_kb.py
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically **http://localhost:5173**).

> The Vite dev server proxies all `/api/*` requests to the FastAPI backend at `:8000`, so no CORS configuration is needed during development.

The default frontend workflow now supports plain-language search setup:

`Research question → Generate Concepts → Review concept chips → Start Search`

Advanced YAML remains available for power users, but non-technical users no longer need to write YAML or keywords manually to start.

---

## Configuration (YAML)

Copy `backend/config.example.yaml` as your starting point. The config has two sections:

### `query` — what to fetch from NIH

```yaml
query:
  fiscal_years: [2023, 2024, 2025]   # Which fiscal years to search
  broad_keywords:                     # Keywords sent to NIH API (Stage 1)
    - health disparities
    - artificial intelligence
  text_search_field: all              # 'all' | 'title' | 'abstract' | 'terms'
  text_search_operator: or            # 'or' (broader) | 'and' (stricter)

  # Optional: MeSH concept expansion
  mesh_expansion:
    enabled: true
    max_terms_per_keyword: 15
    include_entry_terms: true
    include_tree_children: true
    max_tree_depth: 1
    fallback_to_original: true
    cache_enabled: true

  # Optional: semantic MeSH expansion
  semantic_expansion:
    enabled: false
    top_k: 10
    max_terms: 30
    min_score: null
    include_synonyms: true
    require_existing_index: false

  # Optional: AI-powered keyword expansion
  ai_expansion:
    enabled: false
    openai_api_key: null              # Or set OPENAI_API_KEY env var
    model: gpt-4o-mini
    max_expansions_per_keyword: 5
    context: "biomedical research and health disparities"
```

### `topics` — how to filter locally (Stage 2)

Each project's title + abstract + terms are matched against every topic. A project passes a topic when:

| Rule | Meaning |
|---|---|
| `include_any` | At least one term must appear in the text |
| `include_all` | Every term must appear (optional, additive) |
| `exclude_any` | If any term appears, the project is rejected |
| `co_require_groups` | Each sub-list must have at least one match |

```yaml
topics:
  - name: AI + Health Disparities
    include_any:
      - artificial intelligence
      - machine learning
      - deep learning
    exclude_any:
      - mouse
      - mice
      - rat
      - in vitro
    co_require_groups:
      - [health disparities, equity, inequity, underserved]
      - [clinical, community, public health, population]

  - name: Telehealth Equity
    include_any:
      - telehealth
      - telemedicine
      - remote monitoring
    co_require_groups:
      - [disparities, equity, access, rural, underserved]
```

A project can match multiple topics — the `matched_topics` field records which ones for traceability.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `POST` | `/api/runs` | Start a pipeline run (async) |
| `GET` | `/api/runs/{run_id}` | Poll run status, MeSH trace, and keyword expansions |
| `GET` | `/api/runs/{run_id}/results` | Paginated results (`offset`, `limit`) |
| `GET` | `/api/runs/{run_id}/export.csv` | Download full CSV |
| `POST` | `/api/concepts/suggest` | Suggest MeSH-grounded concepts from a research question |
| `POST` | `/api/suggest-keywords` | AI topic → config suggestion |

### POST `/api/runs` payload

```json
{
  "config_yaml": "query:\n  fiscal_years: [2024]\n  ...",
  "max_pages": 10
}
```

`max_pages` controls how many 500-project pages to fetch from the NIH API (capped at 200 server-side). At ~1 req/sec, 10 pages ≈ 10 seconds; 100 pages ≈ 100 seconds.

---

## Environment Variables

All variables are prefixed with `OUTREACH_`:

| Variable | Default | Description |
|---|---|---|
| `OUTREACH_REPORTER_API_BASE_URL` | `https://api.reporter.nih.gov/v2` | NIH API base URL |
| `OUTREACH_REPORTER_RATE_LIMIT_SECONDS` | `1.0` | Seconds between NIH API requests |
| `OUTREACH_CACHE_DIR` | `.cache` | Disk cache directory |
| `OUTREACH_CACHE_TTL_SECONDS` | `2592000` (30 days) | Cache TTL |
| `OPENAI_API_KEY` | — | OpenAI API key for AI features |

Create a `.env` file in the `backend/` directory or export these before starting the server.

---

## MeSH Concept Expansion

The original limitation in this project was exact keyword dependency. If a user searched for `telemedicine`, `health disparities`, or `artificial intelligence`, the NIH query depended on investigators using those exact words in searchable text.

To reduce that fragility, the backend now supports an optional MeSH preprocessing stage that expands YAML `broad_keywords` with National Library of Medicine biomedical terminology before the existing NIH RePORTER query is built.

Before:
`YAML Keywords → Optional AI Expansion → NIH RePORTER → Filtering → PI CSV`

After:
`YAML Keywords → Optional MeSH Expansion → Optional Semantic MeSH Expansion → Optional AI Expansion → NIH RePORTER → Filtering → PI CSV`

Enable it in YAML:

```yaml
query:
  broad_keywords:
    - telemedicine
    - health disparities
  mesh_expansion:
    enabled: true
    max_terms_per_keyword: 15
    include_entry_terms: true
    include_tree_children: true
    max_tree_depth: 1
    fallback_to_original: true
    cache_enabled: true
```

MeSH expansion is optional. If the block is omitted, the app behaves like the original version. If MeSH lookup is unavailable, the run falls back to the original keywords and downstream NIH search, PI extraction, local filtering, and CSV export remain unchanged. Email enrichment is still intentionally deferred because NIH RePORTER does not expose PI emails directly in a reliable way.

## Semantic MeSH Expansion

Lexical MeSH expansion uses MeSH synonyms and hierarchy. Semantic MeSH expansion uses vector search over the local MeSH descriptor metadata to find related concepts even when the exact wording differs.

Semantic expansion is optional. It requires prebuilt embedding artifacts and does not download models or build embeddings during a normal NIH RePORTER run.

The frontend concept generator uses MeSH-grounded suggestions when the semantic index is available, falls back to local MeSH lookup when possible, and finally falls back to simple local phrase extraction so the UI stays responsive.

Build and inspect the semantic index locally:

```bash
cd backend
python scripts/build_mesh_embeddings.py --limit 1000
python scripts/build_mesh_embeddings.py --batch-size 32 --checkpoint-every 25 --resume
python scripts/query_mesh_semantic.py "AI for diabetes in underserved populations" --top-k 10
```

The default embedding model is lightweight for Codespaces. We are not using the PubMed corpus directly in this phase; later phases can swap in biomedical embedding models if needed.

To refresh the local MeSH knowledge base in Codespaces or any other clean checkout:

```bash
cd backend
python scripts/download_mesh_data.py
python scripts/build_mesh_kb.py
```

`download_mesh_data.py` saves the raw yearly XML files under `backend/knowledge/mesh/`. `build_mesh_kb.py` converts those raw inputs into the ignored processed artifacts under `backend/knowledge/processed/`:

- `mesh_descriptors.json`
- `mesh_graph.json`
- `mesh_lookup.pkl`

---

## CSV Export Columns

| Column | Description |
|---|---|
| `pi_name` | Full name of contact PI |
| `pi_first_name` / `pi_last_name` | Split name components |
| `pi_email` | Contact email (often absent in NIH data) |
| `organization_name` | Institution name |
| `organization_city/state/country` | Institution location |
| `admin_ic` | NIH Institute/Center abbreviation (e.g. `NCI`, `NIMHD`) |
| `fiscal_years` | All fiscal years this project appeared in (semicolon-separated) |
| `project_count` | Number of fiscal-year records for this core project |
| `matched_topics` | Which topic rules matched (semicolon-separated) |
| `sample_project_titles` | Up to 3 project titles |
| `project_numbers` | Full project numbers (semicolon-separated) |
| `project_abstracts` | All abstracts across fiscal years |
| `project_terms` | NIH MeSH-style terms |
| `project_ids` | Application IDs (links to NIH RePORTER) |
| `project_urls` | Direct URLs to NIH RePORTER project pages |
| `pi_profile_id` | NIH PI profile ID |
| `total_funding_amount` | Sum of award amounts across fiscal years |
| `project_start_date` | Earliest project start date |
| `project_end_date` | Latest project end date |

---

## CLI Batch Mode

Run searches without the web UI:

```bash
cd backend
source .venv/bin/activate
python -m app.cli \
  --config config.example.yaml \
  --out-dir output/2025 \
  --max-pages 50
```

Output files:
- `results.json` — full PI outreach rows
- `summary.json` — counts by topic, year, IC
- `keyword_expansions.json` — AI expansion mapping
- `outreach.csv` — the final export

---

## AI Features

### Keyword Expansion

When `ai_expansion.enabled: true`, the backend sends each `broad_keyword` to OpenAI and gets back synonyms, acronyms, and related terms. These replace the original keywords in the NIH API query, improving recall.

Example: `"health disparities"` might expand to `["health disparities", "health equity", "health inequities", "underserved populations", "minority health"]`.

Expansions are cached for 30 days so the same keyword set is never re-sent to OpenAI.

### Config Suggestion

Click **🤖 Suggest Keywords** in the UI, describe your research topic in plain English, and the app generates a ready-to-use set of `broad_keywords` and `topic_terms` using OpenAI. The result can be applied to the YAML editor in one click.

Requires an OpenAI API key set in the YAML config or via `OPENAI_API_KEY` env var.

---

## Development Notes

- The NIH RePORTER API returns snake_case field names. The code handles multiple field name variants for robustness (e.g. `org_name` vs `organization_name`).
- The `core_project_num` field is the year-invariant project identifier (e.g. `R01CA123456`). The same project number appears once per fiscal year; the processor groups these and picks the most recent year for PI/org contact info.
- Disk cache keys are SHA-256 hashes of the request payload, so any change to the query (different fiscal years, keywords, etc.) produces a cache miss.
- The run store is in-memory; runs are lost on server restart. For persistent storage, replace `run_store.py` with a SQLite or Redis backend.

---

## License

MIT
