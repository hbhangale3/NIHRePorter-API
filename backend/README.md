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
