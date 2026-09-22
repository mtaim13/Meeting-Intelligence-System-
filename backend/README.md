# Meeting Intelligence API

A FastAPI backend that wraps the trained pipeline from `notebooks/06_meeting_demo.ipynb` into an
HTTP API: send a meeting transcript, get back a narrative summary, key decisions, and action items.

## Architecture

```
backend/
├── run.py                       # `python run.py` starts the dev server
├── requirements.txt
├── .env.example                 # copy to .env to override settings (MODELS_DIR, thresholds, ...)
└── app/
    ├── main.py                  # FastAPI app, CORS, startup model loading (lifespan)
    ├── config.py                # Settings: model paths + every pipeline hyperparameter
    ├── dependencies.py          # get_registry() — injects the loaded models into routes
    ├── schemas/
    │   └── meeting.py           # Request/response Pydantic models
    ├── routers/
    │   ├── health.py            # GET /health
    │   └── meetings.py          # POST /meetings/analyze
    └── services/                # All business logic, framework-agnostic
        ├── model_registry.py    # Loads the 5 trained models once at startup
        ├── preprocessing.py     # Turns request input into scoring-ready chunks
        ├── scoring.py           # Importance / decision / action model inference
        ├── evidence_selection.py# Picks evidence + generates the narrative summary (BART)
        ├── bullet_fusion.py     # Groups flagged evidence + fuses into bullets (T5)
        └── pipeline.py          # Orchestrates the steps above into one response
```

Every hyperparameter (decision threshold, token budgets, beam sizes, grouping gap, ...) matches
`notebooks/06_meeting_demo.ipynb` and `notebooks/07_bullet_fusion.ipynb` exactly — see `app/config.py`.

## Setup

1. Train (or download already-trained) models with the notebooks in `notebooks/`, then place them
   under `models/` at the project root, matching this layout:

   ```
   models/
   ├── importance_classifier.pkl
   ├── bert_decision/
   ├── bert_action_extractor/
   ├── summarization/classifier_guided_bart_ami/
   └── bullet_fusion_t5/bullet_fusion_t5/      (optional — see fallback note below)
   ```

   If your models live elsewhere, copy `.env.example` to `.env` and set `MODELS_DIR`.

2. Install dependencies (a GPU is optional; CPU works, just slower per request):

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. Run the server:

   ```bash
   python run.py
   ```

   Or directly with uvicorn: `uvicorn app.main:app --reload`.

   Interactive docs: http://localhost:8000/docs

**Note on the bullet fusion model**: if `models/bullet_fusion_t5/bullet_fusion_t5/` isn't present, the
API still works — `key_decisions`/`action_items` bullets fall back to raw flagged sentences joined
with `" / "`, and the response's `bullet_fusion_used` field is `false` so callers can tell.

## API

### `GET /health`

```json
{"status": "ok", "device": "cuda", "bullet_fusion_available": true}
```

### `POST /meetings/analyze`

Request — either `utterances` (preferred) or a raw `transcript`:

```json
{
  "meeting_id": "standup-2026-08-11",
  "utterances": [
    {"speaker": "Alice", "text": "I think we should use PostgreSQL instead of MongoDB."},
    {"speaker": "Bob", "text": "Agreed, let's go with Postgres for the relational data."},
    {"speaker": "Alice", "text": "Great, it's decided then, we'll use PostgreSQL."},
    {"speaker": "Carol", "text": "I'll set up the database schema by Friday."}
  ]
}
```

or:

```json
{
  "meeting_id": "standup-2026-08-11",
  "transcript": "Alice: I think we should use PostgreSQL... (long raw text)"
}
```

Response:

```json
{
  "meeting_id": "standup-2026-08-11",
  "summary": "The team discussed database options and decided to use PostgreSQL...",
  "key_decisions": [
    {
      "bullet": "The team decided to use PostgreSQL instead of MongoDB.",
      "confidence": 0.94,
      "speakers": "Alice, Bob",
      "evidence": [
        "I think we should use PostgreSQL instead of MongoDB.",
        "Agreed, let's go with Postgres for the relational data.",
        "Great, it's decided then, we'll use PostgreSQL."
      ]
    }
  ],
  "action_items": [
    {
      "action": "Set up the database schema.",
      "assignee": "Carol",
      "deadline": "Friday",
      "speakers": "Carol",
      "evidence": ["I'll set up the database schema by Friday."]
    }
  ],
  "selected_evidence": [
    {"speaker": "Alice", "text": "...", "importance": 0.81, "decision_score": 0.92, "is_decision": true, "is_action": false}
  ],
  "bullet_fusion_used": true
}
```

Quick test with curl:

```bash
curl -X POST http://localhost:8000/meetings/analyze \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Notes

- Models are loaded once at startup (FastAPI `lifespan`), not per-request — the first request after
  boot is fast, not the one paying the model-loading cost.
- If a required model (importance, decision, action, or summarizer) is missing, the server fails to
  start with a clear error listing exactly which path is missing. The fusion model is the only
  optional one.
- `GET /health` reports `bullet_fusion_available` so a frontend can indicate when bullets are
  raw-fallback vs. fused.
