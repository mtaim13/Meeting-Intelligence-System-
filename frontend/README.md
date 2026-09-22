# Meeting Intelligence Frontend

React + Vite UI for the local Meeting Intelligence API.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

Start the backend separately (`python run.py` from `backend/`) on port 8000. The Vite dev server proxies `/api` → `http://localhost:8000`.

## Usage

1. Drop or browse a meeting `.json` file (see `public/sample_request.json`).
2. Preview speaker turns, optionally edit the meeting ID.
3. Click **Analyze meeting** to call `POST /meetings/analyze`.
4. Read the briefing: summary, key decisions, action items, and selected evidence.

## JSON shape

```json
{
  "meeting_id": "optional-id",
  "utterances": [
    { "speaker": "Alice", "text": "..." }
  ]
}
```

Or use `"transcript": "raw text..."` instead of `utterances`.

## Config

Optional env var `VITE_API_BASE` (default `/api`) if you point at another API origin.
