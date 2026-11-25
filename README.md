# Meeting Intelligence System

This project turns meeting transcripts into a structured briefing. It identifies the most important parts of a discussion and produces a summary, key decisions, action items, and the evidence behind them.

I built it as an NLP pipeline with trained models, a FastAPI backend, and a React frontend. The goal is to make long meetings easier to review without having to read the entire transcript again.

## What it does

- Accepts a meeting as a list of speaker utterances or as a raw transcript.
- Scores utterances to identify the most important information.
- Detects statements that represent decisions.
- Extracts action items, including the action, assignee, and deadline when available.
- Selects supporting evidence from the original meeting.
- Generates a narrative meeting summary.
- Groups related evidence into concise decision and action-item bullets.
- Provides a web interface for uploading a meeting file and reviewing the results.

The system uses trained NLP models for the analysis. The backend loads the models locally and exposes the pipeline through an API.

## How it works

The analysis is split into several stages:

1. **Input and preprocessing** – The API receives the meeting transcript and prepares it for analysis. Speaker information is kept so that results can be linked back to the people who said them.
2. **Importance scoring** – An importance classifier assigns scores to the meeting utterances so the pipeline can focus on the most relevant content.
3. **Decision detection** – A decision classifier estimates which utterances contain decisions.
4. **Action-item extraction** – A token-classification model identifies action-related text and extracts details such as the action, assignee, and deadline.
5. **Evidence selection** – The pipeline selects relevant sentences and keeps the original text as evidence for the results.
6. **Summary generation** – A BART-based summarization model creates a narrative summary using the selected meeting content.
7. **Bullet fusion** – A T5-based model groups related evidence and turns it into concise bullets for key decisions and action items.
8. **Response** – The API returns the summary, decisions, action items, and selected evidence as structured JSON.

## Project structure

```text
graduation/
├── backend/
│   ├── run.py
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── dependencies.py
│       ├── routers/
│       │   ├── health.py
│       │   └── meetings.py
│       ├── schemas/
│       │   └── meeting.py
│       └── services/
│           ├── model_registry.py
│           ├── preprocessing.py
│           ├── scoring.py
│           ├── evidence_selection.py
│           ├── bullet_fusion.py
│           └── pipeline.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── types.ts
│   │   └── components/
│   └── package.json
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_importance_classifier.ipynb
│   ├── 03_decision_classifier.ipynb
│   ├── 04_action_extractor.ipynb
│   ├── 05_classifier_guided_summarization.ipynb
│   ├── 06_meeting_demo.ipynb
│   └── 07_bullet_fusion.ipynb
├── example.json
└── requirements.txt
```

### Main parts

- **`notebooks/`** – The experiments and training workflow for preparing the data, training the classifiers and extraction model, and building the summarization and bullet-fusion stages.
- **`backend/app/services/`** – The core analysis pipeline. Each service handles one part of the meeting analysis.
- **`backend/app/routers/`** – The API endpoints used by the frontend.
- **`backend/app/schemas/`** – Pydantic models that define the request and response formats.
- **`frontend/`** – The React interface for submitting meeting data and viewing the generated briefing.

## Models

The backend expects the trained models to be available in a `models/` directory at the project root:

```text
models/
├── importance_classifier.pkl
├── bert_decision/
├── bert_action_extractor/
├── summarization/
│   └── classifier_guided_bart_ami/
└── bullet_fusion_t5/
    └── bullet_fusion_t5/
```

The importance classifier, decision model, action extractor, and summarization model are required by the API. The bullet-fusion model is optional. If it is not available, the API can still return results, but the decision and action bullets use a simpler fallback instead of the T5 fusion model.

If the models are stored somewhere else, set `MODELS_DIR` in `backend/.env`.

## Requirements

- Python 3.11 or newer
- Node.js 18 or newer
- The trained model files
- A machine that can run the models; a GPU is optional, but inference will generally be slower on CPU

## Setup

### 1. Set up the Python environment

From the project root:

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Configure the backend

Copy the example environment file:

```bash
cd backend
```

On Windows:

```bash
copy .env.example .env
```

On macOS / Linux:

```bash
cp .env.example .env
```

Edit `.env` if the model directory or any pipeline settings need to be changed. Then return to the project root.

### 3. Start the backend

```bash
cd backend
python run.py
```

The API runs at:

```text
http://localhost:8000
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

### 4. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally:

```text
http://localhost:5173
```

Keep the backend running while using the frontend.

## Using the application

1. Open the frontend in your browser.
2. Upload or select a meeting JSON file.
3. Check the meeting preview and speaker turns.
4. Start the analysis.
5. Review the generated summary, key decisions, action items, and supporting evidence.

A sample request is included in `example.json` and in the backend/frontend sample request files.

## API

### `GET /health`

Checks whether the backend is running and reports the device and whether the optional bullet-fusion model is available.

Example response:

```json
{
  "status": "ok",
  "device": "cuda",
  "bullet_fusion_available": true
}
```

### `POST /meetings/analyze`

Analyzes a meeting. The request can contain either `utterances` or a raw `transcript`.

Example using utterances:

```json
{
  "meeting_id": "team-meeting",
  "utterances": [
    {
      "speaker": "Alice",
      "text": "I think we should use PostgreSQL for the new service."
    },
    {
      "speaker": "Bob",
      "text": "Agreed. Let's use PostgreSQL."
    },
    {
      "speaker": "Carol",
      "text": "I'll prepare the database schema by Friday."
    }
  ]
}
```

The response contains:

- `summary` – the generated narrative summary.
- `key_decisions` – decision bullets, confidence scores, speakers, and supporting evidence.
- `action_items` – extracted actions, assignees, deadlines, speakers, and evidence.
- `selected_evidence` – the utterances selected by the pipeline, together with their scores and classification flags.
- `bullet_fusion_used` – indicates whether the optional bullet-fusion model was used.

A raw transcript can be sent instead:

```json
{
  "meeting_id": "team-meeting",
  "transcript": "Alice: We should use PostgreSQL. Bob: Agreed. Carol: I will prepare the schema by Friday."
}
```

## Training and experiments

The notebooks are numbered in the order of the main workflow:

| Notebook | Purpose |
|---|---|
| `01_data_preparation.ipynb` | Prepare the meeting data for model training. |
| `02_importance_classifier.ipynb` | Train and evaluate the importance classifier. |
| `03_decision_classifier.ipynb` | Train and evaluate the decision classifier. |
| `04_action_extractor.ipynb` | Train and evaluate the action-item extraction model. |
| `05_classifier_guided_summarization.ipynb` | Develop the classifier-guided summarization approach. |
| `06_meeting_demo.ipynb` | Bring the trained models together in the meeting analysis pipeline. |
| `07_bullet_fusion.ipynb` | Train and evaluate the model used to combine evidence into concise bullets. |

The root `requirements.txt` contains the dependencies used for the notebooks and model development. The backend also has its own `requirements.txt` for running the API.

## Notes

- Models are loaded once when the backend starts, rather than being loaded again for every request.
- The first startup can take some time because the model files need to be loaded into memory.
- The required models must be present before starting the API.
- The bullet-fusion model is optional; without it, the API uses the fallback bullet formatting.
- The backend can be tested independently through the interactive documentation at `/docs`.

