# AI Conversation Intelligence Platform

A fully local NLP system that analyzes Telegram group chats and produces structured intelligence: topic segments, importance scores, action items, decisions, semantic search, and extractive QA.

No LLMs. No external APIs. Every model runs locally.

---

## Project Structure

```
project/
├── data/
│   ├── raw/           # Downloaded corpora (AMI, ELITR, QMSum)
│   ├── processed/     # Preprocessed JSON files (output of scripts/)
│   └── labeled/       # Your manually annotated Telegram messages
│
├── models/            # Saved model checkpoints (created during training)
│   ├── importance_classifier/
│   ├── action_extractor/
│   └── decision_classifier/
│
├── src/
│   ├── models/        # Model class definitions
│   └── data/          # Dataset classes and data loaders
│
├── training/          # Training scripts (one per model)
├── scripts/           # Data preprocessing utilities
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

---

## Training Datasets

### Where to Download

| Model | Primary Dataset | Secondary Dataset | Download Link |
|---|---|---|---|
| Importance Classifier | AMI Meeting Corpus | QMSum | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) · [QMSum](https://github.com/Yale-LILY/QMSum) |
| Action Extractor | AMI Meeting Corpus | ELITR Minuting Corpus | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) · [ELITR](https://github.com/ELITR/elitr-minuting-corpus) |
| Decision Classifier | AMI Meeting Corpus | ICSI Meeting Corpus | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) · [ICSI](https://groups.inf.ed.ac.uk/ami/icsi/) |

All datasets are free. AMI and ICSI require a free registration.

### Dataset Notes

**AMI Meeting Corpus** (~100 meetings, ~100k words)
The most important dataset for this project. Contains:
- Abstractive meeting summaries with annotated decisions and action items
- Word-level transcriptions
- Used for all three models

**QMSum** (~232 meetings with query-answer pairs)
Sentences that answer a query are treated as "important". Used for the importance classifier.

**ELITR Minuting Corpus** (~50 meetings)
Meeting minutes with explicit action item markers (`[Action]` prefix). Used for the action extractor.

**ICSI Meeting Corpus** (~75 meetings)
Similar to AMI. Adds more decision-annotated examples for the decision classifier.

---

## Step 1: Preprocess AMI Corpus

```bash
# Download AMI to data/raw/ami_corpus/, then:
python scripts/prepare_ami_data.py \
    --ami_root data/raw/ami_corpus \
    --output data/processed
```

This creates:
- `data/processed/ami_decisions.json`
- `data/processed/ami_actions.json`
- `data/processed/ami_importance.json`

---

## Step 2: Add Your Manual Labels

Use [Doccano](https://github.com/doccano/doccano) (free, local browser tool) to annotate
your own Telegram messages. Export in JSON format and place in `data/labeled/`.

Template files are already in `data/labeled/` showing the expected format.

Target annotation counts:
- `importance_labels.json` — 150 messages (binary)
- `action_labels.json`     — 80 messages (token-level BIO)
- `decision_labels.json`   — 70 messages (binary)

---

## Step 3: Train the Models

### Importance Classifier
```bash
python training/train_importance.py \
    --data data/labeled/importance_labels.json \
    --ami_root data/raw/ami_corpus \
    --qmsum data/raw/qmsum/train.jsonl \
    --output models/importance_classifier \
    --epochs 5 \
    --use_class_weights
```

### Action Item Extractor
```bash
python training/train_action_extractor.py \
    --data data/labeled/action_labels.json \
    --ami_root data/raw/ami_corpus \
    --elitr_root data/raw/elitr_corpus \
    --output models/action_extractor \
    --epochs 5
```

### Decision Classifier
```bash
python training/train_decision_classifier.py \
    --data data/labeled/decision_labels.json \
    --ami_root data/raw/ami_corpus \
    --output models/decision_classifier \
    --epochs 5 \
    --use_class_weights
```

All training scripts support `--help` for the full argument list.

---

## Step 4: Load and Run Inference

```python
import torch
from src.models import ImportanceClassifier, ActionExtractor, DecisionClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importance scoring
imp_model     = ImportanceClassifier.load("models/importance_classifier").to(device)
imp_tokenizer = ImportanceClassifier.build_tokenizer("models/importance_classifier")
print(imp_model.predict("We need to refactor auth before Friday.", imp_tokenizer, device))
# → {"label": "important", "score": 0.92}

# Action extraction
act_model     = ActionExtractor.load("models/action_extractor").to(device)
act_tokenizer = ActionExtractor.build_tokenizer("models/action_extractor")
print(act_model.predict("Ahmed should fix the login bug by Friday.", act_tokenizer, device))
# → [{"action": "fix the login bug", "assignee": "Ahmed", "deadline": "Friday"}]

# Decision detection
dec_model     = DecisionClassifier.load("models/decision_classifier").to(device)
dec_tokenizer = DecisionClassifier.build_tokenizer("models/decision_classifier")
print(dec_model.predict("We decided to use PostgreSQL.", dec_tokenizer, device))
# → {"label": "decision", "score": 0.97, "rule_matched": True}
```

---

## MLflow Experiment Tracking

Training metrics are logged automatically. To view them:

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Models Summary

| Model | Architecture | Task | Evaluation Metric |
|---|---|---|---|
| Importance Classifier | DistilBERT + Linear(768→2) | Binary classification | F1, AUC-ROC |
| Action Extractor | BERT + Linear(768→7) | BIO token classification | Entity-level F1 (seqeval) |
| Decision Classifier | DistilBERT + Linear(768→2) + rule filter | Binary classification | F1, AUC-ROC |
