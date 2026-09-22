"""Application configuration.

Every value here mirrors a hyperparameter used in notebooks/06_meeting_demo.ipynb and
notebooks/07_bullet_fusion.ipynb, so the API produces the same behavior as the notebooks.
Any value can be overridden with an environment variable of the same name (case-insensitive),
or via a `.env` file placed next to this backend's `run.py`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Where trained model artifacts live. Matches the layout produced by the training
    # notebooks: importance_classifier.pkl, bert_decision/, bert_action_extractor/,
    # summarization/classifier_guided_bart_ami/, bullet_fusion_t5/bullet_fusion_t5/.
    models_dir: Path = PROJECT_ROOT / "models"

    # Classifier inference (notebooks/06_meeting_demo.ipynb, section 4)
    classifier_batch_size: int = 32
    # Raised from 0.50 to reduce false positives on questions and suggestions. At 0.50 the
    # decision model was flagging modal-verb questions ("Could we...?") as confirmed decisions,
    # causing BART to treat rejected proposals as team decisions.
    decision_threshold: float = 0.68

    # Chunking for raw-transcript input, used only when the request sends `transcript`
    # instead of pre-segmented `utterances` (mirrors chunk_external_text).
    chunk_max_words: int = 40

    # Narrative summary generation (BART), notebooks/06_meeting_demo.ipynb, section 5
    # Source token budget raised from 900 to 1200 so that more transcript context reaches BART
    # and bridging utterances between important chunks are less likely to be dropped.
    summary_max_source_tokens: int = 1200
    summary_min_new_tokens: int = 60
    # Output raised from 220 to 300 to give BART room to cover all decisions without truncating.
    summary_max_new_tokens: int = 300
    summary_num_beams: int = 4
    summary_length_penalty: float = 2.0

    # Bullet fusion generation (T5), notebooks/07_bullet_fusion.ipynb
    fusion_max_source_tokens: int = 160
    fusion_max_target_tokens: int = 64
    fusion_num_beams: int = 4

    # How close two flagged chunks must be (by original transcript position) to be
    # merged into one evidence group before fusion, notebooks/06_meeting_demo.ipynb, section 5b
    group_max_index_gap: int = 2

    # Number of utterances on each side of a scored chunk to include as context when the
    # decision classifier is run, so that rejected suggestions are not mislabelled as decisions.
    decision_context_window: int = 2

    @property
    def importance_path(self) -> Path:
        return self.models_dir / "importance_classifier.pkl"

    @property
    def decision_path(self) -> Path:
        return self.models_dir / "bert_decision"

    @property
    def action_path(self) -> Path:
        return self.models_dir / "bert_action_extractor"

    @property
    def summarizer_path(self) -> Path:
        return self.models_dir / "summarization" / "classifier_guided_bart_ami"

    @property
    def fusion_path(self) -> Path:
        return self.models_dir / "bullet_fusion_t5" / "bullet_fusion_t5"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
