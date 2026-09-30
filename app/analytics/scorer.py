"""Inference wrapper around the trained TabTransformer.

Singleton-loaded on first use, held in module state. If the checkpoint
file is missing the scorer stays None and PDP falls back to the stub.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.analytics.features import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    encode_categorical,
    encode_continuous,
)
from app.analytics.model import TabTransformer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model.pt"


class FlowScorer:
    def __init__(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = TabTransformer(
            cat_cardinalities=ckpt["cat_cardinalities"],
            num_continuous=ckpt["num_continuous"],
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.cont_mean = np.asarray(ckpt["cont_mean"], dtype=np.float32)
        self.cont_std = np.asarray(ckpt["cont_std"], dtype=np.float32)
        # Medians are stored already in log1p space (matches training).
        self.cont_median_log = np.asarray(ckpt["cont_median_log1p"], dtype=np.float32)

    @torch.no_grad()
    def score(self, flow: dict) -> tuple[float, str]:
        """flow: dict with keys matching CATEGORICAL_COLS + CONTINUOUS_COLS.

        Returns (probability_malicious, top_feature_note).
        """
        # Categoricals — vocab lookup.
        cat_ids = np.asarray(
            [encode_categorical(c, flow.get(c)) for c in CATEGORICAL_COLS],
            dtype=np.int64,
        )[None, :]

        # Continuous — coerce, log1p, then z-score with train stats.
        raw = np.asarray(
            [encode_continuous(flow.get(c), median=0.0) for c in CONTINUOUS_COLS],
            dtype=np.float32,
        )
        logged = np.log1p(raw)
        # Impute any zeros as train-set medians — matches the CSV pipeline's
        # behaviour where 0 typically meant "missing" for these fields.
        #logged = np.where(logged == 0.0, self.cont_median_log, logged)
        cont = (logged - self.cont_mean) / self.cont_std

        logits = self.model(
            torch.from_numpy(cat_ids),
            torch.from_numpy(cont[None, :]),
        )
        p = float(torch.sigmoid(logits).item())

        # Cheap "why" note: which categorical stood out from the norm.
        note = f"proto={flow.get('proto')} service={flow.get('service')} state={flow.get('conn_state')}"
        return p, note


_scorer: Optional[FlowScorer] = None
_load_attempted = False


def get_scorer() -> Optional[FlowScorer]:
    """Lazy singleton. Returns None if no checkpoint is present."""
    global _scorer, _load_attempted
    if _load_attempted:
        return _scorer
    _load_attempted = True
    if DEFAULT_CHECKPOINT.exists():
        try:
            _scorer = FlowScorer(DEFAULT_CHECKPOINT)
            print(f"[scorer] loaded {DEFAULT_CHECKPOINT}")
        except Exception as e:
            print(f"[scorer] failed to load checkpoint ({e}); falling back to stub")
            _scorer = None
    else:
        print(f"[scorer] no checkpoint at {DEFAULT_CHECKPOINT}; falling back to stub")
    return _scorer
