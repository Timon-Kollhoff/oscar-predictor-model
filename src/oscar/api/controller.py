"""Helpers for the API: load model and data once, compute the chances of winning."""

from oscar.features import OUT_FILE
from oscar.train import load
from functools import lru_cache
import numpy as np
import pandas as pd
from fastapi import HTTPException

# field name in the API -> column in the model
FEATURE_MAP = {"pga_winner": "pga_win", "dga_winner": "dga_win", "sag_winner": "sag_win"}

NO_SIGNAL = ("All films have the same guild awards, so the model cannot tell them apart. "
             "A prediction is only possible after PGA, DGA and SAG.")


# ---------------------------------------------------------------- Helpers
@lru_cache
def get_model():
    """Loads the model and its metadata on the first call and caches them."""
    try:
        model, meta = load()
    except FileNotFoundError:
        raise HTTPException(503, "No model found. Train it first: uv run python -m oscar.train")
    if set(meta["features"]) != set(FEATURE_MAP.values()):
        raise RuntimeError(f"The model expects {meta['features']}, the API provides {list(FEATURE_MAP.values())}")
    return model, meta


@lru_cache
def get_nominations() -> pd.DataFrame:
    try:
        return pd.read_csv(OUT_FILE)
    except FileNotFoundError:
        raise HTTPException(503, "No data found. Run the pipeline first: "
                                 "uv run python run_pipeline.py")


def probabilities(model, meta: dict, X: pd.DataFrame) -> np.ndarray:
    """Chance of winning per film, normalized to 100 % within the group."""
    p = model.predict_proba(X[meta["features"]])[:, 1]

    # normalize: the chances add up to 1
    return p / p.sum()


def has_signal(X: pd.DataFrame) -> bool:
    """False if all films have the same guild awards (e.g. before they are announced)."""
    return len(X.drop_duplicates()) > 1