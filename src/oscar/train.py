"""Trains the final Best Picture model and saves it for the API.

The features and the weighting were chosen in notebooks/02_modell.ipynb (section 4)
on the years 2005-2014. Nothing is selected here any more: the chosen model is only
trained on all years and saved.

Run:     uv run python -m oscar.train
Output:  models/best_picture.joblib  (the trained model)
         models/metadata.json        (features, training years, versions, test result)
         models/ceremonies.csv       (nominees and guild awards per ceremony, read by the API)
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from oscar.features import OUT_FILE, ROOT
from oscar.model import (FEATURE_SETS, TARGET, YEAR, hits, make_model, rule_baseline,
                         walk_forward, years_with_winner)

FEATURE_SET = "PGA+DGA+SAG"   # result of the model selection in the notebook
FEATURES = FEATURE_SETS[FEATURE_SET]
HALF_LIFE = None              # no extra weight on recent years (also from the notebook)
FIRST_FILM_YEAR = 1995        # all guild awards exist from here on
TEST_FIRST = 2015             # first test year, only for the record in metadata.json

MODEL_DIR = ROOT / "models"
MODEL_FILE = "best_picture.joblib"
METADATA_FILE = "metadata.json"
CEREMONIES_FILE = "ceremonies.csv"
API_COLUMNS = [YEAR, "Film", "imdb_id", TARGET]  # plus FEATURES


def load_training_data(path: Path = OUT_FILE) -> pd.DataFrame:
    """Nominations from film year 1995 on, only ceremonies with exactly one winner.
    This way an upcoming ceremony whose winner is still open never ends up in training."""
    df = pd.read_csv(path)
    df = df[df["year_film"] >= FIRST_FILM_YEAR]
    years = years_with_winner(df, first=int(df[YEAR].min()))
    return df[df[YEAR].isin(years)].reset_index(drop=True)


def evaluate(data: pd.DataFrame) -> dict:
    """Test as in the notebook (walk-forward from 2015), plus the PGA rule for comparison.
    Only for the record, not for choosing the model."""
    wf = walk_forward(make_model(), data, FEATURES, first=TEST_FIRST, half_life=HALF_LIFE)
    pga = rule_baseline(data, "pga_win", years_with_winner(data, TEST_FIRST))
    return {
        "method": f"walk-forward {TEST_FIRST}-{int(data[YEAR].max())}",
        "ceremonies": int(wf[YEAR].nunique()),
        "model_hits": hits(wf),
        "pga_rule_hits": hits(pga),
    }


def train(data: pd.DataFrame):
    """Trains the chosen model on all ceremony years."""
    model = make_model()
    model.fit(data[FEATURES], data[TARGET])
    return model


def save(model, metadata: dict, out_dir: Path = MODEL_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                         encoding="utf-8")


def save_ceremonies(path: Path = OUT_FILE, out_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """Saves the small table that /prediction/{year} reads: every nominee from film year
    1995 on (upcoming ceremonies included) with only the model's features. It holds facts
    from Kaggle and Wikipedia only, no TMDb or OMDb data, so it can live in the repository
    next to the model, and the API runs without the data pipeline."""
    df = pd.read_csv(path)
    df = df.loc[df["year_film"] >= FIRST_FILM_YEAR, API_COLUMNS + FEATURES]
    df[FEATURES] = df[FEATURES].astype("Int64")  # 1/0 instead of 1.0/0.0 in the CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / CEREMONIES_FILE, index=False)
    return df


def load(model_dir: Path = MODEL_DIR):
    """For the API: load the saved model and its metadata."""
    model = joblib.load(model_dir / MODEL_FILE)
    metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Trains the Best Picture model.")
    parser.add_argument("--data", type=Path, default=OUT_FILE, help="nominations.csv")
    parser.add_argument("--out", type=Path, default=MODEL_DIR, help="output folder for the model")
    args = parser.parse_args()

    data = load_training_data(args.data)
    model = train(data)
    evaluation = evaluate(data)

    metadata = {
        "target": "Best Picture (won)",
        "feature_set": FEATURE_SET,
        "features": FEATURES,
        "half_life": HALF_LIFE,
        "coefficients_standardized": dict(zip(FEATURES, model[-1].coef_[0].round(3).tolist())),
        "intercept": round(float(model[-1].intercept_[0]), 3),
        "train_ceremonies": [int(data[YEAR].min()), int(data[YEAR].max())],
        "n_ceremonies": int(data[YEAR].nunique()),
        "n_nominations": len(data),
        "evaluation": evaluation,
        "note": "The model only knows PGA, DGA and SAG. Before those are awarded, every "
                "film gets the same probability.",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    save(model, metadata, args.out)
    ceremonies = save_ceremonies(args.data, args.out)

    ev = evaluation
    print(f"Trained: {FEATURE_SET} on {metadata['n_ceremonies']} ceremonies "
          f"({metadata['train_ceremonies'][0]}-{metadata['train_ceremonies'][1]}, "
          f"{metadata['n_nominations']} nominations)")
    print(f"Test {ev['method']}: model {ev['model_hits']}/{ev['ceremonies']}, "
          f"PGA rule {ev['pga_rule_hits']}/{ev['ceremonies']}")
    print(f"Saved model, metadata and {len(ceremonies)} nominees for the API to {args.out}")


if __name__ == "__main__":
    main()
