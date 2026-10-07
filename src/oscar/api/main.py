"""REST API for the Best Picture prediction. See README.md in this folder.

Train once first:  uv run python -m oscar.train
Start:             uv run uvicorn oscar.api.main:app --reload
Try it out:        http://127.0.0.1:8000/docs
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from oscar.api.schemas import (MoviePrediction, PredictMoviesRequest, PredictMoviesResponse,
                               YearPrediction, YearPredictionResponse)
from oscar.api.controller import get_model, get_nominations, probabilities, has_signal, FEATURE_MAP, NO_SIGNAL
from oscar.model import TARGET, YEAR


app = FastAPI(title="Oscar Predictor",
              description="Predicts the Oscar for Best Picture from the PGA, DGA and SAG awards.")

@app.get("/model")
def model_info() -> dict:
    """Which model is running: features, weights, training years and the honest test result."""
    _, meta = get_model()
    return meta


@app.get("/prediction/{year}", response_model=YearPredictionResponse)
def predict_year(year: int):
    """Chances of winning for the nominees of one ceremony in the data (year of the ceremony)."""
    model, meta = get_model()
    first, last = meta["train_ceremonies"]
    if year < first:
        raise HTTPException(404, f"The model only covers ceremonies from {first} on.")
    films = get_nominations()
    films = films[films[YEAR] == year].reset_index(drop=True)
    if films.empty:
        raise HTTPException(404, f"No nominations for the {year} ceremony in the data.")

    p = probabilities(model, meta, films)
    signal = has_signal(films[meta["features"]])
    best = int(np.argmax(p))
    winners = films.loc[films[TARGET] == 1, "Film"]
    winner = winners.iloc[0] if len(winners) == 1 else None
    pick = films.loc[best, "Film"] if signal else None

    predictions = [
        YearPrediction(
            title=row["Film"],
            imdb_id=row["imdb_id"] if pd.notna(row["imdb_id"]) else None,
            probability=round(float(p[i]), 3),
            pick=signal and i == best,
            won=None if pd.isna(row[TARGET]) else bool(row[TARGET]),
            features={field: None if pd.isna(row[col]) else bool(row[col])
                      for field, col in FEATURE_MAP.items()},
        )
        for i, row in films.iterrows()
    ]
    predictions.sort(key=lambda m: m.probability, reverse=True)

    return YearPredictionResponse(
        year=year,
        pick=pick,
        winner=winner,
        correct=None if pick is None or winner is None else pick == winner,
        in_training=winner is not None and first <= year <= last,
        predictions=predictions,
        warning=None if signal else NO_SIGNAL,
    )


@app.post("/predict", response_model=PredictMoviesResponse)
def predict_movies(request: PredictMoviesRequest):
    """Chances of winning for your own films, e.g. for the next ceremony."""
    model, meta = get_model()
    X = pd.DataFrame([{col: int(getattr(movie.features, field)) for field, col in FEATURE_MAP.items()}
                      for movie in request.movies])
    p = probabilities(model, meta, X)
    signal = has_signal(X)
    best = int(np.argmax(p))

    predictions = [
        MoviePrediction(title=movie.title, probability=round(float(p[i]), 3), pick=signal and i == best)
        for i, movie in enumerate(request.movies)
    ]
    predictions.sort(key=lambda m: m.probability, reverse=True)
    return PredictMoviesResponse(predictions=predictions, warning=None if signal else NO_SIGNAL)
