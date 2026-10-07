"""API data formats: what comes in and what goes out."""

from pydantic import BaseModel, Field


class Features(BaseModel):
    """Which guild awards a film won."""
    pga_winner: bool = False
    dga_winner: bool = False
    sag_winner: bool = False


class Movie(BaseModel):
    title: str
    features: Features


class PredictMoviesRequest(BaseModel):
    movies: list[Movie] = Field(min_length=2, max_length=10,
                                description="The nominees of one ceremony (2 to 10 films)")


class MoviePrediction(BaseModel):
    title: str
    probability: float = Field(description="Chance of winning; adds up to 1 across all films")
    pick: bool = Field(description="The model's pick")


class PredictMoviesResponse(BaseModel):
    predictions: list[MoviePrediction]
    warning: str | None = None


class YearPrediction(MoviePrediction):
    imdb_id: str | None = None
    won: bool | None = Field(description="Did the film win? None if not decided yet")
    features: dict[str, bool | None]


class YearPredictionResponse(BaseModel):
    year: int
    pick: str | None
    winner: str | None
    correct: bool | None
    in_training: bool = Field(description="The year was part of the training data, so the pick is "
                                          "not an honest test. Honest numbers: walk-forward in the notebook")
    predictions: list[YearPrediction]
    warning: str | None = None
