"""Training and evaluation for Best Picture.

Core idea: every ceremony has exactly one winner. The model gives each nominee a
probability, and the film with the highest probability is the pick (`pick_winner`).
All evaluations split by whole ceremony years, never by single rows.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

YEAR = "year_ceremony"
TARGET = "won"
ID_COLS = ["year_ceremony", "year_film", "Ceremony", "Film", "imdb_id", "tmdb_id", "won"]

# Feature sets compared in the notebook
FEATURE_SETS = {
    "Study (Butcher & Abbass)": ["dga_win", "pga_win", "oscar_total_noms", "has_director_nom",
                                 "has_editing_nom", "genre_drama", "gg_win"],
    "PGA+DGA+SAG": ["pga_win", "dga_win", "sag_win"],
    "Guilds + BAFTA/CC": ["pga_win", "dga_win", "sag_win", "bafta_win", "cc_win"],
    "CORE": ["pga_win", "dga_win", "sag_win", "bafta_win", "cc_win",
             "oscar_total_noms", "has_director_nom", "has_editing_nom"],
}

# Colors for the plots
BLUE, RED, INK, MUTED, GRID = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#e5e4e0"


def numeric_features(df: pd.DataFrame) -> list[str]:
    """All numeric columns except IDs, years and the target."""
    return [c for c in df.select_dtypes("number").columns if c not in ID_COLS]


def make_model(C: float = 1.0):
    """Logistic regression with median imputation and standardization."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=1000))


def years_with_winner(data: pd.DataFrame, first: int, last: int | None = None) -> list[int]:
    """Ceremony years in the range with exactly one known winner."""
    last = last if last is not None else int(data[YEAR].max())
    winners = data.groupby(YEAR)[TARGET].sum()
    return [int(y) for y, n in winners.items() if first <= y <= last and n == 1]


# ---------------------------------------------------------------- Predictions
def pick_winner(df: pd.DataFrame, score: str = "p") -> pd.Series:
    """Exactly one pick per year: the film with the highest score (the first one on a tie)."""
    rank = df.groupby(YEAR)[score].rank(ascending=False, method="first")
    return (rank == 1).astype(int)


def predict_split(model, train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                  sample_weight: np.ndarray | None = None) -> pd.DataFrame:
    """Trains a fresh copy of the model and returns the test rows with p and pred_win."""
    m = clone(model)
    m.fit(train[features], train[TARGET], logisticregression__sample_weight=sample_weight)
    out = test.copy()
    out["p"] = m.predict_proba(test[features])[:, 1]
    out["p"] = out["p"] / out.groupby(YEAR)["p"].transform("sum")  # normalize to 100 % per year
    out["pred_win"] = pick_winner(out)
    return out


def walk_forward(model, data: pd.DataFrame, features: list[str], first: int,
                 last: int | None = None, half_life: float | None = None) -> pd.DataFrame:
    """Every test year is predicted only from earlier years (a realistic forecast).
    half_life: weight recent years more, e.g. 10 = a year 10 years back counts half."""
    parts = []
    for year in years_with_winner(data, first, last):
        train = data[data[YEAR] < year]
        weights = None
        if half_life is not None and not pd.isna(half_life):
            weights = (0.5 ** ((year - train[YEAR]) / half_life)).to_numpy()
        parts.append(predict_split(model, train, data[data[YEAR] == year], features, weights))
    return pd.concat(parts)


def group_kfold(model, data: pd.DataFrame, features: list[str], n_splits: int = 5) -> pd.DataFrame:
    """Cross-validation by ceremony year: every year is tested exactly once."""
    splits = GroupKFold(n_splits=n_splits).split(data, data[TARGET], data[YEAR])
    return pd.concat([predict_split(model, data.iloc[tr], data.iloc[te], features) for tr, te in splits])


def group_holdout(model, data: pd.DataFrame, features: list[str], test_size: float = 0.2,
                  seed: int = 42) -> pd.DataFrame:
    """A single random split by ceremony year (as in the study)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(data, data[TARGET], data[YEAR]))
    return predict_split(model, data.iloc[tr], data.iloc[te], features)


def rule_baseline(data: pd.DataFrame, column: str, years: list[int]) -> pd.DataFrame:
    """Baseline without a model: the film with the highest value in `column` wins."""
    out = data[data[YEAR].isin(years)].copy()
    out["p"] = out[column].fillna(0)
    out["pred_win"] = pick_winner(out)
    return out


# ---------------------------------------------------------------- Metrics
def hits(pred: pd.DataFrame) -> int:
    """Number of years in which the pick was correct."""
    return int(pred.loc[pred[TARGET] == 1, "pred_win"].sum())


def metrics(name: str, pred: pd.DataFrame) -> dict:
    """All metrics for a set of predictions.
    With one pick per year, top-1, precision, recall and F1 are identical."""
    y, yhat = pred[TARGET], pred["pred_win"]
    n_years = pred[YEAR].nunique()
    return {
        "Method": name,
        "Hits": f"{hits(pred)}/{n_years}",
        "Top-1": hits(pred) / n_years,
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat, zero_division=0),
        "ROC-AUC": roc_auc_score(y, pred["p"]),
        "Accuracy (rows)": accuracy_score(y, yhat),
        "Accuracy 'all lose'": 1 - y.mean(),
    }


def indicator_hit_rates(data: pd.DataFrame, columns: list[str], split_year: int = 2010) -> pd.DataFrame:
    """How often the winner of a precursor award also won Best Picture, before and from split_year."""
    periods = {f"until {split_year - 1}": data[data[YEAR] < split_year],
               f"from {split_year}": data[data[YEAR] >= split_year]}
    rows = {}
    for col in columns:
        row = {}
        for label, part in periods.items():
            part = part[part[col].notna()]
            n_years = part[YEAR].nunique()
            n_hits = int(part.loc[part[col] == 1, TARGET].sum())
            row[label] = f"{n_hits}/{n_years} ({n_hits / n_years:.0%})" if n_years else "–"
        rows[col.replace("_win", "").upper()] = row
    return pd.DataFrame(rows).T


def seed_stability(model, data: pd.DataFrame, features: list[str], n_seeds: int = 50,
                   test_size: float = 0.2) -> pd.Series:
    """Top-1 for many random 80/20 splits: shows how much a single split varies."""
    scores = []
    for seed in range(n_seeds):
        pred = group_holdout(model, data, features, test_size=test_size, seed=seed)
        scores.append(hits(pred) / pred[YEAR].nunique())
    return pd.Series(scores, name="Top-1")


def yearly_picks(pred: pd.DataFrame, baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per year: the model's pick, the real winner and (optionally) the baseline's pick."""
    tip = pred[pred["pred_win"] == 1].set_index(YEAR)
    real = pred[pred[TARGET] == 1].set_index(YEAR)
    out = pd.DataFrame({"Winner": real["Film"], "Model pick": tip["Film"],
                        "p winner": real["p"].round(2)})
    out["Model correct"] = out["Winner"] == out["Model pick"]
    if baseline is not None:
        base_tip = baseline[baseline["pred_win"] == 1].set_index(YEAR)["Film"]
        out["PGA pick"] = base_tip
        out["PGA correct"] = out["Winner"] == out["PGA pick"]
    return out


# ---------------------------------------------------------------- Plots
def _style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def plot_split_distribution(scores: pd.Series, reference: float = 0.85,
                            reference_label: str = "Study"):
    """How often each top-1 value came out across many random splits."""
    counts = scores.round(2).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(counts.index, counts.values, width=0.035, color=BLUE)
    top = counts.max() * 1.3
    ax.set_ylim(0, top)
    ax.axvline(scores.mean(), color=INK, linewidth=1.5)
    ax.axvline(reference, color=MUTED, linewidth=1.5, linestyle="--")
    ax.text(scores.mean() - 0.01, top * 0.97, f"Mean {_pct(scores.mean())}", color=INK,
            va="top", ha="right")
    ax.text(reference + 0.01, top * 0.97, f"{reference_label}: {_pct(reference)}", color=MUTED,
            va="top", ha="left")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _pct(v)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Top-1 on the test set (share of correctly predicted years)", color=MUTED)
    ax.set_ylabel("Number of splits", color=MUTED)
    ax.set_title(f"Same model, {len(scores)} random 80/20 splits by ceremony year",
                 loc="left", color=INK, fontsize=11)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_coefficients(model, data: pd.DataFrame, features: list[str], title: str):
    """Standardized coefficients: blue raises, red lowers the chance of winning."""
    m = clone(model).fit(data[features], data[TARGET])
    coefs = pd.Series(m[-1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(coefs) + 1.2))
    ax.barh(coefs.index, coefs.values, height=0.6, color=[BLUE if v > 0 else RED for v in coefs])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Weight (standardized)", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _style(ax)
    for label in ax.get_yticklabels():
        label.set_color(INK)
    fig.tight_layout()
    return fig, coefs
