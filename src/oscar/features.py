"""Join the interim tables and compute features -> nominations.csv.

Inputs (data/interim/):
    oscars.csv      KaggleOscars        one row per Oscar nomination (all categories)
    precursors.csv  WikipediaPrecursors award, year_film, film, film_wiki, won
    ids.csv         WikidataIds         film_wiki -> imdb_id
    tmdb.csv        TMDbMovies          imdb_id + metadata
    omdb.csv        OMDbMovies          imdb_id + critics' scores

Output (data/processed/nominations.csv):
    one row per nomination in ONE Oscar category (default: Best Picture),
    key imdb_id, target variable won.
"""

from __future__ import annotations



from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project folder
INTERIM = ROOT / "data" / "interim"
OUT_FILE = ROOT / "data" / "processed" / "nominations.csv"

# The Golden Globes have two film categories; for the model they count as one award
AWARD_GROUPS = {"gg_drama": "gg", "gg_comedy": "gg"}

COMPETITIVE_EXCLUDE = {"Special", "SciTech"}   # don't count honorary and technical awards

# Up to film year 1999 the BAFTAs took place AFTER the Oscars -> leakage, so leave them empty.
# (From the 2001 ceremony on they come first. Worth double-checking against Wikipedia.)
BAFTA_BEFORE_OSCARS_FROM = 2000

SCREENPLAY = {"WRITING (Original Screenplay)", "WRITING (Adapted Screenplay)"}
ACTING = {"ACTOR IN A LEADING ROLE", "ACTRESS IN A LEADING ROLE",
          "ACTOR IN A SUPPORTING ROLE", "ACTRESS IN A SUPPORTING ROLE"}


def load_interim(folder: Path = INTERIM) -> dict[str, pd.DataFrame]:
    """Reads all interim tables."""
    names = ["oscars", "precursors", "ids", "tmdb", "omdb"]
    return {n: pd.read_csv(folder / f"{n}.csv") for n in names}


def _explode_film_ids(oscars: pd.DataFrame) -> pd.DataFrame:
    """One row per (nomination, film). In early years FilmId holds several IDs separated by |."""
    df = oscars.dropna(subset=["FilmId"]).copy()
    df["imdb_id"] = df["FilmId"].str.split("|")
    return df.explode("imdb_id")


def oscar_context(oscars: pd.DataFrame) -> pd.DataFrame:
    """Which other NOMINATIONS the film has at the same ceremony.
    Deliberately no wins in other categories: those are only known after the ceremony (leakage)."""
    df = _explode_film_ids(oscars)
    cat = df["CanonicalCategory"]
    df["is_director"] = (cat == "DIRECTING").astype(int)
    df["is_editing"] = (cat == "FILM EDITING").astype(int)
    df["is_screenplay"] = cat.isin(SCREENPLAY).astype(int)
    df["is_acting"] = cat.isin(ACTING).astype(int)

    return (df.groupby(["Ceremony", "imdb_id"])
              .agg(oscar_total_noms=("CanonicalCategory", "size"),
                   has_director_nom=("is_director", "max"),
                   has_editing_nom=("is_editing", "max"),
                   has_screenplay_nom=("is_screenplay", "max"),
                   n_acting_noms=("is_acting", "sum"))
              .reset_index())


def precursor_features(precursors: pd.DataFrame, ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Long -> wide: one row per imdb_id with pga_nom, pga_win, dga_nom, ...
    Also returns the first film year of each award (it did not exist before)."""
    pre = precursors.merge(ids[["film_wiki", "imdb_id"]], on="film_wiki", how="left")
    pre["award"] = pre["award"].replace(AWARD_GROUPS)
    first_year = pre.groupby("award")["year_film"].min()

    pre = pre.dropna(subset=["imdb_id"])
    wide = (pre.assign(nom=1)
               .pivot_table(index="imdb_id", columns="award", values=["nom", "won"],
                            aggfunc="max", fill_value=0))
    wide.columns = [f"{award}_{'win' if value == 'won' else 'nom'}" for value, award in wide.columns]
    return wide.reset_index(), first_year


def metadata_features(tmdb: pd.DataFrame, omdb: pd.DataFrame) -> pd.DataFrame:
    """TMDb and OMDb merged into one row per imdb_id, plus a few derived columns."""
    meta = (tmdb.drop_duplicates("imdb_id")
                .merge(omdb.drop_duplicates("imdb_id"), on="imdb_id", how="outer"))

    meta["release_month"] = pd.to_datetime(meta["us_release_date"], errors="coerce").dt.month

    # Derived 0/1 columns only for films TMDb knows; otherwise empty instead of a wrong 0
    has_tmdb = meta["tmdb_id"].notna()
    genres = meta["genres"].fillna("")
    flags = {
        "is_english": meta["original_language"] == "en",
        "genre_drama": genres.str.contains("Drama"),
        "genre_comedy": genres.str.contains("Comedy"),
    }
    for name, values in flags.items():
        meta[name] = values.astype(float).where(has_tmdb)
    return meta


def person_history(oscars: pd.DataFrame) -> pd.DataFrame:
    """Per person and ceremony: nominations and wins at all EARLIER ceremonies."""
    df = oscars[~oscars["Class"].isin(COMPETITIVE_EXCLUDE)].dropna(subset=["NomineeIds"]).copy()
    df["person_id"] = df["NomineeIds"].str.split("|")
    df = df.explode("person_id")
    df = df[df["person_id"].str.startswith("nm")]          # ignore companies (co...)
    per = (df.groupby(["person_id", "Ceremony"])
             .agg(noms=("won", "size"), wins=("won", "sum"))
             .reset_index()
             .sort_values(["person_id", "Ceremony"]))
    g = per.groupby("person_id")
    per["prev_oscar_noms"] = g["noms"].cumsum() - per["noms"]   # running total minus the current ceremony
    per["prev_oscar_wins"] = g["wins"].cumsum() - per["wins"]
    return per[["person_id", "Ceremony", "prev_oscar_noms", "prev_oscar_wins"]]


def nominee_history(oscars: pd.DataFrame, category: str = "BEST PICTURE") -> pd.DataFrame:
    """Nominee experience per film: the most experienced nominee counts (max)."""
    rows = oscars[oscars["CanonicalCategory"] == category].dropna(subset=["NomineeIds"]).copy()
    rows["imdb_id"] = rows["FilmId"].str.split("|").str[0]
    rows["person_id"] = rows["NomineeIds"].str.split("|")
    rows = rows.explode("person_id").merge(person_history(oscars), on=["person_id", "Ceremony"], how="left")
    return (rows.groupby(["Ceremony", "imdb_id"])[["prev_oscar_noms", "prev_oscar_wins"]]
                .max().reset_index())

def build_features(data: dict[str, pd.DataFrame] | None = None,
                   category: str = "BEST PICTURE",
                   out_file: Path = OUT_FILE) -> pd.DataFrame:
    """Joins all sources and saves nominations.csv."""
    data = data or load_interim()
    oscars = data["oscars"]

    # 1. Base: all nominations in the category, one row per film
    base = oscars[oscars["CanonicalCategory"] == category].copy()
    base["imdb_id"] = base["FilmId"].str.split("|").str[0]
    base = base[["year_ceremony", "year_film", "Ceremony", "Film", "imdb_id", "won"]]
    base["n_nominees"] = base.groupby("Ceremony")["Film"].transform("size")

    # 2. Oscar context (other nominations at the same ceremony)
    df = base.merge(oscar_context(oscars), on=["Ceremony", "imdb_id"], how="left")

    # 3. Past Oscar nominations and wins of the nominees (leakage: earlier ceremonies only)
    df = df.merge(nominee_history(oscars, category), on=["Ceremony", "imdb_id"], how="left")

    # 4. Precursor awards
    wide, first_year = precursor_features(data["precursors"], data["ids"])
    df = df.merge(wide, on="imdb_id", how="left")
    for award, year in first_year.items():
        cols = [c for c in (f"{award}_nom", f"{award}_win") if c in df.columns]
        exists = df["year_film"] >= year
        df.loc[exists, cols] = df.loc[exists, cols].fillna(0)   # award existed: no nomination = 0
        df.loc[~exists, cols] = np.nan                           # award did not exist yet: empty
    bafta_cols = [c for c in ("bafta_nom", "bafta_win") if c in df.columns]
    df.loc[df["year_film"] < BAFTA_BEFORE_OSCARS_FROM, bafta_cols] = np.nan
    win_cols = [c for c in df.columns if c.endswith("_win")]
    df["precursor_wins"] = df[win_cols].sum(axis=1, min_count=1)

    # 5. Metadata and reviews
    df = df.merge(metadata_features(data["tmdb"], data["omdb"]), on="imdb_id", how="left")

    df = df.sort_values(["year_ceremony", "won"], ascending=[True, False]).reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return df


def report(df: pd.DataFrame, from_year: int = 1995) -> None:
    """Shows how well the joins worked."""
    recent = df[df["year_film"] >= from_year]
    nom_cols = [c for c in df.columns if c.endswith("_nom") and c != "has_director_nom"]
    print(f"{len(df)} nominations, {len(recent)} of them from film year {from_year}")
    print(f"without IMDb ID:       {df['imdb_id'].isna().sum()}")
    print(f"with TMDb data:        {df['tmdb_id'].notna().mean():.0%}" if "tmdb_id" in df else "")
    print(f"from {from_year} without any precursor nomination (check the join):")
    no_pre = recent[recent[nom_cols].fillna(0).sum(axis=1) == 0]
    print(no_pre[["year_film", "Film", "imdb_id"]].to_string(index=False) if len(no_pre) else "  none")


if __name__ == "__main__":
    nominations = build_features()
    report(nominations)