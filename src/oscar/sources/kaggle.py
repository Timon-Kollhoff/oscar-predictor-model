"""Load Oscar nominations and winners from the Kaggle dataset -> oscars.csv."""

from __future__ import annotations

import csv
import shutil
import warnings
from pathlib import Path

import pandas as pd

DATASET = "unanimad/the-oscar-award"
FULL_FILE = "full_data.csv"  # extended file of the dataset (source: github.com/DLu/oscar_data)
ROOT = Path(__file__).resolve().parents[3]  # project folder
RAW_DIR = ROOT / "data" / "raw" / "kaggle"
OUT_FILE = ROOT / "data" / "interim" / "oscars.csv"

# All columns of full_data.csv, in original order.
# Multiple values in one field are separated by "|".
COLUMNS = [
    "Ceremony",           # ceremony number (1 = 1928)
    "Year",               # film year, in the early years e.g. "1927/28"
    "Class",              # Acting, Directing, Writing, Title, Production, Music, SciTech, Special
    "CanonicalCategory",  # consistent category name across all years
    "Category",           # category name used in that year
    "Film",               # film title
    "FilmId",             # IMDb ID of the film (tt...)
    "Name",               # original text of the nomination
    "Nominees",           # nominee names only
    "NomineeIds",         # IMDb IDs of the nominees (nm..., or co... for companies)
    "Winner",             # "True" or empty
    "Detail",             # e.g. role or song name
    "Note",               # notes by the Academy
    "Citation",           # citation for honorary and technical awards
]
# Additional computed columns
EXTRA_COLUMNS = [
    "year_film",      # film year as a number (first four digits of Year)
    "year_ceremony",  # ceremony year, counted as in the_oscar_award.csv
    "won",            # target variable 0/1
]

# The categories we want to predict (names as in CanonicalCategory)
MAIN_CATEGORIES = [
    "BEST PICTURE",
    "DIRECTING",
    "ACTOR IN A LEADING ROLE",
    "ACTRESS IN A LEADING ROLE",
    "ACTOR IN A SUPPORTING ROLE",
    "ACTRESS IN A SUPPORTING ROLE",
    "WRITING (Original Screenplay)",
    "WRITING (Adapted Screenplay)",
]


class KaggleOscars:
    """Loads full_data.csv from the Kaggle dataset 'The Oscar Award' with all columns
    (all years, all categories) and saves it as oscars.csv.

    Example:
        oscars = KaggleOscars().build()
        main = oscars[oscars["CanonicalCategory"].isin(MAIN_CATEGORIES)]
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        out_file: Path = OUT_FILE,
        min_ceremony_year: int | None = None,  # None = all years from 1928
    ):
        self.raw_dir = Path(raw_dir)
        self.out_file = Path(out_file)
        self.min_ceremony_year = min_ceremony_year

    def download(self, force: bool = False) -> Path:
        """Downloads the dataset once to data/raw/kaggle (both CSV files).
        After that only the cache is read."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / FULL_FILE
        if target.exists() and not force:
            self._to_comma_csv(target)
            return target

        try:
            import kagglehub
        except ImportError as err:
            raise RuntimeError(
                "kagglehub is missing: 'pip install kagglehub'. Alternatively, download full_data.csv from "
                f"https://www.kaggle.com/datasets/{DATASET} and put it in {self.raw_dir}."
            ) from err

        source_dir = Path(kagglehub.dataset_download(DATASET, force_download=force))
        if not (source_dir / FULL_FILE).exists():
            raise FileNotFoundError(f"{FULL_FILE} is missing from the Kaggle download: {source_dir}")
        for csv_file in source_dir.glob("*.csv"):
            shutil.copy(csv_file, self.raw_dir / csv_file.name)
        self._to_comma_csv(target)
        return target

    @staticmethod
    def _to_comma_csv(path: Path) -> None:
        """On Kaggle, full_data.csv is tab-separated despite its extension. We save it as
        a real comma-separated CSV. The values stay unchanged."""
        with open(path, encoding="utf-8") as f:
            if "\t" not in f.readline():
                return  # already a comma-separated CSV
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE, encoding="utf-8")
        df.to_csv(path, index=False, encoding="utf-8")

    def load_raw(self) -> pd.DataFrame:
        """Reads full_data.csv unchanged (all values as text)."""
        path = self.download()
        with open(path, encoding="utf-8") as f:
            header = f.readline()
        sep = "\t" if "\t" in header else ","  # the original from Kaggle is tab-separated
        df = pd.read_csv(
            path,
            sep=sep,
            dtype=str,
            keep_default_na=False,
            quoting=csv.QUOTE_NONE if sep == "\t" else csv.QUOTE_MINIMAL,
            encoding="utf-8",
        )
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Columns missing in {path.name}: {missing}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keeps all rows and all original columns and adds year_film,
        year_ceremony and won. Empty fields become NA."""
        df = df.copy()
        won = df["Winner"].str.strip().eq("True").astype(int)
        for col in COLUMNS:
            df[col] = df[col].str.strip().replace("", pd.NA)

        df["Ceremony"] = df["Ceremony"].astype(int)
        df["year_film"] = df["Year"].str[:4].astype(int)
        df["year_ceremony"] = df["year_film"] + 1
        df["won"] = won

        if self.min_ceremony_year is not None:
            df = df[df["year_ceremony"] >= self.min_ceremony_year]
        return df[COLUMNS + EXTRA_COLUMNS].reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        """Full run: load, clean, check, save as oscars.csv."""
        df = self.clean(self.load_raw())
        self._check(df)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_file, index=False)
        return df

    @staticmethod
    def _check(df: pd.DataFrame) -> None:
        """Warns when a main category does not have exactly one winner in a year
        (e.g. the real ties in 1932 and 1969)."""
        main = df[df["CanonicalCategory"].isin(MAIN_CATEGORIES)]
        winners = main.groupby(["year_ceremony", "CanonicalCategory"])["won"].sum()
        odd = winners[winners != 1]
        if not odd.empty:
            warnings.warn("Main categories without exactly one winner:\n" + odd.to_string())


if __name__ == "__main__":
    oscars = KaggleOscars().build()
    print(f"{len(oscars)} nominations ({oscars['year_ceremony'].min()}-"
          f"{oscars['year_ceremony'].max()}), {oscars.shape[1]} columns -> {OUT_FILE}")
