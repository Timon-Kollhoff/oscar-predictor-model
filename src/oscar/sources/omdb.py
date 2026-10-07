"""OMDb API: Metascore, Rotten Tomatoes, IMDb rating and votes -> omdb.csv."""
import json
import os
import time
from pathlib import Path
from wsgiref import headers
 
import pandas as pd
import requests
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[3]  # project folder
RAW_DIR = ROOT / "data" / "raw" / "omdb"
OUT_FILE = ROOT / "data" / "interim" / "omdb.csv"
API = "http://www.omdbapi.com/"

class OMDbMovies:
    def __init__(self, api_key:str | None = None, raw_dir: Path = RAW_DIR, out_file: Path = OUT_FILE):
        load_dotenv(ROOT / ".env")  # read the .env file so OMDb_API_KEY is available
        self.api_key = api_key or os.environ.get("OMDb_API_KEY")
        if not self.api_key:
            raise RuntimeError("OMDb_API_KEY is missing in .env")
        self.raw_dir = Path(raw_dir)
        self.out_file = Path(out_file)
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
    
    def _get(self, **params) -> dict:
        """One request to OMDb. OMDb reports errors with status 200 and "Response": "False"."""
        response = self.session.get(API, params={"apikey": self.api_key, **params}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "False":
            error = data.get("Error", "").lower()
            if "limit" in error:
                raise RuntimeError("OMDb daily limit reached. Run again tomorrow, the cache is kept.")
            if "invalid api key" in error:
                raise RuntimeError("Invalid OMDb key. Did you activate it with the link in the confirmation email?")
        time.sleep(0.1)
        return data
    
    @staticmethod
    def _clean(value: str | None) -> str | None:
        return None if value in (None, "", "N/A") else value
    
    def parse(self, imdb_id: str, json_file: Path) -> dict:
        """Extracts the features from the saved response."""
        movie = json.loads(json_file.read_text(encoding="utf-8"))
        if not movie:
            return {"imdb_id": imdb_id}
        
        ratings = {r["Source"]: r["Value"] for r in movie.get("Ratings", [])}
        votes = self._clean(movie.get("imdbVotes"))
        rating = self._clean(movie.get("imdbRating"))
        meta = self._clean(movie.get("Metascore"))
        rt = self._clean(ratings.get("Rotten Tomatoes"))
        return {
            "imdb_id": imdb_id,
            "imdb_votes": int(votes.replace(",", "")) if votes else None,   # "9,436" -> 9436
            "imdb_rating": float(rating) if rating else None,               # "8.1"   -> 8.1
            "metascore": int(meta) if meta else None,                       # "91"    -> 91
            "rotten_tomatoes": int(rt.rstrip("%")) if rt else None,         # "96%"   -> 96
        }
        
    def download(self, imdb_id: str, force: bool = False) -> Path:
        """Saves the OMDb data of a film as data/raw/omdb/<imdb_id>.json."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / f"{imdb_id}.json"
        if target.exists() and not force:
            return target
 
        found = self._get(i=imdb_id, plot="full")
        if not found:
            target.write_text("{}", encoding="utf-8")  
            return target
        target.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
        return target

    def build(self, imdb_ids) -> pd.DataFrame:
        """Downloads and parses all given films and saves omdb.csv."""
        rows = [self.parse(i, self.download(i)) for i in imdb_ids]
        df = pd.DataFrame(rows)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_file, index=False)
        return df