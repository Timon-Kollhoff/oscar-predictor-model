"""TMDb API: genres, runtime, language, US release date, budget, revenue, companies -> tmdb.csv."""
import json
import os
import time
from pathlib import Path
 
import pandas as pd
import requests
from dotenv import load_dotenv
 
ROOT = Path(__file__).resolve().parents[3]  # project folder
RAW_DIR = ROOT / "data" / "raw" / "tmdb"
OUT_FILE = ROOT / "data" / "interim" / "tmdb.csv"
API = "https://api.themoviedb.org/3"

class TMDbMovies:
    def __init__(self, api_key:str | None = None, raw_dir: Path = RAW_DIR, out_file: Path = OUT_FILE):
        load_dotenv(ROOT / ".env")  # read the .env file so TMDB_API_KEY is available
        self.api_key = api_key or os.environ.get("TMDB_API_KEY")
        if not self.api_key:
            raise RuntimeError("TMDB_API_KEY is missing in .env")
        self.raw_dir = Path(raw_dir)
        self.out_file = Path(out_file)
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

    def _get(self, path: str, **params) -> dict:
        # the token is sent in the Authorization header (see __init__), never in the URL
        response = self.session.get(API + path, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(0.05)  
        return response.json()
    
    def parse(self, imdb_id: str, json_file: Path) -> dict:
        """Extracts the features from the saved response."""
        movie = json.loads(json_file.read_text(encoding="utf-8"))
        if not movie:
            return {"imdb_id": imdb_id}

        us_dates = [
            d["release_date"][:10]
            for country in movie.get("release_dates", {}).get("results", [])
            if country.get("iso_3166_1") == "US"
            for d in country.get("release_dates", [])
            if d.get("type") in (2, 3)  # 2 = theatrical (limited), 3 = theatrical
        ]
        return {
            "imdb_id": imdb_id,
            "tmdb_id": movie.get("id"),
            "genres": "|".join(g["name"] for g in movie.get("genres", [])),
            "runtime_min": movie.get("runtime") or None,
            "original_language": movie.get("original_language"),
            "us_release_date": min(us_dates, default=None),
            "budget_usd": movie.get("budget") or None,    # on TMDb, 0 means unknown
            "revenue_usd": movie.get("revenue") or None,  # worldwide, not only the US
            "companies": "|".join(c["name"] for c in movie.get("production_companies", [])),
        }
        
    def download(self, imdb_id: str, force: bool = False) -> Path:
        """Saves the TMDb data of a film as data/raw/tmdb/<imdb_id>.json."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / f"{imdb_id}.json"
        if target.exists() and not force:
            return target
 
        found = self._get(f"/find/{imdb_id}", external_source="imdb_id")["movie_results"]
        if not found:
            target.write_text("{}", encoding="utf-8")  
            return target
        movie = self._get(f"/movie/{found[0]['id']}", append_to_response="release_dates,keywords")
        target.write_text(json.dumps(movie, ensure_ascii=False), encoding="utf-8")
        return target

    def build(self, imdb_ids) -> pd.DataFrame:
        """Downloads and parses all given films and saves tmdb.csv."""
        rows = [self.parse(i, self.download(i)) for i in imdb_ids]
        df = pd.DataFrame(rows)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_file, index=False)
        return df

