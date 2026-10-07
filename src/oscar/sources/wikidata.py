"""IMDb IDs for the films in the precursor tables, via Wikipedia and Wikidata -> ids.csv."""
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]  # project folder
CACHE_FILE = ROOT / "data" / "raw" / "wikidata" / "film_ids.json"
OUT_FILE = ROOT / "data" / "interim" / "ids.csv"
 
HEADERS = {"User-Agent": "oscar-predictor/0.1 (portfolio project)"}
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
BATCH = 50  # both APIs allow at most this many titles per request

class WikidataIds:
    """Looks up the Wikidata and IMDb IDs of Wikipedia film articles -> ids.csv."""

    def __init__(self, cache_file: Path = CACHE_FILE, out_file: Path = OUT_FILE):
            self.cache_file = Path(cache_file)
            self.out_file = Path(out_file)
            self.session = requests.Session()
            self.session.headers.update(HEADERS)

    
    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        return {}
    
    def _get(self, url: str, **params) -> dict:
        """GET request that waits automatically on 429 (Too Many Requests)."""
        for attempt in range(5):
            response = self.session.get(url, params={"format": "json", **params}, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else 10 * (attempt + 1)
                print(f"429: waiting {wait} seconds (attempt {attempt + 1}/5)")
                time.sleep(wait)
                continue
            response.raise_for_status()
            time.sleep(1)   # was 0.5: a bit more time between requests
            return response.json()
        response.raise_for_status() 
    
    def lookup(self, titles: list[str]) -> dict:
        """Up to 50 titles at once: {title: {"wikidata_id": ..., "imdb_id": ...}}"""
        # 1. Wikipedia: article -> Wikidata ID
        query = self._get(
            WIKIPEDIA_API, action="query", redirects=1, prop="pageprops", ppprop="wikibase_item",
            titles="|".join(t.replace("_", " ") for t in titles),
        )["query"]
        # Wikipedia sometimes rewrites titles (capitalization, redirects)
        renamed = {r["from"]: r["to"] for r in query.get("normalized", []) + query.get("redirects", [])}
        qid_by_page = {p["title"]: p.get("pageprops", {}).get("wikibase_item")
                       for p in query.get("pages", {}).values()}
 
        qids = {}
        for title in titles:
            name = title.replace("_", " ")
            name = renamed.get(name, name)  # normalization
            name = renamed.get(name, name)  # redirect
            qids[title] = qid_by_page.get(name)
 
        # 2. Wikidata: Wikidata ID -> IMDb ID
        imdb = {}
        known = sorted({q for q in qids.values() if q})
        if known:
            entities = self._get(WIKIDATA_API, action="wbgetentities", ids="|".join(known),
                                 props="claims")["entities"]
            for qid, entity in entities.items():
                values = [c["mainsnak"].get("datavalue", {}).get("value")
                          for c in entity.get("claims", {}).get("P345", [])]
                imdb[qid] = next((v for v in values if v and v.startswith("tt")), None)
 
        return {t: {"wikidata_id": qids[t], "imdb_id": imdb.get(qids[t])} for t in titles}
    
    def build(self, wiki_titles) -> pd.DataFrame:
        """Looks up all titles that are not cached yet and saves ids.csv."""
        titles = sorted({t for t in wiki_titles if isinstance(t, str) and t})
        cache = self._load_cache()
        todo = [t for t in titles if t not in cache]
        for start in range(0, len(todo), BATCH):
            cache.update(self.lookup(todo[start:start + BATCH]))
            self._save_cache(cache)  # save after every batch in case something fails
 
        df = pd.DataFrame([{"film_wiki": t, **cache[t]} for t in titles])
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_file, index=False)
        return df
    
    def _save_cache(self, cache: dict) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")