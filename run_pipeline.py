"""Rebuilds data/processed/nominations.csv from scratch."""
from src.oscar.features import build_features, report
from src.oscar.sources.kaggle import KaggleOscars
from src.oscar.sources.omdb import OMDbMovies
from src.oscar.sources.tmdb import TMDbMovies
from src.oscar.sources.wikidata import WikidataIds
from src.oscar.sources.wikipedia import WikipediaPrecursors


oscars = KaggleOscars().build()
bp = oscars[oscars["CanonicalCategory"] == "BEST PICTURE"]
imdb_ids = bp["FilmId"].dropna().str.split("|").explode().unique()   # about 600 films

TMDbMovies().build(imdb_ids)   # -> data/interim/tmdb.csv
OMDbMovies().build(imdb_ids)   # -> data/interim/omdb.csv

pre = WikipediaPrecursors().build()
WikidataIds().build(pre["film_wiki"].dropna().unique()) 

nominations = build_features()
report(nominations)