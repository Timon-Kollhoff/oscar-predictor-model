"""Best Picture precursor awards (PGA, DGA, SAG, BAFTA, Golden Globes, Critics' Choice) from Wikipedia -> precursors.csv."""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]  # project folder
RAW_DIR = ROOT / "data" / "raw" / "wikipedia"
OUT_FILE = ROOT / "data" / "interim" / "precursors.csv"

HEADERS = {"User-Agent": "oscar-predictor/0.1 (portfolio project)"}
WIKI = "https://en.wikipedia.org/wiki/"

WIKI_URLS = {
    "pga": WIKI + "Producers_Guild_of_America_Award_for_Best_Theatrical_Motion_Picture",
    "dga": WIKI + "Directors_Guild_of_America_Award_for_Outstanding_Directing_–_Feature_Film",
    "sag": WIKI + "Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Cast_in_a_Motion_Picture",
    "bafta": WIKI + "BAFTA_Award_for_Best_Film",
    "gg_drama": WIKI + "Golden_Globe_Award_for_Best_Motion_Picture_–_Drama",
    "gg_comedy": WIKI + "Golden_Globe_Award_for_Best_Motion_Picture_–_Musical_or_Comedy",
    "cc": WIKI + "Critics'_Choice_Movie_Award_for_Best_Picture",
}


class WikipediaPrecursors:
    """Downloads the Wikipedia pages of the precursor awards and builds one table from them:
    one row per award, year and nominated film, with `won` = 0/1.

    Example:
        precursors = WikipediaPrecursors().build()
    """

    def __init__(self, raw_dir: Path = RAW_DIR, out_file: Path = OUT_FILE):
        self.raw_dir = Path(raw_dir)
        self.out_file = Path(out_file)

    def download(self, award: str, url: str, force: bool = False) -> Path:
        """Saves the page once as data/raw/wikipedia/<award>.html.
        After that only the saved file is read."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / f"{award}.html"
        if target.exists() and not force:
            return target

        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        # save the page as HTML so it never has to be downloaded again
        target.write_text(response.text, encoding="utf-8")
        time.sleep(1)  # don't flood Wikipedia with requests
        return target

    def parse(self, award: str, html_file: Path) -> pd.DataFrame:
        """Reads all year tables of a page: year, film, link and whether the film won."""
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "lxml")
        for ref in soup.select("sup.reference"):  # remove footnotes like [12]
            ref.decompose()

        records = []
        for table in soup.select("table.wikitable"):
            grid = self._grid(table)
            if not grid:
                continue
            header = [cell.get_text(" ", strip=True).lower() for cell in grid[0]]
            film_col = next((i for i, h in enumerate(header) if h.startswith("film")), None)
            if not header[0].startswith("year") or film_col is None:
                continue  # e.g. statistics tables such as "most wins"

            for row in grid[1:]:
                year = re.search(r"\b(19|20)\d{2}\b", row[0].get_text(" ", strip=True))
                if year is None or len(row) <= film_col:
                    continue  # subheadings like "Best Film from Any Source"
                cell = row[film_col]
                link = cell.find("a", href=True)
                title = link.get_text() if link else cell.get_text(" ", strip=True)
                records.append({
                    "award": award,
                    "year_film": int(year.group()),
                    "film": self._clean(title),
                    "film_wiki": self._wiki_title(link),
                    "won": int(self._is_winner(cell)),
                })

        df = pd.DataFrame(records)
        # A film can appear more than once (e.g. with two directors): keep one row per film
        return (df.groupby(["award", "year_film", "film", "film_wiki"], dropna=False, as_index=False)["won"]
                  .max())

    def build(self) -> pd.DataFrame:
        """Full run: download all pages, parse them, save as precursors.csv."""
        frames = []
        for award, url in WIKI_URLS.items():
            # save the HTML for each award
            html_file = self.download(award, url)

            # build the table from the HTML file
            frames.append(self.parse(award, html_file))

        # combine all tables
        df = pd.concat(frames, ignore_index=True)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)

        # save as CSV
        df.to_csv(self.out_file, index=False)
        return df

    @staticmethod
    def _grid(table) -> list[list]:
        """Turns an HTML table into a grid in which every row has all columns.
        Cells with rowspan/colspan (e.g. the year spanning 5 nominees) are copied."""
        grid, carry = [], {}  # carry: column -> (cell, rows still to fill)
        for tr in table.find_all("tr"):
            cells = iter(tr.find_all(["td", "th"], recursive=False))
            row, col = [], 0
            while True:
                if col in carry:  # cell from a row further up (rowspan)
                    cell, left = carry.pop(col)
                    if left > 1:
                        carry[col] = (cell, left - 1)
                    row.append(cell)
                    col += 1
                    continue
                cell = next(cells, None)
                if cell is None:
                    break
                rowspan = int(cell.get("rowspan", 1) or 1)
                for _ in range(int(cell.get("colspan", 1) or 1)):
                    if rowspan > 1:
                        carry[col] = (cell, rowspan - 1)
                    row.append(cell)
                    col += 1
            grid.append(row)
        return grid

    @staticmethod
    def _is_winner(cell) -> bool:
        """Winners have a colored background on Wikipedia
        (yellow #FAEB86, blue #b0c4de for the Golden Globes)."""
        return "background" in (cell.get("style") or "").lower()

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\[[^\]]*\]|[†‡§*¤]", "", text)  # footnotes and markers
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _wiki_title(link) -> str | None:
        """Title of the Wikipedia article, e.g. 'Conclave_(film)'. More unique than the film name."""
        if link is None:
            return None
        href = link["href"]
        if "/wiki/" not in href or "redlink" in href:
            return None
        return unquote(href.split("/wiki/")[-1].split("#")[0])


if __name__ == "__main__":
    precursors = WikipediaPrecursors().build()
    print(precursors.groupby("award")[["year_film"]].agg(["min", "max", "count"]))