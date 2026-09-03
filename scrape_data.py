
#!pip install requests

#!pip install beautifulsoup4


#THIS IS A FILE MADE FOR JUPYER NOTEBOOKS TRANSLATED TO PY
#mainly as documentation purpose

import re, time, requests, pandas as pd
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup

INDEX = "http://www.darklyrics.com/o/ozzyosbourne.html"
HEADERS = {"User-Agent": "OzzyLyricsResearch/0.1 (alex; alextemocico@email.com)"}
CACHE = Path("data/cache"); CACHE.mkdir(parents=True, exist_ok=True)

def get_html(url):

    fname = CACHE / (re.sub(r"\W+", "_", url.split("darklyrics.com/")[-1]) + ".html")
    if fname.exists():
        return fname.read_text(encoding="utf-8")

    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    fname.write_text(r.text, encoding="utf-8")
    time.sleep(3)
    return r.text

"""By looking at the HTML division we see that each album has its own "div" blocks with h2 being title or year.
Each "a" tag represents the longs with their links, which we are interested about.
"""

def parse_index(url=INDEX):

    soup = BeautifulSoup(get_html(url), "lxml")
    rows = []
    for div in soup.find_all("div", class_="album"):
        m = re.search(r'"(.+?)"\s*\((\d{4})\)', div.find("h2").get_text(" ", strip=True))
        if not m:
            continue
        album, year = m.group(1), int(m.group(2))
        for a in div.find_all("a", href=True):
            num = a["href"].split("#")[-1]
            rows.append({
                "album": album,
                "year_darklyrics": year,
                "track_no": int(num) if num.isdigit() else None,
                "title": a.get_text(strip=True),
                "album_url": urljoin(url, a["href"].split("#")[0]),
            })
    return pd.DataFrame(rows)

def parse_album(album_url):
    """One album page -> [{track_no, title, lyrics}, ...]"""
    soup = BeautifulSoup(get_html(album_url), "lxml")
    body = soup.find("div", class_="lyrics")

    for br in body.find_all("br"):
        br.replace_with("\n")
    for junk in body.find_all("div"):        
        junk.decompose()

    songs, cur, buf = [], None, []
    for el in body.children:
        if getattr(el, "name", None) == "h3":
            if cur:
                cur["lyrics"] = re.sub(r"\n{3,}", "\n\n", "".join(buf)).strip()
                songs.append(cur)
            anchor = el.find("a")
            num = anchor.get("name") if anchor else None
            cur = {"track_no": int(num) if num and str(num).isdigit() else None,
                   "title": re.sub(r"^\s*\d+\.\s*", "", el.get_text(strip=True))}
            buf = []
        elif cur is not None:
            buf.append(el.get_text() if getattr(el, "name", None) else str(el))

    if cur:
        cur["lyrics"] = re.sub(r"\n{3,}", "\n\n", "".join(buf)).strip()
        songs.append(cur)
    return songs

from pathlib import Path

songs = (pd.DataFrame(rows)
         .merge(spine, on=["album_url", "track_no"], how="left",
                suffixes=("", "_index")))

# turn the title collision into a QC signal instead of a naming accident
songs["title_matches"] = (songs.title.str.lower().str.strip()
                          == songs.title_index.str.lower().str.strip())
songs["n_words"] = songs.lyrics.str.split().str.len()
songs["scraped_at"] = pd.Timestamp.utcnow().isoformat()
songs["source"] = "darklyrics"

for d in ["data/raw", "data/interim"]:
    Path(d).mkdir(parents=True, exist_ok=True)

songs.to_json("data/raw/darklyrics.jsonl", orient="records",
              lines=True, force_ascii=False)
songs.to_parquet("data/interim/songs.parquet", index=False)
songs.drop(columns=["lyrics"]).to_csv("data/interim/songs_meta.csv", index=False)

print(len(songs), "songs |", (~songs.title_matches).sum(), "title mismatches")
print(songs.groupby(["year_darklyrics", "album"])
           .agg(n=("title", "count"), med_words=("n_words", "median")))