"""Clean the scraped darklyrics JSONL into plain text an LLM can read.

Just press Run
"""

# this script was generated mostly by claude and refined through multiple attempts until
# the produced output was correct

import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFILE = HERE / "Data" / "darklyrics.jsonl"
OUTFILE = HERE / "Data" / "darklyrics_clean.jsonl"
CORPUS = HERE / "Data" / "corpus.txt"
ALBUMS_FILE = HERE / "Data" / "kb" / "albums.json"

# Ozzy originals that only appear outside the studio albums. title -> (album, year)
KEEP_ANYWAY = {
    "Under the Graveyard": ("Ordinary Man", 2020),  # released as a single a few months before the album
    "Back On Earth": ("The Ozzman Cometh", 1997),  # new song recorded for the compilation
}


# ---------- characters to fix ----------

# Fancy punctuation -> plain ASCII. The last row is invisible characters that just go away.
FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    "…": "...",
    "\ufeff": "", "\u200b": "", "\u200c": "", "\u200d": "",
    "\u00ad": "", "\u200e": "", "\u200f": "", "\u2060": "",
}

# Two or more non-ASCII characters in a row is usually mojibake:
# UTF-8 text that was read with the wrong codec, e.g. "donâ€™t" instead of "don't".
MOJIBAKE_RUN = re.compile(r"[^\x00-\x7f]{2,}")
MOJIBAKE_CODECS = ["cp1252", "mac_roman", "latin-1"]

# Control characters (except tab and newline) that should never be in lyrics.
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

FOOTER = "OZZY OSBOURNE LYRICS"


# ---------- bracketed things ----------

# A whole line that is only [something] or (something).
BRACKET_LINE = re.compile(r"^[\[(]\s*([^\]\)]{1,300}?)\s*[\])]$")

# A [something] or (something) sitting inside a lyric line.
INLINE_TAG = re.compile(r"\s*[\[(]\s*([^\]\)\n]{1,60}?)\s*[\])]")

# Stage directions like [Chorus], [Solo], [x4], [repeat 3 times].
STAGE = re.compile(
    r"^(?:w/|"
    r"(?:chorus|verse|bridge|intro|outro|refrain|solo|instrumental|spoken|"
    r"repeat|guitar|drum|x\s*\d+|\d+\s*x|\d+\s*times|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+times)\b)",
    re.I,
)

# Editorial notes like (live), (bonus track), (feat. X). Not credits.
NOTE = re.compile(
    r"\b(cover|bonus|b-side|feat\.?|featuring|version|remaster(ed)?|itunes|"
    r"japanese|demo|live|reprise|edit)\b",
    re.I,
)

# Songwriting credits look like "Osbourne - Rhoads - Daisley" or just "Zakk Wylde".
NAME = r"[A-Z][\w.'\-]*(?:\s+[A-Z][\w.'\-]*)*"
CREDIT = re.compile(rf"^{NAME}(?:\s*(?:-|,|&|and)\s*{NAME})+$")  # two or more names
CREDIT_SOLO = re.compile(rf"^{NAME}$")  # a single name


def repair_mojibake(match):
    """re.sub calls this for every run of non-ASCII characters. Try to decode it properly."""
    run = match.group()
    for codec in MOJIBAKE_CODECS:
        try:
            fixed = run.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        # A real repair always gets shorter and never produces control characters.
        if len(fixed) < len(run) and not CONTROL.search(fixed):
            return fixed
    return run  # could not repair it, leave it alone


def normalize(text):
    """Fix mojibake, fold punctuation, drop control characters, tidy whitespace.

    Returns a list of non-empty lines.
    """
    text = MOJIBAKE_RUN.sub(repair_mojibake, text)
    text = unicodedata.normalize("NFC", text)
    for old, new in FOLD.items():
        text = text.replace(old, new)
    text = CONTROL.sub("", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def is_credit(tag, at_top):
    """Is this bracketed text a songwriting credit, rather than a stage direction or a note?"""
    tag = tag.strip().rstrip(":").strip()
    # [Chorus] and [SOLO] also look like capitalized surnames, so rule those out first.
    if STAGE.match(tag) or NOTE.search(tag):
        return False
    if CREDIT.match(tag):
        return True
    # A single name only counts as a credit at the very top of the song.
    if at_top and CREDIT_SOLO.match(tag):
        return True
    return False


def drop_stage_tag(match):
    """re.sub calls this for every inline [tag]. Remove stage directions, keep everything else."""
    if STAGE.match(match.group(1)):
        return ""
    return match.group()


def clean_song(raw):
    """Split one raw lyric into (lyrics text, credits text)."""
    body = []
    credits = []
    for line in normalize(raw):
        if line == FOOTER:
            continue
        bracket = BRACKET_LINE.match(line)
        if bracket:
            # A whole-line bracket is either a credit we keep or noise we drop. Never lyrics.
            if is_credit(bracket.group(1), at_top=(len(body) == 0)):
                credits.append(bracket.group(1))
            continue
        line = INLINE_TAG.sub(drop_stage_tag, line).strip()
        if line:
            body.append(line)
    return "\n".join(body), "; ".join(credits)


if __name__ == "__main__":
    # The KB is the verified list of Ozzy's studio albums: lowercase name -> (proper name, year).
    albums = {}
    for album in json.loads(ALBUMS_FILE.read_text(encoding="utf-8")):
        albums[album["album"].lower()] = (album["album"], album["year"])

    songs = []
    for line in INFILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        title = " ".join(normalize(record["title"]))

        if record["album"].lower() in albums:
            album, year = albums[record["album"].lower()]
        elif title in KEEP_ANYWAY:
            album, year = KEEP_ANYWAY[title]
        else:
            continue  # live album, compilation or Black Sabbath song: not an Ozzy studio original

        lyrics, credits = clean_song(record["lyrics"])
        if not lyrics:
            continue  # instrumentals leave nothing behind
        songs.append({"title": title, "album": album, "year": year, "credits": credits, "lyrics": lyrics})

    # One JSON object per line, for the code.
    jsonl = ""
    for song in songs:
        jsonl += json.dumps(song, ensure_ascii=False) + "\n"
    OUTFILE.write_text(jsonl, encoding="utf-8", newline="\n")

    # One readable text file, for humans.
    chunks = []
    for song in songs:
        chunks.append(f"### {song['title']} - {song['album']} ({song['year']})\n\n{song['lyrics']}\n")
    CORPUS.write_text("\n".join(chunks), encoding="utf-8", newline="\n")

    total_words = 0
    with_credits = 0
    for song in songs:
        total_words += len(song["lyrics"].split())
        if song["credits"]:
            with_credits += 1
    print(f"wrote {len(songs)} songs | {total_words:,} words | {with_credits} with credits")
    print(f"  {OUTFILE.name} ({OUTFILE.stat().st_size // 1024} KB) | {CORPUS.name} ({CORPUS.stat().st_size // 1024} KB)")
