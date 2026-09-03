"""Clean the scraped darklyrics JSONL into plain text an LLM can read.

Just press Run --- the paths below are relative to this file, so it works from
any working directory. 

"""



#this script was generated mostly by claude and refined through multiple attempts until 
#the produced output was correct


import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFILE = HERE / "Data" / "darklyrics.jsonl"
OUTFILE = HERE / "Data" / "darklyrics_clean.jsonl"
CORPUS = HERE / "Data" / "corpus.txt"


FOLD_TO_ASCII = True


MOJIBAKE_RUN = re.compile(r"[^\x00-\x7f]{2,}")
MOJIBAKE_CODECS = ("cp1252", "mac_roman", "latin-1")

FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    "…": "...",
}

FOLD.update(dict.fromkeys("﻿​‌‍­\u200E\u200F⁠", ""))
FOLD = str.maketrans(FOLD)

CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


BRACKET_LINE = re.compile(r"^[\[(]\s*([^\]\)]{1,300}?)\s*[\])]$")
INLINE_TAG = re.compile(r"\s*[\[(]\s*([^\]\)\n]{1,60}?)\s*[\])]")


STAGE = re.compile(
    r"^(?:w/|"
    r"(?:chorus|verse|bridge|intro|outro|refrain|solo|instrumental|spoken|"
    r"repeat|guitar|drum|x\s*\d+|\d+\s*x|\d+\s*times|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+times)\b)",
    re.I,
)

NOTE = re.compile(
    r"\b(cover|bonus|b-side|feat\.?|featuring|version|remaster(ed)?|itunes|"
    r"japanese|demo|live|reprise|edit)\b",
    re.I,
)

NAME = r"[A-Z][\w.'\-]*(?:\s+[A-Z][\w.'\-]*)*"
CREDIT = re.compile(rf"^{NAME}(?:\s*(?:-|,|&|and)\s*{NAME})+$")

CREDIT_SOLO = re.compile(rf"^{NAME}$")


def fix_mojibake(text):
    """Repair UTF-8 that was decoded with the wrong single-byte codec."""

    def repair(match):
        run = match.group()
        for codec in MOJIBAKE_CODECS:
            try:
                fixed = run.encode(codec).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

            if len(fixed) < len(run) and not CONTROL.search(fixed):
                return fixed
        return run

    return MOJIBAKE_RUN.sub(repair, text)


def normalize(text):
    """Mojibake, unicode and whitespace in one pass. Returns clean lines."""
    text = unicodedata.normalize("NFC", fix_mojibake(text))
    if FOLD_TO_ASCII:
        text = text.translate(FOLD)
    text = CONTROL.sub("", text)
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return [line for line in lines if line]


def is_credit(tag, at_top):
    """Is this bracketed line a songwriting credit rather than noise?

    Stage directions are ruled out first: "[Chorus]" and "[SOLO]" also look
    like capitalized surnames, so the name test would otherwise swallow them.
    """
    tag = tag.strip().rstrip(":").strip()
    if STAGE.match(tag) or NOTE.search(tag):
        return False
    return bool(CREDIT.match(tag) or (at_top and CREDIT_SOLO.match(tag)))


def clean_song(raw):
    """Split one lyric string into (body, credits). Brackets never survive."""
    credits, body = [], []

    def strip_stage(match):
        """Stage directions can also trail a lyric line, e.g. "... [x5]"."""
        return "" if STAGE.match(match.group(1)) else match.group()

    for line in normalize(raw or ""):
        bracket = BRACKET_LINE.match(line)
        if bracket:
            if is_credit(bracket.group(1), at_top=not body):
                credits.append(bracket.group(1))
            continue
        line = INLINE_TAG.sub(strip_stage, line).strip()
        if line:
            body.append(line)

    return "\n".join(body), "; ".join(credits)


def main():
    songs = []
    for line in INFILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        lyrics, credits = clean_song(record.get("lyrics"))
        if not lyrics:
            continue  # instrumentals leave nothing behind
        songs.append({
            "title": " ".join(normalize(record.get("title") or "")),
            "album": record.get("album"),
            "credits": credits,
            "lyrics": lyrics,
        })

    OUTFILE.write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in songs),
        encoding="utf-8",
        newline="\n",
    )
    CORPUS.write_text(
        "\n".join(
            f"### {s['title']} - {s['album']}\n\n{s['lyrics']}\n" for s in songs
        ),
        encoding="utf-8",
        newline="\n",
    )

    words = sum(len(s["lyrics"].split()) for s in songs)
    print(f"wrote {len(songs)} songs | {words:,} words"
          f" | {sum(1 for s in songs if s['credits'])} with credits")
    print(f"  {OUTFILE.name} ({OUTFILE.stat().st_size // 1024} KB)"
          f" | {CORPUS.name} ({CORPUS.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
