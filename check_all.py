"""Run the critic over every song in the dataset.

The claim for each song is "The song <title> is on the album <album>, released in <year>." Results go to
Data/critic_results.jsonl (one line per song) and a summary is printed at the end.

Takes a few minutes: one model call per song.
"""

import json
from pathlib import Path

from critic import check_claim

HERE = Path(__file__).resolve().parent
SONGS_FILE = HERE / "Data" / "darklyrics_clean.jsonl"
OUT_FILE = HERE / "Data" / "critic_results.jsonl"

songs = []
for line in SONGS_FILE.read_text(encoding="utf-8").splitlines():
    if line.strip():
        songs.append(json.loads(line))

results = []
for i, song in enumerate(songs, 1):
    claim = f"The song {song['title']} is on the album {song['album']}, released in {song['year']}."
    verdict, hits = check_claim(claim)

    if verdict is None:
        label = "NO_JSON"
        evidence = ""
        explanation = ""
    else:
        label = verdict["verdict"]
        evidence = verdict.get("evidence", "")
        explanation = verdict.get("explanation", "")

    retrieved = []
    for hit in hits:
        retrieved.append(hit["album"])

    print(f"{i:>3}/{len(songs)}  {label:<13} {song['title']} - {song['album']}")
    results.append({
        "title": song["title"],
        "album": song["album"],
        "year": song["year"],
        "claim": claim,
        "verdict": label,
        "evidence": evidence,
        "explanation": explanation,
        "retrieved": retrieved,
    })

jsonl = ""
for result in results:
    jsonl += json.dumps(result, ensure_ascii=False) + "\n"
OUT_FILE.write_text(jsonl, encoding="utf-8")

print()
counts = {}
for result in results:
    counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
for label in counts:
    print(f"  {label:<13} {counts[label]}")
print(f"saved {len(results)} results -> {OUT_FILE}")
