"""Embed the songs and the album cards, and search them.

Press Run to build the two indexes. Import search / similar_songs to query them.

    songs   one vector per whole song (181)   -> song similarity
    kb      one vector per album card (13)    -> facts for the critic
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
LYRICS_FILE = HERE / "Data" / "darklyrics_clean.jsonl"
ALBUMS_FILE = HERE / "Data" / "kb" / "albums.json"
INDEX_DIR = HERE / "Data" / "index"

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# bge wants this prefix on a short query that searches longer texts.
# It is NOT used for song-to-song similarity, where both sides are the same kind of text.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Loaded once, when this file is imported. Downloads ~440 MB the first time.
model = SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts, prefix=""):
    """Turn texts into unit-length vectors. Unit length means dot product == cosine similarity."""
    with_prefix = []
    for text in texts:
        with_prefix.append(prefix + text)
    vectors = model.encode(with_prefix, normalize_embeddings=True)
    return vectors.astype("float32")


def album_card(album):
    """Turn one album record from albums.json into a short text we can embed."""
    lines = []
    lines.append(f"Album: {album['album']}")
    lines.append(f"Year: {album['year']}")
    if album["release_date"]:
        lines.append(f"Released: {album['release_date']}")
    else:
        lines.append("Released: month-level only, see note")
    if album["is_covers"]:
        lines.append("Covers album of songs by other artists")
    else:
        lines.append("Album of original songs")
    lines.append(f"Tracks ({len(album['tracks'])}): " + ", ".join(album["tracks"]))
    if album.get("bonus_tracks"):
        lines.append("Bonus tracks: " + ", ".join(album["bonus_tracks"]))
    return "\n".join(lines)


def build_index():
    """Embed every song and every album card, and save the vectors to Data/index/."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # songs: one vector per whole song
    songs = []
    for line in LYRICS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            songs.append(json.loads(line))
    lyrics = []
    song_rows = []
    for song in songs:
        lyrics.append(song["lyrics"])
        song_rows.append({
            "title": song["title"],
            "album": song["album"],
            "year": song["year"],
            "credits": song["credits"],
            "n_words": len(song["lyrics"].split()),
        })
    np.save(INDEX_DIR / "songs.npy", embed(lyrics))
    (INDEX_DIR / "songs.json").write_text(
        json.dumps(song_rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    # kb: one vector per album card
    albums = json.loads(ALBUMS_FILE.read_text(encoding="utf-8"))
    cards = []
    album_rows = []
    for album in albums:
        card = album_card(album)
        cards.append(card)
        album_rows.append({"album": album["album"], "year": album["year"], "card": card})
    np.save(INDEX_DIR / "kb.npy", embed(cards))
    (INDEX_DIR / "kb.json").write_text(
        json.dumps(album_rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(f"built the index in {time.time() - start:.1f} seconds -> {INDEX_DIR}")
    print(f"  {len(songs)} songs | {len(albums)} album cards")


def load_index(name):
    """Load one saved index. name is "songs" or "kb". Returns (vectors, rows)."""
    vectors = np.load(INDEX_DIR / f"{name}.npy")
    rows = json.loads((INDEX_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return vectors, rows


def top_k(vectors, rows, query_vector, k, skip=()):
    """The k rows whose vectors are most similar to query_vector, best first."""
    scores = vectors @ query_vector  # cosine similarity, because every vector has unit length
    hits = []
    for i in np.argsort(-scores):  # row numbers sorted from highest score to lowest
        if i in skip:
            continue
        hit = dict(rows[i])
        hit["score"] = round(float(scores[i]), 4)
        hits.append(hit)
        if len(hits) == k:
            break
    return hits


def search(query, index_name, k=5):
    """Free-text query -> the k most similar songs ("songs") or album cards ("kb")."""
    vectors, rows = load_index(index_name)
    query_vector = embed([query], QUERY_PREFIX)[0]
    return top_k(vectors, rows, query_vector, k)


def similar_songs(title, k=5):
    """The k songs most like the given one. No query prefix: both sides are songs."""
    vectors, rows = load_index("songs")
    # The same title can appear on more than one album (studio + live), so find every
    # copy and leave all of them out of the results.
    positions = []
    for i, row in enumerate(rows):
        if row["title"].lower() == title.lower():
            positions.append(i)
    if not positions:
        raise KeyError(f"no song titled {title!r}")
    return top_k(vectors, rows, vectors[positions[0]], k, skip=positions)


if __name__ == "__main__":
    build_index()

    print("\n-- kb lookup (grounds the critic) --")
    for hit in search("which album was the covers record", "kb", k=2):
        print(f"  {hit['score']:.3f}  {hit['album']} ({hit['year']})")

    print("\n-- corpus search (grounds the critic) --")
    for hit in search("regret and looking back on a wasted life", "songs", k=3):
        print(f"  {hit['score']:.3f}  {hit['title']} - {hit['album']} ({hit['year']})")

    print("\n-- song similarity (the demo) --")
    for hit in similar_songs("Crazy Train", k=3):
        print(f"  {hit['score']:.3f}  {hit['title']} - {hit['album']} ({hit['year']})")
