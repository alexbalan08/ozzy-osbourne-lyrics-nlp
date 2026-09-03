"""Embedding + retrieval layer over the cleaned lyrics and the album KB.

Press Run to build the indexes; import Retriever to query them.

Two indexes, one per consumer:

    songs   one vector per whole song (181)   -> song-similarity demo, and the
                                                 corpus side of the critic
    kb      one card per album (13)           -> fact grounding for the critic

Whole songs are embedded directly rather than chunked or pooled: only 2 of 181
exceed the model's 512-token window, so nothing meaningful is truncated. The
critic judges style, theme and factual accuracy, all of which want the gist of a
whole song rather than a fragment of one.

(A passage-level index would only earn its place for line-by-line originality
checking, where a single copied couplet has to spike a score. That is not what
this critic does.)

The model is loaded once, lazily, and reused by every call.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LYRICS = HERE / "Data" / "darklyrics_clean.jsonl"
ALBUMS = HERE / "data" / "kb" / "albums.json"
INDEX_DIR = HERE / "data" / "index"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32

# bge wants this prefix on short queries searching longer passages. It is NOT
# used for song-to-song similarity, which compares two texts of the same kind.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model():
    """Load the encoder once and reuse it. Downloads ~440 MB on first call."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def encode(texts, prefix=""):
    """Embed texts into L2-normalized vectors, so cosine similarity == dot."""
    texts = list(texts)
    if prefix:
        texts = [prefix + t for t in texts]
    return get_model().encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
    ).astype("float32")


def load_songs():
    return [
        json.loads(line)
        for line in LYRICS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def album_card(album):
    """Flatten one KB album record into text worth embedding."""
    parts = [
        f"Album: {album['album']}",
        f"Year: {album['year']}",
        f"Released: {album['release_date'] or 'month-level only, see note'}",
        "Covers album of songs by other artists" if album["is_covers"]
        else "Album of original songs",
        f"Tracks ({len(album['tracks'])}): " + ", ".join(album["tracks"]),
    ]
    if album.get("bonus_tracks"):
        parts.append("Bonus tracks: " + ", ".join(album["bonus_tracks"]))
    return "\n".join(parts)


def save(name, vectors, meta):
    np.save(INDEX_DIR / f"{name}.npy", vectors)
    (INDEX_DIR / f"{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def build():
    """Embed the corpus and the KB, and write both indexes to data/index/."""
    songs = load_songs()
    albums = json.loads(ALBUMS.read_text(encoding="utf-8"))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    save("songs", encode(s["lyrics"] for s in songs), [
        {"title": s["title"], "album": s["album"], "credits": s["credits"],
         "n_words": len(s["lyrics"].split())}
        for s in songs
    ])

    cards = [album_card(a) for a in albums]
    save("kb", encode(cards), [
        {"album": a["album"], "year": a["year"], "card": c}
        for a, c in zip(albums, cards)
    ])

    (INDEX_DIR / "meta.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "dim": int(get_model().get_sentence_embedding_dimension()),
        "counts": {"songs": len(songs), "kb": len(cards)},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"built in {time.time() - started:.1f}s -> {INDEX_DIR}")
    print(f"  songs {len(songs)} | kb {len(cards)}")


class Retriever:
    """Query the built indexes. Loads vectors on first use, model on first query."""

    def __init__(self, index_dir=INDEX_DIR):
        self.dir = Path(index_dir)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"no index in {self.dir} -- run `python retrieval.py`")
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))

    @lru_cache(maxsize=4)
    def _index(self, name):
        vectors = np.load(self.dir / f"{name}.npy")
        rows = json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
        return vectors, rows

    def _top(self, name, vector, k, skip=()):
        vectors, rows = self._index(name)
        scores = vectors @ vector
        hits = []
        for i in np.argsort(-scores):
            if i in skip:
                continue
            hits.append({**rows[i], "score": round(float(scores[i]), 4)})
            if len(hits) == k:
                break
        return hits

    def search(self, query, k=5, index="songs"):
        """Free-text query -> songs (or album cards, with index='kb')."""
        return self._top(index, encode([query], QUERY_PREFIX)[0], k)

    def search_kb(self, query, k=3):
        """Album facts, to ground a factual claim."""
        return self.search(query, k, index="kb")

    def similar_songs(self, title, k=5):
        """Songs most like this one. Symmetric, so no query prefix."""
        vectors, rows = self._index("songs")
        matches = [i for i, r in enumerate(rows) if r["title"].lower() == title.lower()]
        if not matches:
            raise KeyError(f"no song titled {title!r}")
        return self._top("songs", vectors[matches[0]], k, skip=set(matches))


def smoke_test():
    """Prove both consumers work off one loaded model."""
    r = Retriever()
    print(f"\nindex: {r.meta['counts']} | {r.meta['model']} | dim {r.meta['dim']}")

    print("\n-- kb lookup (grounds the critic) --")
    for hit in r.search_kb("which album was the covers record", k=2):
        print(f"  {hit['score']:.3f}  {hit['album']} ({hit['year']})")

    print("\n-- corpus search (grounds the critic) --")
    for hit in r.search("regret and looking back on a wasted life", k=3):
        print(f"  {hit['score']:.3f}  {hit['title']} - {hit['album']}")

    print("\n-- song similarity (the demo) --")
    for hit in r.similar_songs("Crazy Train", k=3):
        print(f"  {hit['score']:.3f}  {hit['title']} - {hit['album']}")


if __name__ == "__main__":
    build()
    smoke_test()
