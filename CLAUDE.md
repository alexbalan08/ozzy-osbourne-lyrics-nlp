# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code style: keep it simple

**This is a learning project. The code must stay very simple and basic** — it exists to be read and understood, not to be production-grade.

- Prefer the obvious, boring solution over the clever one.
- Plain functions, plain loops, plain data structures. Add a class only when it genuinely simplifies something.
- No new abstractions, patterns, factories, decorators or indirection layers unless asked for.
- No new dependencies without asking first.
- Don't add error handling, caching, threading or configurability "just in case" — add it when something actually needs it.
- Explain the *why* in a short comment when a line isn't self-evident.

If existing code could be simpler and still do its job, say so.


## Commands

No test suite, no linter, no build step. Each script self-verifies when run directly:

    python retrieval.py     # rebuild both indexes into Data/index/, then run smoke_test()
    python critic.py        # fact-check the 5 built-in DEMO_CLAIMS
    python critic.py "Crazy Train is on Blizzard of Ozz."   # claims as argv instead
    python clean_lyrics.py  # regenerate darklyrics_clean.jsonl + corpus.txt (needs Data/kb/albums.json)
    python check_all.py     # run the critic over every song -> Data/critic_results.jsonl

`retrieval.py` must have run at least once before `critic.py` works — otherwise `load_index` fails on the missing `Data/index/kb.npy`.

`scrape_data.py` is **not runnable**. It is a Jupyter notebook exported to .py as documentation; its module-level code references `rows` and `spine`, which are never defined. Don't try to execute or "repair" it.

## Pipeline

    scrape_data.py (documentation only)
      -> Data/darklyrics.jsonl          raw scrape
    clean_lyrics.py  (reads Data/kb/albums.json to decide what to keep)
      -> Data/darklyrics_clean.jsonl    {title, album, year, credits, lyrics}
      -> Data/corpus.txt
    retrieval.py
      -> Data/index/{songs,kb}.{npy,json}
    critic.py                           consumes the kb index
    check_all.py                        critic over every song -> Data/critic_results.jsonl

## Architecture

**Two indexes, one embedding model** (`retrieval.py`). `BAAI/bge-base-en-v1.5` serves both:

- `songs` — 122 vectors, one per whole song; powers `similar_songs()`.
- `kb` — 12 vectors, one per album card; grounds the fact-checker.

Vectors are L2-normalized, so the `vectors @ query_vector` dot product in `top_k` *is* cosine similarity.

There is no class and no caching: `search` and `similar_songs` are plain functions that re-read the small `.npy`/`.json` files on every call, and both models (`bge` in `retrieval.py`, Llama in `critic.py`) load at module level when the file is imported. That is deliberate — don't wrap them in lazy loaders.

One thing that is easy to get wrong:

- `QUERY_PREFIX` belongs on **queries only** (`search`), never on song-to-song comparison (`similar_songs`), because BGE was trained asymmetrically.

**The critic** (`critic.py`) retrieves `TOP_K=3` album cards, supplies them as the only permitted source of truth, and returns SUPPORTED / CONTRADICTED / UNVERIFIABLE.

The model is **local** `Qwen/Qwen3-4B-Instruct-2507` in bf16 on the GPU via `transformers` (~8 GB of weights, cached in the HF hub cache). It has no native structured-output API, so the schema is enforced by hand in four coupled pieces:

1. `SYSTEM` states the JSON shape and forbids any other text.
2. `parse_verdict` cuts from the first `{` to the last `}` and `json.loads` it.
3. It checks that `verdict` is one of the three allowed values; anything else counts as a failed parse (returns `None`).
4. On failure, one repair turn feeds the bad output back and asks for corrected JSON.

The schema exists only as text inside `SYSTEM` plus the keys the print block reads — change one, change the other.

**The corpus is filtered by the KB.** `clean_lyrics.py` keeps a song only if its album is one of the 12 studio albums in `albums.json` (and takes the album's verified `year` from there), or if the title is in `KEEP_ANYWAY` — currently the two Ozzy originals that live outside the studio albums (*Under the Graveyard*, *Back On Earth*). Everything else in the raw scrape is a live album, compilation, covers album or Black Sabbath song and is dropped. Bonus tracks that darklyrics carries are listed under `bonus_tracks` in the KB so the critic can confirm them.

**The KB encodes deliberate edge cases** (`Data/kb/albums.json`) that the critic's rules are written against:

- *Diary of a Madman* has `release_date: null`, so its card reads "month-level only" and a claim about its exact day should be UNVERIFIABLE, not CONTRADICTED.
- The KB holds original studio tracklists only, so a song absent from *every* card is UNVERIFIABLE. A claim naming one specific album is different — that album's tracklist is complete, so a song missing from it is genuinely CONTRADICTED.

The prompt was deliberately simplified (it now allows general knowledge and has few rules). Known consequences: the model answers CONTRADICTED rather than UNVERIFIABLE for month-level dates and for albums not in the KB, and the 4B model occasionally contradicts its own explanation. These are model/prompt trade-offs, not bugs to chase.

## Gotchas

- Despite the repo name, **the network-science half is not implemented** — there is no graph code yet, only the RAG half.
- `Data/index/*.npy` and `__pycache__/` are committed, so rebuilding the index produces diff noise.
