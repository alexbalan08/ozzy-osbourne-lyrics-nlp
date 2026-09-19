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

## Environment

Conda env `ozzy` (Python 3.12). Call the interpreter directly:

    C:\Users\alext\anaconda3\envs\ozzy\python.exe <script>.py

**torch must be the CUDA build** (currently `2.11.0+cu128`) because `critic.py` runs an 8B model on the GPU:

    pip install --index-url https://download.pytorch.org/whl/cu128 torch --force-reinstall --no-deps

Plain `pip install torch` may resolve to a CPU wheel; `retrieval.py` is fine with that, but `critic.py` then raises "CUDA GPU required". `--force-reinstall` is needed because pip treats an installed `+cpu` build as already satisfying `torch`.

## Commands

No test suite, no linter, no build step. Each script self-verifies when run directly:

    python retrieval.py     # rebuild both indexes into Data/index/, then run smoke_test()
    python critic.py        # fact-check the 5 built-in DEMO_CLAIMS
    python critic.py "Crazy Train is on Blizzard of Ozz."   # claims as argv instead
    python clean_lyrics.py  # regenerate darklyrics_clean.jsonl + corpus.txt

`retrieval.py` must have run at least once before `critic.py` works — `Retriever.__init__` raises `FileNotFoundError` if `Data/index/meta.json` is missing.

`scrape_data.py` is **not runnable**. It is a Jupyter notebook exported to .py as documentation; its module-level code references `rows` and `spine`, which are never defined. Don't try to execute or "repair" it.

## Pipeline

    scrape_data.py (documentation only)
      -> Data/darklyrics.jsonl          raw scrape
    clean_lyrics.py
      -> Data/darklyrics_clean.jsonl    {title, album, credits, lyrics}
      -> Data/corpus.txt
    retrieval.py
      -> Data/index/{songs,kb}.{npy,json} + meta.json
    critic.py                           consumes the kb index

## Architecture

**Two indexes, one embedding model** (`retrieval.py`). `BAAI/bge-base-en-v1.5` serves both:

- `songs` — 181 vectors, one per whole song; powers `similar_songs()`.
- `kb` — 13 vectors, one per album card; grounds the fact-checker.

Vectors are L2-normalized, so the `vectors @ vector` dot product in `Retriever._top` *is* cosine similarity.

Two things that are easy to get wrong:

- `QUERY_PREFIX` belongs on **queries only** (`search`), never on song-to-song comparison (`similar_songs`), because BGE was trained asymmetrically.
- BGE uses **CLS pooling, not mean pooling** — the vector is the `[CLS]` hidden state, not an average of token vectors. A whole song collapses into one 768-dim vector. Median song is 246 tokens against a 512-token window; exactly one song ("Don't Blame Me", 670 tokens) is truncated, losing its last quarter.

**The critic** (`critic.py`) retrieves `TOP_K=3` album cards, supplies them as the only permitted source of truth, and returns SUPPORTED / CONTRADICTED / UNVERIFIABLE.

The model is **local** `meta-llama/Llama-3.1-8B-Instruct` in bf16 on the GPU via `transformers` (~16 GB of weights, ~15 s first load, cached in the HF hub cache). It has no native structured-output API, so the schema is enforced by hand in four coupled pieces:

1. `SYSTEM` states the JSON shape and forbids any other text.
2. `_extract_json` brace-matches the first `{...}` out of the raw reply.
3. The pydantic `Verdict` model validates it.
4. On failure, one repair turn feeds the bad output back and asks for corrected JSON.

Changing the `Verdict` fields means updating the schema text inside `SYSTEM` too.

**The KB encodes deliberate edge cases** (`Data/kb/albums.json`) that the critic's rules are written against:

- *Diary of a Madman* has `release_date: null`, so its card reads "month-level only" and a claim about its exact day should be UNVERIFIABLE, not CONTRADICTED.
- *Under Cover* has `is_covers: true`.
- The KB holds original studio tracklists only, so a song absent from *every* card is UNVERIFIABLE. A claim naming one specific album is different — that album's tracklist is complete, so a song missing from it is genuinely CONTRADICTED.

Llama-3.1-8B does not reliably honour the month-level rule and tends to answer CONTRADICTED there. That is a model-capability gap, not a prompt bug.

## Gotchas

- **Path casing.** `retrieval.py` mixes `HERE / "Data"` (lyrics) with `HERE / "data"` (KB and index). It only works because Windows is case-insensitive; the committed directory is `Data/`. Normalize before running on Linux or macOS.
- Despite the repo name, **the network-science half is not implemented** — there is no graph code yet, only the RAG half.
- `Data/index/*.npy` and `__pycache__/` are committed, so rebuilding the index produces diff noise.
