# Ozzy Osbourne Lyrics NLP

What did Ozzy actually sing about, and did it change over 42 years? This repo goes from scraping his lyrics all the way to a topic model of his whole solo career (1980–2022), but with careful steps taken.

The main thing here is the **BERTopic notebook**. Instead of the usual "import, fit, done", we build every step of BERTopic by hand first, and look at what each step does. This has an education-style purpose and it can be a useful and in depth guide on how topic modeling works. 

## What's inside the repo and what we cover

- `scrape_data.py`: scrapes all Ozzy albums from DarkLyrics (cached, so each page is downloaded only once)
- `clean_lyrics.py`: cleans the text and keeps only original studio tracks, so no live albums, compilations or Black Sabbath songs - this is a crucial and main step we need to take for a correct analysis
- `Data/kb/albums.json`: a small knowledge base with every studio album, its exact release date and original tracklist (from Wikipedia)
- `retrieval.py`: embeds songs and albums with `bge-base-en-v1.5` for RAG search
- `critic.py` / `check_all.py`: a local LLM (Qwen3 4B) that fact-checks every song–album–year against the KB, because one wrong year would mess up the whole timeline. Ok, it s not hard to check manually a couple of albums and their years, but this monstration on how we can use a factual checker. 
- `BERTopic_ozzy_analysis.ipynb`: the tutorial on BERTopic, step-by step

- Other concepts like tokenization, word representations or chunking are covered, but expected with prior knowledge. 

## The BERTopic educational tutorial

Required read before starting: [BERTopic paper by Maarten Grootendorst](https://arxiv.org/abs/2203.05794).


What we do, step by step:

1. Load the corpus and remove repeated chorus lines
2. Look at tokenization (what happens to a word like *diabolus*? do we need to deal with music-related stopwords?)
3. Chunk the songs into verses and embed them with `all-mpnet-base-v2`
4. Test the embeddings: what's closest to *Crazy Train*?
5. Reduce dimensions with UMAP (768 → 5)
6. Cluster with HDBSCAN (and see why a third of the verses end up as outliers)
7. Read inside the clusters before naming anything
8. Build c-TF-IDF by hand to name the clusters
9. Fix the stop-words (again), because *yeah* and *gonna* are not topics
10. Track the topics album by album and plot the timeline
11. Exercise for you



## What we found

- The 80s are full of rock 'n' roll rebellion songs
- Madness and voices in the head show up early, almost vanish around 2010, then come back on *Patient Number 9*, which is kind of expected giving the medical problems Ozzy started to have. In one interview with Sharon, she states how negative in thoughts Ozzy became, which makes sense in relation to the last album's theme.
- Addiction peaks at the very end of his career, possibly as a flashback of his early life and career. 
- Death shows up twice for different reasons: Cold War fear on *The Ultimate Sin* (1986), then his own death on the last album, due to age and sickness 
- Love and goodbyes are mostly a mid-career thing (*No More Tears*, *Ozzmosis*), which again makes a lot of sense given that is the period when he started building his family, etc. 


pip install -r requirements.txt before running. 

If you only want the tutorial, just open the notebook. You need to upload the clean "corpus" from the Data section and runs fine on a CPU or in Colab.

To rebuild everything yourself:

```bash
python scrape_data.py
python clean_lyrics.py
python retrieval.py
python check_all.py   # the critic needs a GPU with ~6 GB
```

## Note on the data

Lyrics are copyrighted by their authors. They're used here only for learning and research. If you run the scraper, be nice to the site and keep the delays in.

---

Made by Alex Balan. Feel free to give your feedback or any other observations. 
