# Nightforce Market Voice

A small, **local** market-research pipeline. It pulls public customer conversation
about scope **mounts, rings, and bases** across seven brands from **YouTube** (Data
API) and **Reddit** (public no-auth JSON), filters to a clean corpus, builds
differential word clouds, classifies the mounts comments with a **local Ollama
model**, and produces two brand-by-attribute heat maps plus validation artifacts
and a one-page findings memo.

Public data only, polite request pacing, everything cached so nothing is fetched twice.

The whole thing is **one file**: `market_voice.py`. Open it in IDLE and press **F5**.

---

## 1. One-time setup

**Python 3.12** (works on 3.10+). Install the libraries once, in a terminal:

```
pip install requests pandas matplotlib wordcloud
```

**Ollama** must be running with your Gemma model pulled. Check your exact tag:

```
ollama list
```

The script defaults to model tag **`gemma4:e2b`**. If `ollama list` shows a
different tag, open `market_voice.py` and edit the `OLLAMA_MODEL` line near the top.
(The script also self-checks on startup and prints your installed tags if the one
it wants is missing — so you'll know exactly what to type.)

> Whatever `ollama list` prints is the source of truth for the tag.

**YouTube API key.** The script finds your key automatically, in this order:
1. the `YT_API_KEY` constant at the top of `market_voice.py` (paste it there if you like),
2. a `YT_API_KEY` environment variable,
3. a `.env` file next to the script containing `YT_API_KEY=...`,
4. your Desktop file `analytics - nightforce.txt` (it extracts the `AIza…` token).

Since you already keep the key in that Desktop `.txt`, option 4 means it should
just work with no extra step.

---

## 2. Run it

**In IDLE:** open `market_voice.py`, press **F5**. That runs every stage in order
on a cold cache and writes everything to `outputs/`.

**From a terminal**, run a single stage by passing its name:

```
python market_voice.py collect     # fetch YouTube + Reddit (cached)
python market_voice.py corpus      # hygiene + filter -> outputs/corpus.csv
python market_voice.py clouds      # 8 differential word-cloud PNGs
python market_voice.py classify    # local Ollama labels the mounts docs
python market_voice.py analyze     # 2 heat maps + precision/tactical CSVs
python market_voice.py report      # verbatims, validation sample, memo
python market_voice.py all         # everything, in order (same as F5)
```

To run a single stage from **inside IDLE** without a terminal, change the
`STAGE = "all"` line near the top of the file to e.g. `STAGE = "clouds"`, then F5.

Re-running is cheap: every HTTP response is cached under `cache/`, so a second run
makes **zero** network calls. Classification is written incrementally to
`outputs/classified.csv`, so if it crashes or you stop it, the next run resumes
where it left off.

---

## 3. What you get (in `outputs/`)

| File | What it is |
|------|------------|
| `corpus.csv` | one clean row per document (source, segment, url, matched brands, bucket, labels) |
| `cloud_<brand>.png` ×7 + `cloud_baseline.png` | differential (log-odds) word clouds + whole-corpus baseline |
| `heat_mention_share.png` | brand × attribute mention share |
| `heat_net_sentiment.png` | brand × attribute net sentiment, (pos − neg) / total |
| `mention_share_<segment>.csv`, `net_sentiment_<segment>.csv` | precision vs tactical splits |
| `classified.csv` | every classified mounts-bucket document (resumable log) |
| `verbatims.md` | best quotable comments per notable cell, with links |
| `validation_sample.csv` | 50-row stratified sample with empty human-label columns |
| `agreement.py` | scores human-vs-model agreement (attribute and sentiment separately) |
| `memo.md` | one-page findings template with real corpus stats filled in |
| `corpus_stats.txt` | the corpus statistics block |

### Validating the model (the interview-ready part)
1. Open `outputs/validation_sample.csv`, fill in `human_attribute` and
   `human_sentiment` for each row (your judgment vs the model's).
2. From `outputs/`, run: `python agreement.py`
3. It prints percent agreement for attribute and for sentiment. Put those two
   numbers in `memo.md`.

---

## 4. If a source fails

The pipeline never lets one failing source block the rest — it logs the gap,
skips, and continues.
- **Reddit blocked (403/429)** → it backs off once, then skips Reddit for the run;
  you still get the full YouTube-based analysis. (You can also drop in a manual
  Reddit CSV export later.)
- **YouTube key missing** → Reddit-only run.
- **Model slow** → classification is resumable, so you can stop and continue; or
  classify a subset by trimming `corpus.csv` before the `classify` stage.

---

## 5. Tuning

Near the top of `market_voice.py`:
- `OLLAMA_MODEL` — your Gemma tag.
- `TARGET_COMMENTS` — soft cap on how many comments to pull (default 2500).
- `YOUTUBE_QUOTA_BUDGET` — hard stop at 5000 units (search = 100, comments = 1).
- `REDDIT_SECONDS_BETWEEN` — pacing, default 2.0s.

Brands, category tokens, attributes, subreddits, YouTube queries, and the
classification prompt are all in the `CONFIG` block and match the prompt pack exactly.
