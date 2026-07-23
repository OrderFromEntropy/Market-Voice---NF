# -*- coding: utf-8 -*-
"""
Nightforce Market Voice -- single-file local pipeline.

WHAT THIS DOES
    Pulls public customer conversation about scope mounts / rings / bases across
    seven brands from YouTube (Data API) and Reddit (public no-auth JSON), filters
    to a clean corpus, builds differential (log-odds) word clouds, classifies the
    mounts-bucket comments with a LOCAL Ollama model, and produces two brand-by-
    attribute heat maps plus validation artifacts and a findings memo.

HOW TO RUN (IDLE, Python 3.12)
    1. Install deps once, in a terminal / cmd:
         pip install requests pandas matplotlib wordcloud
       (python-dotenv is optional; this script parses .env itself.)
    2. Make sure Ollama is running and your Gemma model is pulled:
         ollama list
       If your model tag is not "gemma4:e2b", edit OLLAMA_MODEL below.
    3. Make sure the YouTube key is reachable (see YT_API_KEY resolution below).
    4. Open this file in IDLE and press F5 (Run). It runs every stage in order.
       To run a single stage, set STAGE below (or pass it on the command line:
       `python market_voice.py analyze`). Everything is cached, so re-runs make
       zero redundant network calls.

OUTPUTS  (written to ./outputs/)
    corpus.csv                     Layer 1 -- one clean corpus row per document
    cloud_<brand>.png (x7)         Layer 2 -- differential clouds per brand
    cloud_baseline.png             Layer 2 -- whole-corpus baseline cloud
    heat_mention_share.png         Layer 3 -- brand x attribute mention share
    heat_net_sentiment.png         Layer 3 -- brand x attribute net sentiment
    mention_share_<segment>.csv    Layer 3 -- precision / tactical splits
    net_sentiment_<segment>.csv    Layer 3 -- precision / tactical splits
    classified.csv                 Layer 3 -- every classified mounts-bucket doc
    verbatims.md                   Layer 4 -- best quotable comments per cell
    validation_sample.csv          Layer 4 -- 50-row stratified human-label sheet
    agreement.py                   Layer 4 -- human-vs-model agreement scorer
    memo.md                        Layer 4 -- one-page findings template
    corpus_stats.txt               Corpus statistics block

This is a personal, one-off research tool: public data only, polite pacing,
everything cached so nothing is fetched twice.
"""

import os
import re
import sys
import csv
import json
import time
import math
import glob
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests

# ============================================================================
#  ORCHESTRATION
# ============================================================================
# Which stage to run when you press F5 in IDLE. One of:
#   "collect"  -> fetch YouTube + Reddit (cached)
#   "corpus"   -> hygiene + filter -> outputs/corpus.csv
#   "clouds"   -> differential word clouds
#   "classify" -> local Ollama classification of mounts-bucket docs
#   "analyze"  -> heat maps + segment CSVs
#   "report"   -> verbatims / validation sample / agreement.py / memo
#   "all"      -> everything, in order  (default)
STAGE = "all"

# ============================================================================
#  CONFIG  (transcribed exactly from PROMPT_PACK.md -- do not invent brands,
#           tokens, or attributes)
# ============================================================================

# --- Brands and aliases. Lowercase matching on word boundaries. Misspellings are intentional.
BRANDS = {
    "nightforce":      ["nightforce", "night force", "unimount", "uni-mount", "magmount",
                        "mag mount", "x-treme duty", "xtreme duty", "ultralite"],
    "leupold":         ["leupold", "leopold", "luepold", "lupold", "backcountry rings",
                        "mark ar mount"],
    "reptilia":        ["reptilia", "reptillia", "reptila", "aus mount", "reptilia aus"],
    "badger_ordnance": ["badger ordnance", "badger ord", "badger", "condition one", "c1 mount"],
    "spuhr":           ["spuhr", "sphur"],
    "geissele":        ["geissele", "geiselle", "giselle", "gieselle", "super precision"],
    "warne":           ["warne", "warne skyline", "mountain tech"],
}
# Note: "badger" and "condition one" will pull some noise (the animal, carry-condition talk).
# The category-token requirement below filters most of it. Accept the remainder.
# Bare "NF" is excluded on purpose; collision rate is too high. Add it only if corpus runs thin.
# To drop a brand from the study, delete its line.

# --- A document must match >=1 brand alias AND >=1 token from one of these buckets.
CATEGORY_TOKENS_MOUNTS = [
    "mount", "mounts", "ring", "rings", "base", "bases", "cantilever", "one piece",
    "one-piece", "qd", "quick detach", "quick-detach", "torque", "in-lb", "in lb",
    "inch pound", "hold zero", "holds zero", "held zero", "holding zero", "lost zero",
    "return to zero", "rtz", "lapping", "lapped", "ring gap", "picatinny", "pic rail",
    "1913", "riser", "offset", "piggyback", "top ring", "30mm", "34mm", "35mm", "36mm",
]
CATEGORY_TOKENS_ACCESSORY_OTHER = [
    "flip cap", "flip caps", "flip-up cap", "lens cap", "lens caps", "sunshade",
    "kill flash", "killflash", "ard", "anti-cant", "bubble level", "scope level",
    "throw lever", "cat tail", "cattail",
]

# --- Fixed attribute taxonomy for classification. Do not extend without re-running validation.
ATTRIBUTES = [
    "return_to_zero", "durability", "weight", "machining_finish", "price_value",
    "qd_repeatability", "customer_service_warranty", "availability_leadtime",
    "fit_height_options", "install_experience", "other",
]

# --- Sources and segments.
SUBREDDITS = {
    "longrange": "precision",
    "qualitytacticalgear": "tactical",
    "tacticalgear": "tactical",
    # Optional expansions if the corpus runs thin. Uncomment deliberately.
    # "ar15": "tactical",
    # "Hunting": "hunting",
}
# Reddit search queries are generated as: "<primary brand name> mount" and
# "<primary brand name> rings" per brand per subreddit, paginated via `after`.

YOUTUBE_QUERIES = [
    "nightforce unimount review", "nightforce magmount review", "spuhr mount review",
    "badger ordnance condition one review", "reptilia aus mount review",
    "geissele super precision review", "warne mountain tech review",
    "leupold scope rings review", "best scope mount long range",
    "best lpvo mount", "scope mount comparison", "scope ring torque",
]
# YouTube segment tag: "precision" if the video title matches long range / precision / PRS,
# else "tactical" if it matches lpvo / ar15 / carbine, else "mixed".

# --- Pacing and identification.
REDDIT_SECONDS_BETWEEN = 2.0
USER_AGENT = "personal-market-research/0.1 (one-off student project; contact in profile)"
YOUTUBE_QUOTA_BUDGET = 5000

# --- Classification prompt for the local model. Temperature 0. JSON only.
CLASSIFY_PROMPT = """You label short comments about riflescope mounting accessories.
Return ONLY a JSON object, no prose, exactly matching this schema:
{"brands": [zero or more of: nightforce, leupold, reptilia, badger_ordnance, spuhr,
geissele, warne, other, none], "attribute": one of [return_to_zero, durability, weight,
machining_finish, price_value, qd_repeatability, customer_service_warranty,
availability_leadtime, fit_height_options, install_experience, other],
"sentiment": one of [positive, negative, neutral, mixed], "quotable": true or false}
Pick the single attribute the comment is MOST about. quotable is true only if the comment
is vivid, specific, and under about 60 words. Watch for sarcasm; "only lost zero twice,
great value" is negative.

Comment: "Swapped to a Spuhr and my zero survives barrel swaps, worth every penny."
{"brands": ["spuhr"], "attribute": "return_to_zero", "sentiment": "positive", "quotable": true}

Comment: "The badger c1 is a tank but man it is heavy on a 6lb hunting rig"
{"brands": ["badger_ordnance"], "attribute": "weight", "sentiment": "mixed", "quotable": true}

Comment: "my warne only lost zero twice this season, great value lol"
{"brands": ["warne"], "attribute": "return_to_zero", "sentiment": "negative", "quotable": true}

Comment: "What height rings do I need for a 56mm objective on a 700?"
{"brands": ["none"], "attribute": "fit_height_options", "sentiment": "neutral", "quotable": false}

Comment: {comment}
"""

# --- Sample comments for the Prompt 1 classification probe.
SAMPLE_COMMENTS = [
    "Swapped to a Spuhr and my zero survives barrel swaps, worth every penny.",
    "The badger c1 is a tank but man it is heavy on a 6lb hunting rig",
    "my warne only lost zero twice this season, great value lol",
    "What height rings do I need for a 56mm objective on a 700?",
    "Leupold rings were fine I guess, nothing special, they hold the scope.",
]

# ============================================================================
#  LOCAL SETTINGS  (edit these two if needed)
# ============================================================================
# Your installed Ollama model tag. Run `ollama list` to confirm. The script
# self-checks this on startup and prints your installed tags if it is missing.
OLLAMA_MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434"

# Paste your YouTube Data API key here to hard-wire it (optional). If left blank,
# the script looks for YT_API_KEY in the environment, then a .env file next to
# this script, then your Desktop "analytics - nightforce.txt".
YT_API_KEY = ""

# How many comments to target overall before we stop pulling (soft cap; keeps a
# cold run inside a couple evenings). Set to None to pull everything found.
TARGET_COMMENTS = 2500

# ============================================================================
#  PATHS
# ============================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR = os.path.join(ROOT, "outputs")
CORPUS_CSV = os.path.join(OUT_DIR, "corpus.csv")
CLASSIFIED_CSV = os.path.join(OUT_DIR, "classified.csv")

for d in (CACHE_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================================
#  SMALL UTILITIES
# ============================================================================
def log(msg):
    print(msg, flush=True)


def resolve_api_key():
    """Constant -> env -> .env -> Desktop txt. Returns key or None."""
    if YT_API_KEY.strip():
        return YT_API_KEY.strip()
    if os.environ.get("YT_API_KEY", "").strip():
        return os.environ["YT_API_KEY"].strip()
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line.startswith("YT_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # Convenience fallback: the .txt on the user's Desktop.
    desktop_guesses = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "analytics - nightforce.txt"),
        os.path.join(os.path.expanduser("~"), "Desktop", "analytics - nightforce.txt"),
    ]
    for p in desktop_guesses:
        if os.path.exists(p):
            txt = open(p, encoding="utf-8", errors="ignore").read()
            m = re.search(r"AIza[0-9A-Za-z_\-]{35}", txt)
            if m:
                return m.group(0)
    return None


def cache_path_for(url):
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, h + ".json")


def cached_get_json(url, headers=None, pace_seconds=0.0, params=None):
    """GET a URL, caching the raw JSON to disk keyed by the full URL.
    A cache hit makes zero network calls and does not pace. Returns
    (data, from_cache) or (None, from_cache) on failure."""
    full = url
    if params:
        # Build a stable full URL so the cache key includes the params.
        from urllib.parse import urlencode
        sep = "&" if "?" in url else "?"
        full = url + sep + urlencode(sorted(params.items()))
    cp = cache_path_for(full)
    if os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8")), True
        except Exception:
            pass  # corrupt cache -> refetch
    if pace_seconds:
        time.sleep(pace_seconds)
    try:
        r = requests.get(url, headers=headers or {}, params=params, timeout=30)
    except Exception as e:
        log("    ! network error: %s" % e)
        return None, False
    if r.status_code != 200:
        # Surface the API's own reason (e.g. "commentsDisabled") without
        # echoing the URL, which contains the API key.
        reason = ""
        try:
            errs = r.json().get("error", {}).get("errors", [])
            reason = errs[0].get("reason", "") if errs else ""
        except Exception:
            pass
        who = "youtube" if "googleapis.com" in url else ("reddit" if "reddit.com" in url else url[:40])
        log("    ! HTTP %s (%s) from %s" % (r.status_code, reason or "no-reason", who))
        return {"__error__": r.status_code, "__reason__": reason, "__body__": r.text[:400]}, False
    try:
        data = r.json()
    except Exception:
        log("    ! non-JSON response")
        return None, False
    json.dump(data, open(cp, "w", encoding="utf-8"))
    return data, False


def iso_from_epoch(epoch):
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


# ============================================================================
#  STAGE 0  ENVIRONMENT / STARTUP CHECKS
# ============================================================================
def check_dependencies(need_plotting=True, need_ollama=False):
    missing = []
    try:
        import pandas  # noqa
    except Exception:
        missing.append("pandas")
    if need_plotting:
        try:
            import matplotlib  # noqa
        except Exception:
            missing.append("matplotlib")
        try:
            import wordcloud  # noqa
        except Exception:
            missing.append("wordcloud")
    if missing:
        log("\n*** Missing packages: %s" % ", ".join(missing))
        log("*** Install them, then re-run:")
        log("      pip install %s\n" % " ".join(missing))
        return False
    return True


def check_ollama_model():
    """Return True if OLLAMA_MODEL is reachable; print available tags if not."""
    data, _ = cached_get_json(OLLAMA_URL + "/api/tags")
    if not data or "models" not in data:
        # Don't cache a failed reach; delete any stale cache entry.
        cp = cache_path_for(OLLAMA_URL + "/api/tags")
        if os.path.exists(cp):
            os.remove(cp)
        log("\n*** Could not reach Ollama at %s" % OLLAMA_URL)
        log("*** Start Ollama (it should be listening on 11434), then re-run.\n")
        return False
    tags = [m.get("name", "") for m in data.get("models", [])]
    if OLLAMA_MODEL in tags or any(t.split(":")[0] == OLLAMA_MODEL for t in tags):
        return True
    log("\n*** Ollama is up but model '%s' is not installed." % OLLAMA_MODEL)
    log("*** Installed tags: %s" % (", ".join(tags) or "(none)"))
    log("*** Fix: `ollama pull %s`  OR edit OLLAMA_MODEL at the top of this file.\n" % OLLAMA_MODEL)
    return False


# ============================================================================
#  STAGE 1a  COLLECT -- YOUTUBE
# ============================================================================
YT = "https://www.googleapis.com/youtube/v3"
_quota_used = {"units": 0}


def yt_segment_for_title(title):
    t = (title or "").lower()
    if re.search(r"long range|precision|\bprs\b", t):
        return "precision"
    if re.search(r"lpvo|ar15|ar-15|carbine", t):
        return "tactical"
    return "mixed"


def collect_youtube(api_key):
    """Returns list of raw doc dicts from YouTube comments."""
    docs = []
    if not api_key:
        log("  [youtube] no API key resolved -- skipping YouTube (Reddit-only fallback).")
        return docs
    seen_videos = set()
    for q in YOUTUBE_QUERIES:
        if _quota_used["units"] + 100 > YOUTUBE_QUOTA_BUDGET:
            log("  [youtube] quota budget reached; stopping search.")
            break
        data, cached = cached_get_json(
            YT + "/search",
            params={"key": api_key, "q": q, "part": "snippet",
                    "type": "video", "maxResults": 25, "relevanceLanguage": "en"},
        )
        if not cached:
            _quota_used["units"] += 100
        if not data or "items" not in data:
            if data and data.get("__error__"):
                log("  [youtube] search error %s for %r" % (data["__error__"], q))
            continue
        for it in data["items"]:
            vid = it.get("id", {}).get("videoId")
            if not vid or vid in seen_videos:
                continue
            seen_videos.add(vid)
            title = it.get("snippet", {}).get("title", "")
            seg = yt_segment_for_title(title)
            if _quota_used["units"] + 1 > YOUTUBE_QUOTA_BUDGET:
                log("  [youtube] quota budget reached; stopping comments.")
                return docs
            cdata, ccached = cached_get_json(
                YT + "/commentThreads",
                params={"key": api_key, "videoId": vid, "part": "snippet",
                        "maxResults": 100, "textFormat": "plainText", "order": "relevance"},
            )
            if not ccached:
                _quota_used["units"] += 1
            if not cdata or "items" not in cdata:
                continue  # comments disabled or error -> skip video
            for c in cdata["items"]:
                sn = c.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = sn.get("textDisplay", "")
                docs.append({
                    "id": "yt_" + c.get("id", vid),
                    "source": "youtube",
                    "subreddit_or_video": vid,
                    "segment": seg,
                    "url": "https://www.youtube.com/watch?v=%s&lc=%s" % (vid, c.get("id", "")),
                    "created_utc": iso_from_epoch(_yt_epoch(sn.get("publishedAt"))),
                    "text": text,
                })
            if TARGET_COMMENTS and len(docs) >= TARGET_COMMENTS:
                log("  [youtube] hit target of %d comments." % TARGET_COMMENTS)
                return docs
    log("  [youtube] collected %d raw comments (quota used ~%d units)."
        % (len(docs), _quota_used["units"]))
    return docs


def _yt_epoch(iso):
    if not iso:
        return 0
    try:
        return datetime.strptime(iso.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except Exception:
        return 0


# ============================================================================
#  STAGE 1b  COLLECT -- REDDIT (public, no auth)
# ============================================================================
PRIMARY_NAME = {
    "nightforce": "nightforce", "leupold": "leupold", "reptilia": "reptilia",
    "badger_ordnance": "badger ordnance", "spuhr": "spuhr",
    "geissele": "geissele", "warne": "warne",
}
REDDIT_HEADERS = {"User-Agent": USER_AGENT}


def _reddit_walk_comments(children, out, video_ctx):
    """Recursively collect t1 comment bodies from a Reddit comment tree."""
    for ch in children:
        if ch.get("kind") != "t1":
            continue
        d = ch.get("data", {})
        body = d.get("body", "") or ""
        out.append({
            "id": "rd_" + d.get("id", ""),
            "source": "reddit",
            "subreddit_or_video": video_ctx["subreddit"],
            "segment": video_ctx["segment"],
            "url": "https://www.reddit.com" + video_ctx["permalink"] + d.get("id", ""),
            "created_utc": iso_from_epoch(d.get("created_utc")),
            "text": body,
        })
        replies = d.get("replies")
        if isinstance(replies, dict):
            _reddit_walk_comments(replies.get("data", {}).get("children", []), out, video_ctx)


def collect_reddit():
    """Returns list of raw doc dicts from Reddit. Self-skips on 403/429."""
    docs = []
    request_budget = 60  # keep total requests bounded and polite
    made = 0
    blocked = False
    for sub, segment in SUBREDDITS.items():
        if blocked:
            break
        for brand, name in PRIMARY_NAME.items():
            for suffix in ("mount", "rings"):
                if blocked or made >= request_budget:
                    break
                query = "%s %s" % (name, suffix)
                from urllib.parse import quote
                url = ("https://www.reddit.com/r/%s/search.json"
                       "?q=%s&restrict_sr=1&sort=relevance&t=all&limit=25&raw_json=1"
                       % (sub, quote(query)))
                data, cached = cached_get_json(url, headers=REDDIT_HEADERS,
                                               pace_seconds=0 if cached_exists(url) else REDDIT_SECONDS_BETWEEN)
                if not cached:
                    made += 1
                if data and data.get("__error__") in (403, 429):
                    log("  [reddit] HTTP %s -- backing off once." % data["__error__"])
                    time.sleep(5)
                    data, cached = cached_get_json(url, headers=REDDIT_HEADERS, pace_seconds=0)
                    if not cached:
                        made += 1
                    if data and data.get("__error__") in (403, 429):
                        log("  [reddit] still blocked; skipping Reddit for this run.")
                        blocked = True
                        break
                if not data or "data" not in data:
                    continue
                children = data.get("data", {}).get("children", [])
                for post in children[:5]:  # top few threads per query
                    if made >= request_budget:
                        break
                    pd = post.get("data", {})
                    permalink = pd.get("permalink")
                    if not permalink:
                        continue
                    turl = "https://www.reddit.com" + permalink + ".json?limit=200&raw_json=1"
                    tdata, tcached = cached_get_json(
                        turl, headers=REDDIT_HEADERS,
                        pace_seconds=0 if cached_exists(turl) else REDDIT_SECONDS_BETWEEN)
                    if not tcached:
                        made += 1
                    if not tdata or not isinstance(tdata, list) or len(tdata) < 2:
                        continue
                    ctx = {"subreddit": sub, "segment": segment, "permalink": permalink}
                    # include the post selftext as a document too
                    if pd.get("selftext"):
                        docs.append({
                            "id": "rd_" + pd.get("id", ""),
                            "source": "reddit",
                            "subreddit_or_video": sub,
                            "segment": segment,
                            "url": "https://www.reddit.com" + permalink,
                            "created_utc": iso_from_epoch(pd.get("created_utc")),
                            "text": pd.get("title", "") + ". " + pd.get("selftext", ""),
                        })
                    _reddit_walk_comments(tdata[1].get("data", {}).get("children", []), docs, ctx)
    log("  [reddit] collected %d raw comments in %d requests%s."
        % (len(docs), made, " (blocked early)" if blocked else ""))
    return docs


def cached_exists(url, params=None):
    full = url
    if params:
        from urllib.parse import urlencode
        sep = "&" if "?" in url else "?"
        full = url + sep + urlencode(sorted(params.items()))
    return os.path.exists(cache_path_for(full))


# ============================================================================
#  STAGE 1  COLLECT (driver) -> writes cache; returns raw docs
# ============================================================================
def stage_collect():
    log("[collect] YouTube ...")
    api_key = resolve_api_key()
    if api_key:
        log("  [youtube] API key resolved (ends ...%s)." % api_key[-4:])
    yt_docs = collect_youtube(api_key)
    log("[collect] Reddit ...")
    rd_docs = collect_reddit()
    raw = yt_docs + rd_docs
    # persist raw docs so downstream stages don't need to re-run collect
    with open(os.path.join(CACHE_DIR, "raw_docs.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f)
    log("[collect] total raw documents: %d (youtube=%d, reddit=%d)"
        % (len(raw), len(yt_docs), len(rd_docs)))
    return raw


def load_raw_docs():
    p = os.path.join(CACHE_DIR, "raw_docs.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return stage_collect()


# ============================================================================
#  STAGE 2  BUILD CORPUS -- hygiene + filter + tag
# ============================================================================
def matched_brands(text_low):
    hits = []
    for brand, aliases in BRANDS.items():
        for alias in aliases:
            # word-boundary-ish match; aliases may contain spaces/hyphens
            pat = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
            if re.search(pat, text_low):
                hits.append(brand)
                break
    return hits


def bucket_for(text_low):
    for tok in CATEGORY_TOKENS_MOUNTS:
        pat = r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])"
        if re.search(pat, text_low):
            return "mounts"
    for tok in CATEGORY_TOKENS_ACCESSORY_OTHER:
        pat = r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])"
        if re.search(pat, text_low):
            return "accessory_other"
    return None


def stage_corpus():
    import pandas as pd
    raw = load_raw_docs()
    log("[corpus] hygiene + filter over %d raw docs ..." % len(raw))
    seen_texts = set()
    rows = []
    dropped_short = dropped_dupe = dropped_bot = dropped_nomatch = 0
    for d in raw:
        text = (d.get("text") or "").strip()
        if len(text) < 25:
            dropped_short += 1
            continue
        # strip obvious bot boilerplate lines
        lines = [ln for ln in text.splitlines() if "i am a bot" not in ln.lower()]
        text = "\n".join(lines).strip()
        if not text:
            dropped_bot += 1
            continue
        key = text.lower()
        if key in seen_texts:
            dropped_dupe += 1
            continue
        low = text.lower()
        brands = matched_brands(low)
        bucket = bucket_for(low)
        if not brands or not bucket:
            dropped_nomatch += 1
            continue
        seen_texts.add(key)
        rows.append({
            "id": d.get("id", ""),
            "source": d.get("source", ""),
            "subreddit_or_video": d.get("subreddit_or_video", ""),
            "segment": d.get("segment", ""),
            "url": d.get("url", ""),
            "created_utc": d.get("created_utc", ""),
            "text": text,
            "window": window_text(text),
            "brands_matched": "|".join(brands),
            "bucket": bucket,
            "attribute": "",
            "sentiment": "",
            "quotable": "",
            "model_raw": "",
        })
    df = pd.DataFrame(rows, columns=[
        "id", "source", "subreddit_or_video", "segment", "url", "created_utc",
        "text", "window", "brands_matched", "bucket", "attribute", "sentiment", "quotable", "model_raw"])
    df.to_csv(CORPUS_CSV, index=False)
    log("[corpus] wrote %s : %d documents" % (CORPUS_CSV, len(df)))
    log("[corpus] dropped -> short:%d dupe:%d bot:%d no-brand/token:%d"
        % (dropped_short, dropped_dupe, dropped_bot, dropped_nomatch))
    mounts = (df["bucket"] == "mounts").sum()
    nbrands = len(set("|".join(df["brands_matched"]).split("|")) - {""})
    log("[corpus] mounts-bucket docs: %d | distinct brands: %d" % (mounts, nbrands))
    write_corpus_stats(df)
    return df


def write_corpus_stats(df):
    lines = []
    lines.append("CORPUS STATISTICS")
    lines.append("=" * 40)
    def _counts(col):
        return {str(k): int(v) for k, v in df[col].value_counts().items()}
    lines.append("total documents : %d" % len(df))
    lines.append("by source       : %s" % _counts("source"))
    lines.append("by segment      : %s" % _counts("segment"))
    lines.append("by bucket       : %s" % _counts("bucket"))
    bc = Counter()
    for b in df["brands_matched"]:
        for x in b.split("|"):
            if x:
                bc[x] += 1
    lines.append("by brand        : %s" % dict(bc.most_common()))
    if len(df):
        dates = [d for d in df["created_utc"] if d]
        if dates:
            lines.append("date range      : %s .. %s" % (min(dates), max(dates)))
    txt = "\n".join(lines)
    open(os.path.join(OUT_DIR, "corpus_stats.txt"), "w", encoding="utf-8").write(txt + "\n")
    return txt


# ============================================================================
#  TEXT PROCESSING -- windowing, tokenizing, n-grams (shared by Layers 2 & 3)
# ============================================================================
STOPWORDS = set("""
a an the and or but if then than so as of to in on at for with without from by about into over
under again further is are was were be been being have has had do does did doing will would can
could should may might must this that these those it its it's i you he she they we me him her them
my your his their our not no yes just really very too also more most some any all one two get got
like dont don't im i'm youre you're thats that's what which who whom whose when where why how there
here out up down off out again once only own same such nor own too s t re ve ll d m o
scope scopes rifle rifles gun guns thing things stuff good bad great nice pretty much lot lots
i'll couldn didn don isn won
definitely recently currently basically honestly literally probably
""".split())

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']+")
# Domain abbreviations worth keeping despite the length-3 minimum.
SHORT_KEEP = {"qd"}


def tokenize(text):
    """Lowercase, strip punctuation, minimum token length 3."""
    toks = []
    for t in TOKEN_RE.findall(text.lower()):
        t = t.strip("-'")
        if len(t) < 3 and t not in SHORT_KEEP:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks


# --- Sentence windowing around brand mentions -------------------------------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ALL_ALIASES = sorted({a for al in BRANDS.values() for a in al}, key=len, reverse=True)
ALIAS_RE = re.compile("|".join(
    r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])" for a in _ALL_ALIASES))
# Every token that is part of a brand or product name -- excluded from term tables.
BRAND_TOKENS = set()
for _al in BRANDS.values():
    for _a in _al:
        for _t in re.findall(r"[a-z0-9]+", _a.lower()):
            BRAND_TOKENS.add(_t)


def window_text(full):
    """Keep only the sentence(s) containing a brand mention plus one sentence on
    each side. Falls back to the whole text when there is nothing to split on or
    no explicit mention (e.g. a one-line comment)."""
    flat = full.replace("\n", " ").strip()
    sents = [s for s in SENT_SPLIT.split(flat) if s.strip()]
    if len(sents) <= 1:
        return flat
    keep = set()
    for i, s in enumerate(sents):
        if ALIAS_RE.search(s.lower()):
            keep.update((i - 1, i, i + 1))
    idx = [i for i in sorted(keep) if 0 <= i < len(sents)]
    if not idx:
        return flat
    return " ".join(sents[i] for i in idx).strip()


# --- n-grams and distinctive-term scoring -----------------------------------
def ngrams(tokens, nmax=3):
    grams = []
    for n in range(1, nmax + 1):
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i:i + n]))
    return grams


def _ok_term(term):
    # drop any n-gram containing a brand/product token
    return not any(t in BRAND_TOKENS for t in term.split())


def distinctive_ngrams(group_docs, rest_docs, alpha0=1000.0, top=15, min_docs=3):
    """group_docs / rest_docs: lists of token-lists (one per document).
    Returns [(term, zscore, doc_count), ...] ranked by log-odds (Monroe et al.
    2008, informative Dirichlet prior), keeping only unigram..trigram terms that
    appear in >= min_docs group documents and are not brand/product names."""
    group_tf, rest_tf, group_df = Counter(), Counter(), Counter()
    for toks in group_docs:
        grams = ngrams(toks)
        group_tf.update(grams)
        for g in set(grams):
            group_df[g] += 1
    for toks in rest_docs:
        rest_tf.update(ngrams(toks))
    cands = [t for t, dfc in group_df.items() if dfc >= min_docs and _ok_term(t)]
    pooled = Counter()
    pooled.update(group_tf)
    pooled.update(rest_tf)
    total = sum(pooled.values()) or 1
    n_g = sum(group_tf.values())
    n_r = sum(rest_tf.values())
    scored = []
    for t in cands:
        a_w = alpha0 * (pooled[t] / total)
        num_g = group_tf.get(t, 0) + a_w
        num_r = rest_tf.get(t, 0) + a_w
        den_g = n_g + alpha0 - num_g
        den_r = n_r + alpha0 - num_r
        if den_g <= 0 or den_r <= 0:
            continue
        delta = math.log(num_g / den_g) - math.log(num_r / den_r)
        z = delta / math.sqrt(1.0 / num_g + 1.0 / num_r)
        scored.append((t, z, group_df[t]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top]


def _doc_window(r):
    w = str(r.get("window", "") or "").strip()
    return w if w else str(r.get("text", ""))


def _brands_of(r):
    return [b for b in str(r["brands_matched"]).split("|") if b]


def stage_clouds():
    """Layer 2 -- distinctive-term tables (single-brand docs), a brand-by-brand
    co-mention matrix (multi-brand docs), two targeted cut CSVs, and secondary
    cloud PNGs. Counting uses the windowed text, not the full comment."""
    import pandas as pd
    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    df = pd.read_csv(CORPUS_CSV).fillna("")
    if df.empty:
        log("[terms] corpus is empty -- nothing to analyze.")
        return

    # --- split single-brand vs multi-brand documents ------------------------
    single_rows = defaultdict(list)   # brand -> [(tokens, snippet, url), ...]
    multi_rows = []                   # [(brands, snippet, url), ...]
    for _, r in df.iterrows():
        bs = _brands_of(r)
        win = _doc_window(r)
        snippet = " ".join(win.split())
        toks = tokenize(win)
        if len(bs) == 1:
            single_rows[bs[0]].append((toks, snippet, str(r["url"])))
        elif len(bs) >= 2:
            multi_rows.append((bs, snippet, str(r["url"])))

    single_counts = {b: len(single_rows.get(b, [])) for b in BRANDS}
    thin = [(b, n) for b, n in single_counts.items() if n < 40]

    # --- per-brand distinctive terms (single-brand docs only) ---------------
    analyzed = {b: [row[0] for row in single_rows.get(b, [])] for b in BRANDS}
    tables = {}
    for b in BRANDS:
        group = analyzed[b]
        rest = [toks for ob in BRANDS if ob != b for toks in analyzed[ob]]
        scored = distinctive_ngrams(group, rest, top=15)
        rows_out = []
        for term, z, dc in scored:
            ex_snip, ex_url = "", ""
            needle = " " + term + " "
            for toks, snip, url in single_rows.get(b, []):
                if needle in (" " + " ".join(toks) + " "):
                    ex_snip, ex_url = snip[:160], url
                    break
            rows_out.append((term, z, dc, ex_snip, ex_url))
        tables[b] = rows_out

    # --- write markdown + CSV tables ----------------------------------------
    md = ["# Distinctive terms per brand",
          "_Single-brand docs only, windowed text, log-odds over 1-3 grams, "
          "brand/product names removed, terms in >= 3 docs._\n"]
    with open(os.path.join(OUT_DIR, "distinctive_terms.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["brand", "rank", "term", "logodds_z", "doc_count", "example", "url"])
        for b in BRANDS:
            n = single_counts[b]
            flag = "  **THIN (<40 docs)**" if n < 40 else ""
            md.append("## %s -- rests on %d single-brand docs%s\n" % (b, n, flag))
            md.append("| # | term | score | docs | example |")
            md.append("|---|------|-------|------|---------|")
            for i, (term, z, dc, snip, url) in enumerate(tables[b], 1):
                ex = ("[%s](%s)" % (snip.replace("|", "/")[:110], url)) if url else snip[:110]
                md.append("| %d | %s | %.2f | %d | %s |" % (i, term, z, dc, ex))
                wr.writerow([b, i, term, "%.3f" % z, dc, snip, url])
            if not tables[b]:
                md.append("| - | _(no terms with >= 3 docs)_ | | | |")
            md.append("")
    open(os.path.join(OUT_DIR, "distinctive_terms.md"), "w", encoding="utf-8").write("\n".join(md))
    log("[terms] wrote outputs/distinctive_terms.md + distinctive_terms.csv")

    # --- co-mention matrix (multi-brand docs) -------------------------------
    brands = list(BRANDS.keys())
    co = defaultdict(lambda: defaultdict(int))
    for bs, snip, url in multi_rows:
        uniq = sorted(set(bs))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                co[uniq[i]][uniq[j]] += 1
                co[uniq[j]][uniq[i]] += 1
    with open(os.path.join(OUT_DIR, "comention_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["brand"] + brands)
        for b in brands:
            wr.writerow([b] + [co[b].get(o, 0) for o in brands])
    log("[terms] wrote outputs/comention_matrix.csv (%d multi-brand docs)" % len(multi_rows))
    _comention_heatmap(co, brands)

    # --- targeted cuts ------------------------------------------------------
    _targeted_cut(df, ["lap", "lapping", "lapped", "alignment", "misalign"],
                  os.path.join(OUT_DIR, "cut_lapping_alignment.csv"))
    _targeted_cut(df, ["price", "priced", "expensive", "cheap", "worth", "$"],
                  os.path.join(OUT_DIR, "cut_price_terms.csv"))

    # --- secondary clouds (cheap) -------------------------------------------
    _draw_clouds(tables)

    # --- reporting ----------------------------------------------------------
    log("[terms] single-brand doc counts: %s" % single_counts)
    if thin:
        log("[terms] THIN brands (<40 single-brand docs): %s"
            % ", ".join("%s=%d" % (b, n) for b, n in thin))
    _print_brand_table("nightforce", tables.get("nightforce", []),
                       single_counts.get("nightforce", 0))


def _targeted_cut(df, tokens, path):
    toks_low = [t.lower() for t in tokens]

    def hits(low):
        found = []
        for t in toks_low:
            if t == "$":
                if "$" in low:
                    found.append("$")
            elif re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low):
                found.append(t)
        return found

    per_brand = defaultdict(lambda: {"count": 0, "urls": []})
    for _, r in df.iterrows():
        if not hits(_doc_window(r).lower()):
            continue
        for b in _brands_of(r):
            per_brand[b]["count"] += 1
            if len(per_brand[b]["urls"]) < 10:
                per_brand[b]["urls"].append(str(r["url"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["brand", "doc_count", "example_urls"])
        for b in BRANDS:
            d = per_brand.get(b, {"count": 0, "urls": []})
            wr.writerow([b, d["count"], " ".join(d["urls"])])
    log("[terms] wrote %s" % path)


def _comention_heatmap(co, brands):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log("[terms] (skip co-mention heat map) %s" % e)
        return
    M = np.array([[co[a].get(b, 0) for b in brands] for a in brands], dtype=float)
    plt.figure(figsize=(max(6, len(brands)), max(5, len(brands) * 0.8)))
    im = plt.imshow(M, cmap="Purples", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(brands)), brands, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(brands)), brands, fontsize=9)
    for i in range(len(brands)):
        for j in range(len(brands)):
            if M[i, j] > 0:
                plt.text(j, i, "%d" % M[i, j], ha="center", va="center", fontsize=8)
    plt.title("Brand co-mention (multi-brand docs)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comention_heat.png"), dpi=120)
    plt.close()
    log("[terms] wrote outputs/comention_heat.png")


def _draw_clouds(tables):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except Exception as e:
        log("[terms] (skip clouds) %s" % e)
        return
    made = 0
    for b, rows_out in tables.items():
        freqs = {term: z for term, z, dc, snip, url in rows_out if z > 0}
        if not freqs:
            continue
        wc = WordCloud(width=1000, height=600, background_color="white",
                       prefer_horizontal=0.9, collocations=False)
        wc.generate_from_frequencies(freqs)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Distinctive terms: %s" % b, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cloud_%s.png" % b), dpi=120)
        plt.close()
        made += 1
    log("[terms] produced %d secondary cloud PNGs." % made)


def _print_brand_table(brand, rows_out, n):
    log("\n=== Distinctive terms: %s  (rests on %d single-brand docs%s) ==="
        % (brand, n, "; THIN" if n < 40 else ""))
    log("%-3s %-24s %8s %5s  %s" % ("#", "term", "score", "docs", "example"))
    for i, (term, z, dc, snip, url) in enumerate(rows_out, 1):
        log("%-3d %-24s %8.2f %5d  %s" % (i, term[:24], z, dc, snip[:56]))
    if not rows_out:
        log("(no terms with >= 3 docs)")


# ============================================================================
#  STAGE 4  CLASSIFY (local Ollama; incremental + resumable)
# ============================================================================
VALID_SENT = {"positive", "negative", "neutral", "mixed"}
VALID_BRANDS = set(BRANDS) | {"other", "none"}


def ollama_classify(text):
    """Return (parsed_dict_or_None, raw_string)."""
    prompt = CLASSIFY_PROMPT.replace("{comment}", json.dumps(text)[1:-1])
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "format": "json", "options": {"temperature": 0}}
    try:
        r = requests.post(OLLAMA_URL + "/api/generate", json=payload, timeout=120)
    except Exception as e:
        return None, "REQUEST_ERROR: %s" % e
    if r.status_code != 200:
        return None, "HTTP %s" % r.status_code
    raw = r.json().get("response", "")
    parsed = _parse_label(raw)
    return parsed, raw


def _parse_label(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    attr = obj.get("attribute")
    sent = obj.get("sentiment")
    brands = obj.get("brands")
    if attr not in ATTRIBUTES or sent not in VALID_SENT:
        return None
    if not isinstance(brands, list):
        return None
    brands = [b for b in brands if b in VALID_BRANDS]
    return {"brands": brands, "attribute": attr, "sentiment": sent,
            "quotable": bool(obj.get("quotable", False))}


CLASSIFIED_FIELDS = ["id", "brands", "attribute", "sentiment", "quotable", "status", "model_raw"]


def _load_done_ids():
    done = set()
    if os.path.exists(CLASSIFIED_CSV):
        for row in csv.DictReader(open(CLASSIFIED_CSV, encoding="utf-8")):
            done.add(row["id"])
    return done


def stage_classify():
    import pandas as pd
    if not check_ollama_model():
        log("[classify] Ollama/model not ready -- skipping classification.")
        return
    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    df = pd.read_csv(CORPUS_CSV).fillna("")
    targets = df[df["bucket"] == "mounts"]
    done = _load_done_ids()
    new_file = not os.path.exists(CLASSIFIED_CSV)
    out = open(CLASSIFIED_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=CLASSIFIED_FIELDS)
    if new_file:
        w.writeheader()
    total = len(targets)
    todo = [r for _, r in targets.iterrows() if r["id"] not in done]
    log("[classify] %d mounts-bucket docs; %d already done; %d to classify."
        % (total, total - len(todo), len(todo)))
    n_ok = n_bad = 0
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        # classification runs on the windowed text, not the full comment
        text = str(r["window"]).strip() or str(r["text"])
        parsed, raw = ollama_classify(text)
        if parsed is None:  # retry once
            parsed, raw = ollama_classify(text)
        if parsed is None:
            n_bad += 1
            w.writerow({"id": r["id"], "brands": "", "attribute": "", "sentiment": "",
                        "quotable": "", "status": "unclassified", "model_raw": raw[:500]})
        else:
            n_ok += 1
            w.writerow({"id": r["id"], "brands": "|".join(parsed["brands"]),
                        "attribute": parsed["attribute"], "sentiment": parsed["sentiment"],
                        "quotable": parsed["quotable"], "status": "ok",
                        "model_raw": raw[:500]})
        out.flush()
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            log("  [classify] %d/%d  (~%.2fs/doc, ok=%d bad=%d)"
                % (i, len(todo), rate, n_ok, n_bad))
    out.close()
    attempted = n_ok + n_bad
    if attempted:
        log("[classify] valid-JSON rate this run: %.1f%% (%d/%d)"
            % (100.0 * n_ok / attempted, n_ok, attempted))
    _merge_labels_into_corpus()


def _merge_labels_into_corpus():
    import pandas as pd
    if not os.path.exists(CLASSIFIED_CSV):
        return
    df = pd.read_csv(CORPUS_CSV).fillna("")
    lab = pd.read_csv(CLASSIFIED_CSV).fillna("")
    lab_ok = lab[lab["status"] == "ok"].drop_duplicates("id", keep="last").set_index("id")
    for idx, r in df.iterrows():
        if r["id"] in lab_ok.index:
            L = lab_ok.loc[r["id"]]
            df.at[idx, "attribute"] = L["attribute"]
            df.at[idx, "sentiment"] = L["sentiment"]
            df.at[idx, "quotable"] = L["quotable"]
            df.at[idx, "model_raw"] = str(L["model_raw"])[:200]
    df.to_csv(CORPUS_CSV, index=False)
    log("[classify] merged labels back into corpus.csv")


# ============================================================================
#  STAGE 5  ANALYZE -- heat maps + segment CSVs
# ============================================================================
def _matrices(df):
    """Return (mention_counts, net_sentiment, totals) as brand x attribute dicts."""
    mention = defaultdict(lambda: defaultdict(int))
    pos = defaultdict(lambda: defaultdict(int))
    neg = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(lambda: defaultdict(int))
    for _, r in df.iterrows():
        attr = r.get("attribute", "")
        sent = r.get("sentiment", "")
        if attr not in ATTRIBUTES:
            continue
        brands = [b for b in str(r.get("brands_matched", "")).split("|") if b]
        for b in brands:
            mention[b][attr] += 1
            tot[b][attr] += 1
            if sent == "positive":
                pos[b][attr] += 1
            elif sent == "negative":
                neg[b][attr] += 1
    net = defaultdict(lambda: defaultdict(float))
    for b in tot:
        for a in tot[b]:
            t = tot[b][a]
            net[b][a] = (pos[b][a] - neg[b][a]) / t if t else 0.0
    return mention, net, tot


def _write_matrix_csv(mat, path, brands, attrs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["brand"] + attrs)
        for b in brands:
            wr.writerow([b] + [mat.get(b, {}).get(a, 0) for a in attrs])
    log("[analyze] wrote %s" % path)


def _heatmap(mat, brands, attrs, title, path, cmap, center_zero=False, fmt="%.0f"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[float(mat.get(b, {}).get(a, 0)) for a in attrs] for b in brands])
    plt.figure(figsize=(max(8, len(attrs) * 0.9), max(4, len(brands) * 0.6)))
    if center_zero:
        lim = max(1e-6, np.abs(M).max())
        im = plt.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = plt.imshow(M, cmap=cmap, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(attrs)), attrs, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(brands)), brands, fontsize=9)
    for i in range(len(brands)):
        for j in range(len(attrs)):
            v = M[i, j]
            if abs(v) > 1e-9:
                plt.text(j, i, fmt % v, ha="center", va="center", fontsize=7,
                         color="black")
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    log("[analyze] wrote %s" % path)


def stage_analyze():
    import pandas as pd
    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    df = pd.read_csv(CORPUS_CSV).fillna("")
    labeled = df[df["attribute"].isin(ATTRIBUTES)]
    if labeled.empty:
        log("[analyze] no classified rows yet -- run the 'classify' stage first.")
        return
    brands = list(BRANDS.keys())
    attrs = ATTRIBUTES
    mention, net, tot = _matrices(labeled)
    # mention share = mentions normalized to a share of total mentions
    grand = sum(sum(mention[b].values()) for b in mention) or 1
    share = defaultdict(lambda: defaultdict(float))
    for b in mention:
        for a in mention[b]:
            share[b][a] = mention[b][a] / grand
    _heatmap(share, brands, attrs, "Mention share (brand x attribute)",
             os.path.join(OUT_DIR, "heat_mention_share.png"), "Blues", fmt="%.2f")
    _heatmap(net, brands, attrs, "Net sentiment (pos-neg)/total",
             os.path.join(OUT_DIR, "heat_net_sentiment.png"), "RdYlGn",
             center_zero=True, fmt="%.2f")
    # segment splits
    for seg in ("precision", "tactical"):
        sub = labeled[labeled["segment"] == seg]
        if sub.empty:
            log("[analyze] (skip) no rows for segment %s" % seg)
            continue
        m2, n2, t2 = _matrices(sub)
        _write_matrix_csv(m2, os.path.join(OUT_DIR, "mention_share_%s.csv" % seg), brands, attrs)
        _write_matrix_csv(n2, os.path.join(OUT_DIR, "net_sentiment_%s.csv" % seg), brands, attrs)
    # stash the net matrix for the "surprising cells" printout
    _dump_net_for_summary(net, tot, brands, attrs)


def _dump_net_for_summary(net, tot, brands, attrs):
    cells = []
    for b in brands:
        for a in attrs:
            t = tot.get(b, {}).get(a, 0)
            if t >= 3:  # only cells with a little support
                cells.append((b, a, net[b][a], t))
    json.dump(cells, open(os.path.join(CACHE_DIR, "net_cells.json"), "w"))


# ============================================================================
#  STAGE 6  REPORT -- verbatims / validation sample / agreement.py / memo
# ============================================================================
AGREEMENT_PY = '''# -*- coding: utf-8 -*-
"""
agreement.py -- human-vs-model agreement scorer.

After you fill in the human_attribute and human_sentiment columns in
validation_sample.csv, run:   python agreement.py
It reports percent agreement SEPARATELY for attribute and for sentiment.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "validation_sample.csv")


def main():
    rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
    a_tot = a_ok = s_tot = s_ok = 0
    for r in rows:
        h_a = (r.get("human_attribute") or "").strip()
        h_s = (r.get("human_sentiment") or "").strip()
        if h_a:
            a_tot += 1
            if h_a == (r.get("model_attribute") or "").strip():
                a_ok += 1
        if h_s:
            s_tot += 1
            if h_s == (r.get("model_sentiment") or "").strip():
                s_ok += 1
    print("Labeled rows scored:")
    if a_tot:
        print("  attribute agreement: %.1f%% (%d/%d)" % (100.0 * a_ok / a_tot, a_ok, a_tot))
    else:
        print("  attribute agreement: (no human labels yet)")
    if s_tot:
        print("  sentiment agreement: %.1f%% (%d/%d)" % (100.0 * s_ok / s_tot, s_ok, s_tot))
    else:
        print("  sentiment agreement: (no human labels yet)")


if __name__ == "__main__":
    main()
'''


def stage_report():
    import pandas as pd
    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    df = pd.read_csv(CORPUS_CSV).fillna("")
    labeled = df[df["attribute"].isin(ATTRIBUTES)]

    # ---- verbatims.md : best quotable comments per notable cell ----
    notable = []
    if os.path.exists(os.path.join(CACHE_DIR, "net_cells.json")):
        cells = json.load(open(os.path.join(CACHE_DIR, "net_cells.json")))
        cells.sort(key=lambda c: (abs(c[2]), c[3]), reverse=True)
        notable = cells[:12]
    lines = ["# Verbatims -- best quotable comments per notable cell\n"]
    for b, a, netval, t in notable:
        lines.append("## %s / %s   (net sentiment %+.2f over %d docs)\n" % (b, a, netval, t))
        sub = labeled[(labeled["brands_matched"].str.contains(b, na=False)) &
                      (labeled["attribute"] == a)]
        q = sub[sub["quotable"].astype(str).str.lower().isin(["true", "1"])]
        pick = q if not q.empty else sub
        for _, r in pick.head(5).iterrows():
            snippet = " ".join(str(r["text"]).split())[:240]
            lines.append("- \"%s\"  \n  <%s>  (%s, %s)" %
                         (snippet, r["url"], r["source"], r["sentiment"]))
        lines.append("")
    if not notable:
        lines.append("_No classified cells with enough support yet. Run classify + analyze._")
    open(os.path.join(OUT_DIR, "verbatims.md"), "w", encoding="utf-8").write("\n".join(lines))
    log("[report] wrote outputs/verbatims.md")

    # ---- validation_sample.csv : stratified 50, empty human columns ----
    _write_validation_sample(labeled)

    # ---- agreement.py ----
    open(os.path.join(OUT_DIR, "agreement.py"), "w", encoding="utf-8").write(AGREEMENT_PY)
    log("[report] wrote outputs/agreement.py")

    # ---- memo.md ----
    _write_memo(df, labeled)


def _write_validation_sample(labeled):
    import pandas as pd
    if labeled.empty:
        open(os.path.join(OUT_DIR, "validation_sample.csv"), "w", encoding="utf-8").write(
            "id,text,url,model_brands,model_attribute,model_sentiment,"
            "human_attribute,human_sentiment\n")
        log("[report] wrote empty validation_sample.csv (nothing classified yet)")
        return
    # stratify across brand x sentiment
    buckets = defaultdict(list)
    for _, r in labeled.iterrows():
        b = (str(r["brands_matched"]).split("|") or ["none"])[0]
        buckets[(b, r["sentiment"])].append(r)
    picked = []
    # round-robin across strata until we have 50
    keys = list(buckets.keys())
    i = 0
    while len(picked) < 50 and any(buckets.values()):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop(0))
        i += 1
        if i > 5000:
            break
    rows = []
    for r in picked[:50]:
        rows.append({
            "id": r["id"],
            "text": " ".join(str(r["text"]).split())[:300],
            "url": r["url"],
            "model_brands": r["brands_matched"],
            "model_attribute": r["attribute"],
            "model_sentiment": r["sentiment"],
            "human_attribute": "",
            "human_sentiment": "",
        })
    pd.DataFrame(rows, columns=["id", "text", "url", "model_brands", "model_attribute",
                                "model_sentiment", "human_attribute", "human_sentiment"]
                 ).to_csv(os.path.join(OUT_DIR, "validation_sample.csv"), index=False)
    log("[report] wrote outputs/validation_sample.csv (%d rows)" % len(rows))


def _write_memo(df, labeled):
    stats = write_corpus_stats(df)
    valid_rate = ""
    if os.path.exists(CLASSIFIED_CSV):
        import pandas as pd
        lab = pd.read_csv(CLASSIFIED_CSV).fillna("")
        if len(lab):
            ok = (lab["status"] == "ok").sum()
            valid_rate = "%.1f%% (%d/%d)" % (100.0 * ok / len(lab), ok, len(lab))
    memo = """# Nightforce Market Voice -- Findings Memo (one page)

## What this is
A local, one-off analysis of public customer conversation about scope mounts,
rings, and bases across seven brands (Reddit + YouTube), classified by a local
model and summarized as differential word clouds and brand-by-attribute heat maps.

## Corpus at a glance
```
{stats}
```
Valid-JSON classification rate: {valid_rate}

## Three findings
1. TODO -- <lead finding; cite a heat-map cell and a verbatim>
2. TODO -- <second finding>
3. TODO -- <third finding>

## One limitation, named unprompted
TODO -- <e.g. self-selected forum voices; small n in some cells; single-model labels>

## How I validated
I audited my own labels the way I'd audit a build: a stratified 50-row sample
(validation_sample.csv) hand-labeled and scored with agreement.py, reported
separately for attribute and sentiment. TODO -- <insert the two agreement numbers>

## Method notes
- Filter: a document survives only if it matches >=1 brand alias AND >=1 category token.
- Clouds: log-odds with an informative Dirichlet prior vs the corpus baseline.
- Sentiment: net = (positive - negative) / total per brand x attribute cell.
""".format(stats=stats, valid_rate=valid_rate or "TODO -- run classify")
    open(os.path.join(OUT_DIR, "memo.md"), "w", encoding="utf-8").write(memo)
    log("[report] wrote outputs/memo.md")


# ============================================================================
#  SURPRISING-CELLS PRINTOUT
# ============================================================================
def print_surprises():
    p = os.path.join(CACHE_DIR, "net_cells.json")
    if not os.path.exists(p):
        return
    cells = json.load(open(p))
    if not cells:
        return
    cells.sort(key=lambda c: (abs(c[2]), c[3]), reverse=True)
    log("\nThree most surprising net-sentiment cells (|net| x support):")
    for b, a, netval, t in cells[:3]:
        log("  %-16s %-22s net=%+.2f  (n=%d)" % (b, a, netval, t))
    stats_path = os.path.join(OUT_DIR, "corpus_stats.txt")
    if os.path.exists(stats_path):
        log("\n" + open(stats_path, encoding="utf-8").read())


# ============================================================================
#  MAIN
# ============================================================================
def run_stage(stage):
    if stage in ("collect", "all"):
        stage_collect()
    if stage in ("corpus", "all"):
        if not check_dependencies(need_plotting=False):
            return
        stage_corpus()
    if stage in ("clouds", "all"):
        if not check_dependencies(need_plotting=True):
            return
        stage_clouds()
    if stage in ("classify", "all"):
        if not check_dependencies(need_plotting=False):
            return
        stage_classify()
    if stage in ("analyze", "all"):
        if not check_dependencies(need_plotting=True):
            return
        stage_analyze()
    if stage in ("report", "all"):
        if not check_dependencies(need_plotting=False):
            return
        stage_report()
    if stage == "all":
        print_surprises()


def main():
    stage = STAGE
    if len(sys.argv) > 1:
        stage = sys.argv[1].strip().lower()
    valid = {"collect", "corpus", "clouds", "classify", "analyze", "report", "all"}
    if stage not in valid:
        log("Unknown stage %r. Choose one of: %s" % (stage, ", ".join(sorted(valid))))
        return
    log("=== Nightforce Market Voice :: stage = %s ===" % stage)
    run_stage(stage)
    log("=== done (stage = %s) ===" % stage)


if __name__ == "__main__":
    main()
