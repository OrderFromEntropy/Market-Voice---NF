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
import html
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

# --- Attribute taxonomy for classification. Collapsed from the original 11-way
# set to 7 domain buckets after validation showed the fine-grained split hurt both
# human and model agreement. Re-validate (agreement.py) after any change.
ATTRIBUTES = [
    "zero_retention",            # holding/returning to zero, QD repeatability, zero shift
    "build_quality",             # durability, toughness, materials, machining, finish
    "weight",                    # how heavy or light the mount is
    "price_value",               # cost, worth, expensive/cheap, value for money
    "fit_and_install",           # ring height, clearance, cantilever/offset, torque, install
    "availability_and_service",  # in stock / lead time / where to buy, CS, warranty, returns
    "other",                     # anything else, or too vague to place
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
    # --- widened pull: brand + model/problem queries for more single-brand text
    "nightforce x-treme duty ultramount", "nightforce ultralite unimount",
    "spuhr ideal scope mount", "badger ordnance c1 mount review",
    "reptilia dot mount review", "geissele super precision mount review",
    "warne skyline rings review", "leupold mark 4 rings review",
    "scope ring lapping", "return to zero scope mount",
    "34mm one piece scope mount", "picatinny scope mount review",
]
# YouTube segment tag: "precision" if the video title matches long range / precision / PRS,
# else "tactical" if it matches lpvo / ar15 / carbine, else "mixed".

# --- Pacing and identification.
REDDIT_SECONDS_BETWEEN = 2.0
USER_AGENT = "personal-market-research/0.1 (one-off student project; contact in profile)"
YOUTUBE_QUOTA_BUDGET = 5000
# Reddit source. Reddit killed off free anonymous access to its own .json
# endpoints, so "direct" now 403s. "pullpush" uses the PullPush public archive
# (api.pullpush.io) of Reddit's public data -- no key, no dev app, and it reaches
# deep history. Set to "direct" only if you have a working authenticated setup.
# Heat-map / verbatim cells with fewer than this many supporting docs are treated
# as too thin to headline (kept in the matrices, flagged in findings_candidates).
MIN_CELL_SUPPORT = 10
# Optional recency cut applied to the corpus before the analysis stages (clouds,
# analyze, report). Set via the --since YYYY-MM-DD command-line flag. None = no cut.
SINCE = None
REDDIT_SOURCE = "pullpush"          # "pullpush" | "direct"
PULLPUSH_PAGES = 5                  # pages of 100 comments per brand query
PULLPUSH_SECONDS_BETWEEN = 2.0      # polite pacing for the PullPush archive
PULLPUSH_MAX_RETRIES = 4            # retries on HTTP 429 / transient errors
PULLPUSH_BACKOFF = 5               # base backoff seconds (5, 10, 20, 40)
# Widened-pull knobs. search.list costs 100 units regardless of result count, so
# 50 results is free extra coverage. Each comment page costs 1 unit and yields up
# to 100 comments, so paginating a few pages per video is the cheapest way to add
# text. Collection stops when it hits YOUTUBE_QUOTA_BUDGET or TARGET_COMMENTS.
YT_SEARCH_RESULTS = 50        # videos per search query (max 50)
YT_COMMENT_PAGES = 3          # comment pages to pull per video (100 comments each)

# --- Classification prompt for the local model. Temperature 0. JSON only.
CLASSIFY_PROMPT = """You label short comments about riflescope MOUNTS, rings, and bases.
Return ONLY a JSON object, no prose, exactly matching this schema:
{"brands": [zero or more of: nightforce, leupold, reptilia, badger_ordnance, spuhr,
geissele, warne, other, none],
"attribute": one of [zero_retention, build_quality, weight, price_value,
fit_and_install, availability_and_service, other],
"sentiment": one of [positive, negative, neutral, mixed], "quotable": true or false}

Pick the SINGLE attribute the comment is MOST about:
- zero_retention: holding or returning to zero, repeatability, QD lockup, zero shift after remounting.
- build_quality: durability, toughness, materials, machining, finish, how solid it feels.
- weight: how heavy or light the mount is.
- price_value: cost, worth, expensive/cheap, value for money.
- fit_and_install: ring height, objective clearance, cantilever/offset, torque, install experience, whether it fits.
- availability_and_service: in stock / lead time / where to buy, customer service, warranty, returns.
- other: anything else, or too vague to place (general questions, build lists, chit-chat).

sentiment is the commenter's stance toward the MOUNT. WATCH SARCASM: "only lost zero
twice, great value" is negative. Use neutral for a plain fact or a question; mixed only
when clear positives AND negatives appear together. quotable is true only if the comment
is vivid, specific, and under about 60 words.

Comment: "Swapped to a Spuhr and my zero survives barrel swaps, worth every penny."
{"brands": ["spuhr"], "attribute": "zero_retention", "sentiment": "positive", "quotable": true}

Comment: "The badger c1 is a tank but man it is heavy on a 6lb hunting rig"
{"brands": ["badger_ordnance"], "attribute": "weight", "sentiment": "mixed", "quotable": true}

Comment: "my warne only lost zero twice this season, great value lol"
{"brands": ["warne"], "attribute": "zero_retention", "sentiment": "negative", "quotable": true}

Comment: "Machining on the Reptilia is gorgeous, feels milled from billet."
{"brands": ["reptilia"], "attribute": "build_quality", "sentiment": "positive", "quotable": true}

Comment: "Been waiting 3 months for the geissele mount to restock, ridiculous."
{"brands": ["geissele"], "attribute": "availability_and_service", "sentiment": "negative", "quotable": true}

Comment: "Leupold CS replaced my cross-slot base no charge, shipped in 2 days."
{"brands": ["leupold"], "attribute": "availability_and_service", "sentiment": "positive", "quotable": true}

Comment: "What height rings do I need for a 56mm objective on a 700?"
{"brands": ["none"], "attribute": "fit_and_install", "sentiment": "neutral", "quotable": false}

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
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_URL = "http://localhost:11434"
# Classification prompts are short (a windowed comment + examples ~ under 1500
# tokens), so a small context window is plenty -- and a smaller window frees VRAM,
# letting more of a 14B model sit on the GPU instead of spilling to CPU.
OLLAMA_NUM_CTX = 2048

# Paste your YouTube Data API key here to hard-wire it (optional). If left blank,
# the script looks for YT_API_KEY in the environment, then a .env file next to
# this script, then your Desktop "analytics - nightforce.txt".
YT_API_KEY = ""

# How many comments to target overall before we stop pulling (soft cap; keeps a
# cold run inside a couple evenings). Set to None to pull everything found.
TARGET_COMMENTS = 8000

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
    """Return True if OLLAMA_MODEL is reachable; print available tags if not.
    Always queries Ollama live -- the installed-model list is dynamic state and
    must never be read from the disk cache (a stale snapshot would hide models
    you pulled after the first run)."""
    data = None
    try:
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=15)
        if r.status_code == 200:
            data = r.json()
    except Exception:
        data = None
    if not data or "models" not in data:
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
            params={"key": api_key, "q": q, "part": "snippet", "type": "video",
                    "maxResults": YT_SEARCH_RESULTS, "relevanceLanguage": "en"},
        )
        if not cached:
            _quota_used["units"] += 100
        if not data or "items" not in data:
            if data and data.get("__error__"):
                log("  [youtube] search %s (%s) for %r"
                    % (data["__error__"], data.get("__reason__", ""), q))
            continue
        for it in data["items"]:
            vid = it.get("id", {}).get("videoId")
            if not vid or vid in seen_videos:
                continue
            seen_videos.add(vid)
            title = it.get("snippet", {}).get("title", "")
            seg = yt_segment_for_title(title)
            # pull up to YT_COMMENT_PAGES pages of comments for this video
            page_token = None
            for _page in range(YT_COMMENT_PAGES):
                if _quota_used["units"] + 1 > YOUTUBE_QUOTA_BUDGET:
                    log("  [youtube] quota budget reached; stopping comments.")
                    return docs
                params = {"key": api_key, "videoId": vid, "part": "snippet",
                          "maxResults": 100, "textFormat": "plainText", "order": "relevance"}
                if page_token:
                    params["pageToken"] = page_token
                cdata, ccached = cached_get_json(YT + "/commentThreads", params=params)
                if not ccached:
                    _quota_used["units"] += 1
                if not cdata or "items" not in cdata:
                    break  # comments disabled or error -> next video
                for c in cdata["items"]:
                    sn = c.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    docs.append({
                        "id": "yt_" + c.get("id", vid),
                        "source": "youtube",
                        "subreddit_or_video": vid,
                        "segment": seg,
                        "url": "https://www.youtube.com/watch?v=%s&lc=%s" % (vid, c.get("id", "")),
                        "created_utc": iso_from_epoch(_yt_epoch(sn.get("publishedAt"))),
                        "text": sn.get("textDisplay", ""),
                    })
                page_token = cdata.get("nextPageToken")
                if not page_token:
                    break  # no more comment pages
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
#  STAGE 1c  COLLECT -- REDDIT via PULLPUSH archive (no auth, deep history)
# ============================================================================
PULLPUSH_COMMENT = "https://api.pullpush.io/reddit/search/comment/"


def _pp_permalink(c):
    """Best-effort reconstruction of a Reddit URL for a PullPush comment."""
    p = c.get("permalink")
    if p:
        return p if str(p).startswith("http") else "https://www.reddit.com" + p
    link = str(c.get("link_id", "")).replace("t3_", "")
    cid = c.get("id", "")
    if link and cid:
        return "https://www.reddit.com/comments/%s/_/%s/" % (link, cid)
    sub = c.get("subreddit", "")
    return "https://www.reddit.com/r/%s/" % sub if sub else "https://www.reddit.com/"


def _pullpush_get(params, made):
    """GET one PullPush page with exponential backoff on 429 / transient errors.
    Returns (data_or_None, updated_made_count)."""
    data = None
    for attempt in range(PULLPUSH_MAX_RETRIES + 1):
        paced = 0 if cached_exists(PULLPUSH_COMMENT, params) else PULLPUSH_SECONDS_BETWEEN
        data, cached = cached_get_json(PULLPUSH_COMMENT, headers=REDDIT_HEADERS,
                                       params=params, pace_seconds=paced)
        if not cached:
            made += 1
        transient = (data is None) or (isinstance(data, dict)
                    and data.get("__error__") in (429, 500, 502, 503, 504))
        if not transient:
            return data, made
        if attempt < PULLPUSH_MAX_RETRIES:
            wait = PULLPUSH_BACKOFF * (2 ** attempt)
            code = data.get("__error__", "network") if data else "network"
            log("  [pullpush] %s -- backoff %ds (retry %d/%d)"
                % (code, wait, attempt + 1, PULLPUSH_MAX_RETRIES))
            time.sleep(wait)
    return data, made


def collect_pullpush():
    """Pull public Reddit comments from the PullPush archive. Queries '<brand>
    mount' and '<brand> rings' per brand, paginated backwards through history via
    the `before` cursor. Segment is tagged from each comment's subreddit."""
    docs = []
    made = 0
    seg_map = {k.lower(): v for k, v in SUBREDDITS.items()}
    for brand, name in PRIMARY_NAME.items():
        for suffix in ("mount", "rings"):
            q = "%s %s" % (name, suffix)
            before = None
            for _pg in range(PULLPUSH_PAGES):
                params = {"q": q, "size": 100, "sort": "desc", "sort_type": "created_utc"}
                if before:
                    params["before"] = before
                data, made = _pullpush_get(params, made)
                if not data or "data" not in data:
                    if data and data.get("__error__"):
                        log("  [pullpush] gave up page for %r after retries (HTTP %s)"
                            % (q, data["__error__"]))
                    break
                items = data.get("data", [])
                if not items:
                    break
                for c in items:
                    body = (c.get("body") or "").strip()
                    if not body or body in ("[removed]", "[deleted]"):
                        continue
                    sub = c.get("subreddit", "") or ""
                    docs.append({
                        "id": "pp_" + str(c.get("id", "")),
                        "source": "reddit",
                        "subreddit_or_video": sub,
                        "segment": seg_map.get(sub.lower(), "mixed"),
                        "url": _pp_permalink(c),
                        "created_utc": iso_from_epoch(c.get("created_utc")),
                        "text": body,
                    })
                before = items[-1].get("created_utc")
                if not before:
                    break
    log("  [pullpush] collected %d raw comments in %d requests." % (len(docs), made))
    return docs


# ============================================================================
#  STAGE 1  COLLECT (driver) -> writes cache; returns raw docs
# ============================================================================
def stage_collect():
    log("[collect] YouTube ...")
    api_key = resolve_api_key()
    if api_key:
        log("  [youtube] API key resolved (ends ...%s)." % api_key[-4:])
    yt_docs = collect_youtube(api_key)
    log("[collect] Reddit via %s ..." % REDDIT_SOURCE)
    rd_docs = collect_pullpush() if REDDIT_SOURCE == "pullpush" else collect_reddit()
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
        # decode HTML entities (&amp; -> &) so near-duplicates collapse and text is clean
        text = html.unescape(text)
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
            "model_brands": "",
            "model_raw": "",
        })
    df = pd.DataFrame(rows, columns=[
        "id", "source", "subreddit_or_video", "segment", "url", "created_utc",
        "text", "window", "brands_matched", "bucket", "attribute", "sentiment",
        "quotable", "model_brands", "model_raw"])
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
    # commit rates -- how often the model assigns a real attribute rather than
    # "other" (matches the audit / stats_update.txt: denominator = mounts docs).
    mounts = df[df["bucket"] == "mounts"]
    has_labels = mounts["attribute"].isin(ATTRIBUTES).any() if len(mounts) else False
    doc_total = len(mounts) if has_labels else 0
    doc_commit = int((mounts["attribute"] != "other").sum()) if has_labels else 0
    attr_total = attr_commit = 0
    if has_labels:
        for _, r in mounts.iterrows():
            committed = r["attribute"] != "other"
            for _b in _model_brands(r):
                attr_total += 1
                if committed:
                    attr_commit += 1
    if doc_total:
        lines.append("document-level attribute commit rate    : %.1f%% (%d/%d)"
                     % (100.0 * doc_commit / doc_total, doc_commit, doc_total))
    else:
        lines.append("document-level attribute commit rate    : n/a (run classify)")
    if attr_total:
        lines.append("attribution-level attribute commit rate : %.1f%% (%d/%d)"
                     % (100.0 * attr_commit / attr_total, attr_commit, attr_total))
    else:
        lines.append("attribution-level attribute commit rate : n/a (run classify)")
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

# Adjacent products that co-occur in Reddit "build list" posts but are NOT scope
# mounts -- scope brands/models, red dots, bipods, triggers, chassis/actions, and
# a few scope-part words. Stripped from every brand's distinctive-term table so
# what remains is mount vocabulary. Mount-relevant tokens (moa, mil, mrad, 20moa,
# 30mm/34mm/35mm, arca, picatinny, cant) are deliberately NOT listed here.
# Edit freely -- this is a denylist of noise, not part of the study taxonomy.
ADJACENT_PRODUCTS = set("""
atacr nx8 nxs nxr shv beast ataccr
vortex razor viper pst athlon cronus midas ares helos argos talos
arken swfa burris xtr veracity signature bushnell match-pro dmr
sig tango whiskey sierra steiner march kahles schmidt s&b pmii zco tangent theta
trijicon acog credo huron tenmile ventus eotech vudu maven riton monstrum
aimpoint holosun romeo delta stryker leupold-mark mark5 mk5 vx3 vx5 vx6 hamr
harris atlas ckye ckyepod accutac magpod tacpod talon talons
triggertech timney hellfire jewell
mdt magpul aics archangel krg manners mcmillan grayboe foundation xlr
tikka bergara defiance terminus zermatt bighorn curtis kelbly mausingfield borden
savage howa ruger remington rem 700 m1a ar15 ar10 ar47 mk12
creedmoor grendel prc arc nato
glass reticle illumination turret turrets parallax magnification objective
area419 arc419 nrl prs precisionrifle
""".split())


def add_adjacent_products(words):
    """Extend the adjacent-product denylist at runtime (lowercased tokens)."""
    ADJACENT_PRODUCTS.update(w.lower() for w in words)


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


# Per-brand alias matcher (used to require a quote actually names its brand).
BRAND_ALIAS_RE = {b: re.compile("|".join(
    r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])" for a in al))
    for b, al in BRANDS.items()}


def text_has_brand_alias(text, brand):
    return bool(BRAND_ALIAS_RE[brand].search((text or "").lower()))


def _apply_since(df):
    """Filter a corpus DataFrame to created_utc >= SINCE (a YYYY-MM-DD string).
    Undated rows are dropped when a cut is active. No-op when SINCE is None."""
    if not SINCE:
        return df
    before = len(df)
    kept = df[df["created_utc"].astype(str) >= SINCE].copy()
    log("[since] recency cut created_utc >= %s : %d -> %d docs"
        % (SINCE, before, len(kept)))
    return kept


def _ok_term(term):
    # drop any n-gram containing a brand name, product name, or scope-model token
    toks = term.split()
    if any(t in BRAND_TOKENS for t in toks):
        return False
    if any(t in ADJACENT_PRODUCTS for t in toks):
        return False
    return True


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
    df = _apply_since(df)
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
               "format": "json", "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX}}
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
    if "model_brands" not in df.columns:
        df["model_brands"] = ""
    lab = pd.read_csv(CLASSIFIED_CSV).fillna("")
    lab_ok = lab[lab["status"] == "ok"].drop_duplicates("id", keep="last").set_index("id")
    for idx, r in df.iterrows():
        if r["id"] in lab_ok.index:
            L = lab_ok.loc[r["id"]]
            df.at[idx, "attribute"] = L["attribute"]
            df.at[idx, "sentiment"] = L["sentiment"]
            df.at[idx, "quotable"] = L["quotable"]
            df.at[idx, "model_brands"] = str(L["brands"])   # brands the MODEL judged the comment to be about
            df.at[idx, "model_raw"] = str(L["model_raw"])[:200]
    df.to_csv(CORPUS_CSV, index=False)
    log("[classify] merged labels back into corpus.csv")


def _model_brands(r):
    """Brands the classifier judged the comment to be ABOUT (drops none/other).
    Falls back to the keyword match only if the model gave nothing usable."""
    raw = str(r.get("model_brands", "") or "")
    hits = [b for b in raw.split("|") if b in BRANDS]
    if hits:
        return hits
    # fallback: keyword match (older corpora without model_brands)
    return [b for b in str(r.get("brands_matched", "")).split("|") if b in BRANDS]


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
        # attribute the cell to the brand(s) the MODEL judged the comment to be
        # about -- not every brand a build-list post happens to name
        brands = _model_brands(r)
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


def _heatmap(mat, brands, attrs, title, path, cmap, support=None, gate=0,
             center_zero=False, fmt="%.0f"):
    """Cells with support < gate are masked gray and labeled 'n<10'; every valid
    cell is annotated with its value and (n=X)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[float(mat.get(b, {}).get(a, 0)) for a in attrs] for b in brands])
    N = np.array([[int(support.get(b, {}).get(a, 0)) if support else 10 ** 9
                   for a in attrs] for b in brands])
    masked = N < gate
    Mm = np.ma.masked_where(masked, M)
    base = plt.get_cmap(cmap).copy()
    base.set_bad("lightgray")
    plt.figure(figsize=(max(9, len(attrs) * 1.05), max(4, len(brands) * 0.65)))
    if center_zero:
        valid = M[~masked]
        lim = max(1e-6, np.abs(valid).max() if valid.size else 1.0)
        im = plt.imshow(Mm, cmap=base, vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = plt.imshow(Mm, cmap=base, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(attrs)), attrs, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(brands)), brands, fontsize=9)
    for i in range(len(brands)):
        for j in range(len(attrs)):
            if masked[i, j]:
                plt.text(j, i, "n<10", ha="center", va="center", fontsize=6,
                         color="dimgray")
            else:
                plt.text(j, i, (fmt % M[i, j]) + "\n(n=%d)" % N[i, j],
                         ha="center", va="center", fontsize=6, color="black")
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    log("[analyze] wrote %s" % path)


def stage_analyze():
    import pandas as pd
    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    _merge_labels_into_corpus()   # ensure model labels (incl. model_brands) are synced
    df = pd.read_csv(CORPUS_CSV).fillna("")
    df = _apply_since(df)
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
    _heatmap(share, brands, attrs, "Mention share (brand x attribute, n>=%d)" % MIN_CELL_SUPPORT,
             os.path.join(OUT_DIR, "heat_mention_share.png"), "Blues",
             support=tot, gate=MIN_CELL_SUPPORT, fmt="%.2f")
    _heatmap(net, brands, attrs, "Net sentiment (pos-neg)/total, n>=%d" % MIN_CELL_SUPPORT,
             os.path.join(OUT_DIR, "heat_net_sentiment.png"), "RdYlGn",
             support=tot, gate=MIN_CELL_SUPPORT, center_zero=True, fmt="%.2f")
    # overall matrices as CSV (counts, net sentiment, support) -- easy to read/cite
    _write_matrix_csv(mention, os.path.join(OUT_DIR, "mention_counts_all.csv"), brands, attrs)
    _write_matrix_csv(net, os.path.join(OUT_DIR, "net_sentiment_all.csv"), brands, attrs)
    # segment splits
    for seg in ("precision", "tactical"):
        sub = labeled[labeled["segment"] == seg]
        if sub.empty:
            log("[analyze] (skip) no rows for segment %s" % seg)
            continue
        m2, n2, t2 = _matrices(sub)
        _write_matrix_csv(m2, os.path.join(OUT_DIR, "mention_share_%s.csv" % seg), brands, attrs)
        _write_matrix_csv(n2, os.path.join(OUT_DIR, "net_sentiment_%s.csv" % seg), brands, attrs)
    # per-brand sentiment distribution -- surfaces the positive skew honestly
    _write_sentiment_distribution(labeled, brands)
    # robust cells only -> findings candidates + the "surprising cells" printout
    _dump_net_for_summary(net, tot, brands, attrs)
    _write_findings_candidates(mention, net, tot, brands, attrs)


def _write_sentiment_distribution(labeled, brands):
    dist = {b: Counter() for b in brands}
    for _, r in labeled.iterrows():
        s = r.get("sentiment", "")
        for b in _model_brands(r):
            dist[b][s] += 1
    path = os.path.join(OUT_DIR, "sentiment_distribution.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["brand", "positive", "negative", "neutral", "mixed", "total", "pct_positive"])
        for b in brands:
            c = dist[b]
            tot = sum(c.values())
            pct = (100.0 * c["positive"] / tot) if tot else 0.0
            wr.writerow([b, c["positive"], c["negative"], c["neutral"], c["mixed"],
                         tot, "%.0f%%" % pct])
    log("[analyze] wrote %s" % path)


def _dump_net_for_summary(net, tot, brands, attrs):
    cells = []
    for b in brands:
        for a in attrs:
            t = tot.get(b, {}).get(a, 0)
            if t >= MIN_CELL_SUPPORT:   # only cells robust enough to headline
                cells.append((b, a, net[b][a], t))
    json.dump(cells, open(os.path.join(CACHE_DIR, "net_cells.json"), "w"))


def _write_findings_candidates(mention, net, tot, brands, attrs):
    """Rank the robust cells (support >= MIN_CELL_SUPPORT) so real findings are
    obvious instead of small-n flukes."""
    robust = []
    for b in brands:
        for a in attrs:
            t = tot.get(b, {}).get(a, 0)
            if t >= MIN_CELL_SUPPORT:
                robust.append((b, a, mention[b].get(a, 0), net[b].get(a, 0.0), t))
    lines = ["# Findings candidates",
             "_Only cells with >= %d supporting docs. Brand = the model's judgment "
             "of what each comment is about. Net sentiment is skewed positive across "
             "the board (enthusiast forums) -- compare brands, don't read absolutes._\n"
             % MIN_CELL_SUPPORT]

    lines.append("## Most talked-about cells (by mention volume)\n")
    lines.append("| brand | attribute | mentions | net sentiment | n |")
    lines.append("|---|---|---|---|---|")
    for b, a, m_, nv, t in sorted(robust, key=lambda x: x[2], reverse=True)[:12]:
        lines.append("| %s | %s | %d | %+.2f | %d |" % (b, a, m_, nv, t))

    lines.append("\n## Most positive cells (robust)\n")
    lines.append("| brand | attribute | net sentiment | mentions | n |")
    lines.append("|---|---|---|---|---|")
    for b, a, m_, nv, t in sorted(robust, key=lambda x: x[3], reverse=True)[:8]:
        lines.append("| %s | %s | %+.2f | %d | %d |" % (b, a, nv, m_, t))

    lines.append("\n## Least positive cells (robust) -- where criticism concentrates\n")
    lines.append("| brand | attribute | net sentiment | mentions | n |")
    lines.append("|---|---|---|---|---|")
    for b, a, m_, nv, t in sorted(robust, key=lambda x: x[3])[:8]:
        lines.append("| %s | %s | %+.2f | %d | %d |" % (b, a, nv, m_, t))

    open(os.path.join(OUT_DIR, "findings_candidates.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    log("[analyze] wrote outputs/findings_candidates.md (%d robust cells)" % len(robust))


# ============================================================================
#  STAGE 6  REPORT -- verbatims / validation sample / agreement.py / memo
# ============================================================================
AGREEMENT_PY = '''# -*- coding: utf-8 -*-
"""
agreement.py -- human-vs-model agreement scorer + confusion diagnostics.

After you fill in the human_attribute and human_sentiment columns in
validation_sample.csv, run:   python agreement.py
Reports overall agreement AND where the model fails, so refinement is targeted.
"""
import csv, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "validation_sample.csv")


def pct(n, d):
    return "%.1f%% (%d/%d)" % (100.0 * n / d, n, d) if d else "(no labels)"


def main():
    rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
    A = [((r.get("human_attribute") or "").strip(), (r.get("model_attribute") or "").strip())
         for r in rows if (r.get("human_attribute") or "").strip()]
    S = [((r.get("human_sentiment") or "").strip(), (r.get("model_sentiment") or "").strip())
         for r in rows if (r.get("human_sentiment") or "").strip()]

    print("=== Overall agreement ===")
    print("  attribute:", pct(sum(h == m for h, m in A), len(A)))
    print("  sentiment:", pct(sum(h == m for h, m in S), len(S)))

    if A:
        print("\\n=== Attribute diagnostics ===")
        other = sum(1 for h, m in A if m == "other")
        print("  model said 'other':", pct(other, len(A)))
        committed = [(h, m) for h, m in A if m != "other"]
        print("  agreement when model committed (model != other):",
              pct(sum(h == m for h, m in committed), len(committed)))
        mis = Counter((h, m) for h, m in A if h != m)
        print("  top human -> model mismatches:")
        for (h, m), c in mis.most_common(10):
            print("     %-26s -> %-26s x%d" % (h or "(blank)", m or "(blank)", c))

    if S:
        print("\\n=== Sentiment confusion (rows = your label, cols = model) ===")
        labels = ["positive", "negative", "neutral", "mixed"]
        conf = defaultdict(Counter)
        for h, m in S:
            conf[h][m] += 1
        print("  %-12s %s" % ("you\\\\model", " ".join("%-9s" % l for l in labels)))
        for h in labels:
            print("  %-12s %s" % (h, " ".join("%-9d" % conf[h][m] for m in labels)))


if __name__ == "__main__":
    main()
'''


def stage_report():
    import pandas as pd
    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    _merge_labels_into_corpus()   # ensure model labels (incl. model_brands) are synced
    df = pd.read_csv(CORPUS_CSV).fillna("")
    df = _apply_since(df)
    labeled = df[df["attribute"].isin(ATTRIBUTES)]

    # Verbatim eligibility: the comment must be about EXACTLY ONE study brand (per
    # the model) AND its text must actually name that brand.
    def _eligible(r, b):
        mb = _model_brands(r)
        if len(mb) != 1 or mb[0] != b:
            return False
        return text_has_brand_alias(str(r.get("window") or r["text"]), b)

    # ---- verbatims.md : best quotable comments per notable (robust) cell ----
    notable = []
    if os.path.exists(os.path.join(CACHE_DIR, "net_cells.json")):
        cells = json.load(open(os.path.join(CACHE_DIR, "net_cells.json")))
        cells.sort(key=lambda c: (abs(c[2]), c[3]), reverse=True)
        notable = cells[:12]
    lines = ["# Verbatims -- best quotable comments per notable cell",
             "",
             "**HAND-VERIFY EVERY QUOTE AT ITS LINK**",
             "",
             "_Cells with >= %d docs. A quote is eligible only if the model judged the "
             "comment to be about exactly one study brand AND the text names that "
             "brand; snippet is the windowed text._\n" % MIN_CELL_SUPPORT]
    for b, a, netval, t in notable:
        lines.append("## %s / %s   (net sentiment %+.2f over %d docs)\n" % (b, a, netval, t))
        sub = labeled[(labeled["attribute"] == a)
                      & labeled.apply(lambda r: _eligible(r, b), axis=1)]
        q = sub[sub["quotable"].astype(str).str.lower().isin(["true", "1"])]
        pick = q if not q.empty else sub
        for _, r in pick.head(5).iterrows():
            snippet = " ".join(str(r.get("window") or r["text"]).split())[:240]
            lines.append("- \"%s\"  \n  <%s>  (%s, %s)" %
                         (snippet, r["url"], r["source"], r["sentiment"]))
        lines.append("")
    if not notable:
        lines.append("_No cells with >= %d supporting docs yet. Run classify + analyze._"
                     % MIN_CELL_SUPPORT)
    open(os.path.join(OUT_DIR, "verbatims.md"), "w", encoding="utf-8").write("\n".join(lines))
    log("[report] wrote outputs/verbatims.md")

    # ---- validation_sample.csv : stratified 50, empty human columns ----
    _write_validation_sample(labeled)
    _write_label_guide()

    # ---- agreement.py ----
    open(os.path.join(OUT_DIR, "agreement.py"), "w", encoding="utf-8").write(AGREEMENT_PY)
    log("[report] wrote outputs/agreement.py")

    # ---- memo.md ----
    _write_memo(df, labeled)


def _write_label_guide():
    """Reference so hand-labels use the EXACT taxonomy strings (a vocabulary
    mismatch here silently tanks the agreement score)."""
    guide = (
        "VALIDATION LABELING GUIDE\n" + "=" * 40 + "\n\n"
        "In validation_sample.csv, fill:\n"
        "  human_attribute  = EXACTLY ONE of:\n    " + ", ".join(ATTRIBUTES) + "\n"
        "  human_sentiment  = EXACTLY ONE of:\n    positive, negative, neutral, mixed\n\n"
        "Copy the strings verbatim -- 'product_quality' or 'build quality' will\n"
        "score as a miss. Use the model's columns only as a reference, not a crutch.\n\n"
        "Attribute definitions:\n"
        "  zero_retention           holding/returning to zero, QD repeatability, zero shift\n"
        "  build_quality            durability, materials, machining, finish, how solid it feels\n"
        "  weight                   how heavy or light the mount is\n"
        "  price_value              cost, worth, expensive/cheap, value for money\n"
        "  fit_and_install          ring height, clearance, offset, torque, install, fitment\n"
        "  availability_and_service in stock / lead time / where to buy, CS, warranty, returns\n"
        "  other                    anything else, or too vague (questions, build lists)\n\n"
        "Sentiment: stance toward the MOUNT. Watch sarcasm ('only lost zero twice,\n"
        "great value' = negative). neutral = a plain fact/question; mixed = clear\n"
        "positives AND negatives together.\n")
    open(os.path.join(OUT_DIR, "label_guide.txt"), "w", encoding="utf-8").write(guide)
    log("[report] wrote outputs/label_guide.txt")


def _write_validation_sample(labeled):
    import pandas as pd
    vpath = os.path.join(OUT_DIR, "validation_sample.csv")
    # NEVER overwrite a sample that already exists -- it may hold your hand labels.
    # Delete the file yourself if you want a fresh, unlabeled sample.
    if os.path.exists(vpath):
        try:
            existing = pd.read_csv(vpath).fillna("")
            labeled_n = ((existing.get("human_attribute", "").astype(str).str.strip() != "") |
                         (existing.get("human_sentiment", "").astype(str).str.strip() != "")).sum()
        except Exception:
            labeled_n = 0
        log("[report] validation_sample.csv already exists (%d rows hand-labeled) "
            "-- leaving it untouched. Delete it to regenerate." % int(labeled_n))
        return
    if labeled.empty:
        open(vpath, "w", encoding="utf-8").write(
            "id,text,url,model_brands,model_attribute,model_sentiment,"
            "human_attribute,human_sentiment\n")
        log("[report] wrote empty validation_sample.csv (nothing classified yet)")
        return
    # stratify across brand x sentiment (brand = model's judgment)
    buckets = defaultdict(list)
    for _, r in labeled.iterrows():
        mb = _model_brands(r)
        b = mb[0] if mb else "none"
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
            "text": " ".join(str(r.get("window") or r["text"]).split())[:300],
            "url": r["url"],
            "model_brands": "|".join(_model_brands(r)),
            "model_attribute": r["attribute"],
            "model_sentiment": r["sentiment"],
            "human_attribute": "",
            "human_sentiment": "",
        })
    pd.DataFrame(rows, columns=["id", "text", "url", "model_brands", "model_attribute",
                                "model_sentiment", "human_attribute", "human_sentiment"]
                 ).to_csv(vpath, index=False)
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
_Pull from findings_candidates.md -- it ranks only cells with >= {min_support} docs._
1. TODO -- <lead finding; cite a robust heat-map cell and a verbatim>
2. TODO -- <second finding>
3. TODO -- <third finding>

## One limitation, named unprompted
TODO -- Labels come from a local model that agreed with my 50-comment hand audit
~66% of the time, so I read the DIRECTION and RANKING of cells, not absolute
values. The model marks most comments neutral (see sentiment_distribution.csv),
which compresses net-sentiment magnitudes -- brand-to-brand differences matter,
the decimals do not. The corpus is also self-selected forum/enthusiast voices;
pair with a returns/warranty dataset before drawing absolute conclusions.

## How I validated
I audited my own labels the way I'd audit a build: a stratified 50-row sample
(validation_sample.csv) hand-labeled and scored with agreement.py, reported
separately for attribute and sentiment. TODO -- <insert the two agreement numbers>

## Method notes
- Filter: a document survives only if it matches >=1 brand alias AND >=1 category token.
- Windowing: counting and classification use the brand-mention sentence +/-1, not the full comment.
- Cells: a comment is attributed to the brand(s) the MODEL judged it to be about,
  not every brand a build-list post happens to name.
- Support gate: cells with < {min_support} docs are excluded from findings_candidates.
- Terms: log-odds (informative Dirichlet prior) over 1-3 grams, single-brand docs,
  brand/product/scope names stripped.
- Sentiment: net = (positive - negative) / total per brand x attribute cell.
""".format(stats=stats, valid_rate=valid_rate or "TODO -- run classify",
           min_support=MIN_CELL_SUPPORT)
    open(os.path.join(OUT_DIR, "memo.md"), "w", encoding="utf-8").write(memo)
    log("[report] wrote outputs/memo.md")


# ============================================================================
#  STAGE 7  AUDIT -- faithful port of fix_outputs.py (audit-consistent artifacts)
# ============================================================================
# Expanded alias set for the audit's single-brand quote check. Substring match on
# a space-padded lowercased window (so "nf ", "c1 ", "aus " match as tokens).
AUDIT_ALIAS = {
    "nightforce": ["nightforce", "night force", "unimount", "uni-mount", "magmount",
                   "mag mount", "x-treme duty", "xtreme duty", "ultralite", "ultramount", "nf "],
    "leupold": ["leupold", "leopold", "luepold", "lupold", "backcountry", "mark ar"],
    "reptilia": ["reptilia", "reptillia", "reptila", "aus "],
    "badger_ordnance": ["badger", "condition one", "c1 "],
    "spuhr": ["spuhr", "sphur"],
    "geissele": ["geissele", "geiselle", "giselle", "gieselle", "super precision"],
    "warne": ["warne", "mountain tech", "skyline"],
}


def stage_audit():
    """Reproduce the fix_outputs.py reference artifacts from corpus.csv:
    verbatims_clean.md, heat_{net_sentiment,mention_share}_gated.png,
    stats_update.txt, nf_other_sample.csv, recency_summary.txt, and the 24-month
    matrices. Self-contained: reads the full mounts corpus and applies its own
    recency cutoff (SINCE if given, else 2024-07-24)."""
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brands = list(BRANDS.keys())
    attrs = list(ATTRIBUTES)
    gate = MIN_CELL_SUPPORT
    cutoff = SINCE or "2024-07-24"

    if not os.path.exists(CORPUS_CSV):
        stage_corpus()
    _merge_labels_into_corpus()
    df = pd.read_csv(CORPUS_CSV)
    df = df[df["bucket"] == "mounts"].copy()
    df["model_brands"] = df["model_brands"].fillna("")
    df["window"] = df["window"].fillna(df["text"]).astype(str)
    df["blist"] = df["model_brands"].apply(lambda s: [b for b in str(s).split("|") if b in brands])
    df["n_brands"] = df["blist"].apply(len)

    # ---- 1. commit rates ----
    doc_total = len(df)
    doc_committed = int((df["attribute"] != "other").sum())
    attrib = df.explode("blist").dropna(subset=["blist"])
    att_total = len(attrib)
    att_committed = int((attrib["attribute"] != "other").sum())
    per_brand_other = (attrib.groupby("blist")["attribute"]
                       .apply(lambda s: (s == "other").mean()).round(3))

    # ---- matrices (attribution-level over model brands) ----
    def matrices(frame):
        a = frame.explode("blist").dropna(subset=["blist"])
        cnt = (a.pivot_table(index="blist", columns="attribute", values="id", aggfunc="count")
               .reindex(index=brands, columns=attrs).fillna(0).astype(int))
        grp = a.groupby(["blist", "attribute"])["sentiment"]
        pos = grp.apply(lambda s: (s == "positive").sum())
        neg = grp.apply(lambda s: (s == "negative").sum())
        n = a.groupby(["blist", "attribute"]).size()
        ns = ((pos - neg) / n).round(2).unstack().reindex(index=brands, columns=attrs)
        return cnt, ns

    cnt_all, ns_all = matrices(df)

    # ---- 2. Nightforce 'other' sample ----
    nf_other = attrib[(attrib["blist"] == "nightforce") & (attrib["attribute"] == "other")]
    (nf_other.sample(min(20, len(nf_other)), random_state=7)[["window", "url", "sentiment"]]
     .to_csv(os.path.join(OUT_DIR, "nf_other_sample.csv"), index=False))

    # ---- 3. clean verbatims (single-brand docs, alias-confirmed) ----
    def has_alias(text, brand):
        t = " " + str(text).lower() + " "
        return any(a in t for a in AUDIT_ALIAS[brand])
    singles = df[df["n_brands"] == 1].copy()
    singles["brand"] = singles["blist"].str[0]
    singles = singles[singles.apply(lambda r: has_alias(r["window"], r["brand"]), axis=1)]
    lines = ["# Verbatims (single-brand docs only) — HAND-VERIFY EVERY QUOTE AT ITS LINK BEFORE SHARING\n"]
    for b in brands:
        for at in attrs[:-1]:   # skip "other"
            n = int(cnt_all.loc[b, at])
            if n < gate:
                continue
            cell = singles[(singles["brand"] == b) & (singles["attribute"] == at)].copy()
            if cell.empty:
                continue
            cell["pref"] = (cell["quotable"] == True).astype(int) * 2 + (cell["sentiment"] != "neutral").astype(int)
            cell["lenfit"] = -abs(cell["window"].str.len() - 180)
            cell = cell.sort_values(["pref", "lenfit"], ascending=False).drop_duplicates("url").head(5)
            ns_v = ns_all.loc[b, at]
            lines.append("\n## %s / %s   (net %+.2f, n=%d)" % (b, at, ns_v, n))
            for _, r in cell.iterrows():
                q = re.sub(r"\s+", " ", str(r["window"]))[:280]
                lines.append('- "%s"\n  <%s>  (%s, %s)' % (q, r["url"], r["source"], r["sentiment"]))
    open(os.path.join(OUT_DIR, "verbatims_clean.md"), "w", encoding="utf-8").write("\n".join(lines))
    log("[audit] wrote outputs/verbatims_clean.md")

    # ---- 4. gated heat maps ----
    def heat(cnt, ns, mode, fname, title):
        vals = ns if mode == "net" else cnt.div(cnt.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        data = vals.values.astype(float)
        mask = cnt.values < gate
        show = np.ma.masked_where(mask, data)
        cmap = (plt.cm.RdYlGn if mode == "net" else plt.cm.Blues).copy()
        cmap.set_bad("#d9d9d9")
        vmin, vmax = (-1, 1) if mode == "net" else (0, np.nanmax(np.where(mask, np.nan, data)))
        im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(attrs))); ax.set_xticklabels(attrs, rotation=40, ha="right")
        ax.set_yticks(range(len(brands))); ax.set_yticklabels(brands)
        for i in range(len(brands)):
            for j in range(len(attrs)):
                n = int(cnt.values[i, j])
                if mask[i, j]:
                    ax.text(j, i, "n<%d" % gate, ha="center", va="center", fontsize=7, color="#777777")
                else:
                    v = data[i, j]
                    s = ("%+.2f\n(n=%d)" % (v, n)) if mode == "net" else ("%.0f%%\n(n=%d)" % (v * 100, n))
                    ax.text(j, i, s, ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.close(fig)
        log("[audit] wrote outputs/%s" % fname)

    heat(cnt_all, ns_all, "net", "heat_net_sentiment_gated.png",
         "Net sentiment (pos-neg)/total — cells under n=%d masked" % gate)
    heat(cnt_all, ns_all, "share", "heat_mention_share_gated.png",
         "Share of each brand's mentions by attribute — cells under n=%d masked" % gate)

    # ---- 5. recency check ----
    rec = df[df["created_utc"].astype(str) >= cutoff]
    cnt_r, ns_r = matrices(rec)
    cnt_r.to_csv(os.path.join(OUT_DIR, "mention_counts_24mo.csv"))
    ns_r.round(2).to_csv(os.path.join(OUT_DIR, "net_sentiment_24mo.csv"))

    def realattr_totals(cnt):
        return cnt.drop(columns=["other"]).sum(axis=0).sort_values(ascending=False)
    tot_all, tot_r = realattr_totals(cnt_all), realattr_totals(cnt_r)
    f1 = tot_r.index[0] == "fit_and_install" if len(tot_r) else None
    if (len(cnt_r) and cnt_r.loc["warne", "price_value"] >= gate
            and cnt_r.loc["nightforce", "price_value"] >= gate):
        f2 = (ns_r.loc["warne", "price_value"] > 0.25) and (
            ns_r.loc["warne", "price_value"] > ns_r.loc["nightforce", "price_value"])
    else:
        f2 = None
    bq = []
    for b in brands:
        row = ns_r.loc[b][[a for a in attrs[:-1] if cnt_r.loc[b, a] >= gate]] if len(cnt_r) else []
        if len(row):
            bq.append(row.idxmax() == "build_quality")
    f3 = (sum(bq) >= max(1, len(bq) // 2 + 1)) if bq else None
    verdict = {True: "HOLDS", False: "FLIPS", None: "UNDER-SUPPORTED"}
    with open(os.path.join(OUT_DIR, "recency_summary.txt"), "w", encoding="utf-8") as f:
        f.write("RECENCY CHECK — docs from %s on\n" % cutoff)
        f.write("24mo corpus: %d of %d docs (%.0f%%)\n\n"
                % (len(rec), doc_total, 100.0 * len(rec) / doc_total if doc_total else 0))
        f.write("Finding 1 (fitment is biggest real attribute): "
                + ("HOLDS" if f1 else "FLIPS") + "  | 24mo totals: %s\n" % tot_r.to_dict())
        f.write("Finding 2 (Warne value anchor > NF on price): " + verdict[f2]
                + " | warne %s (n=%d), nf %s (n=%d)\n"
                % (ns_r.loc["warne", "price_value"] if len(cnt_r) else float("nan"),
                   int(cnt_r.loc["warne", "price_value"]) if len(cnt_r) else 0,
                   ns_r.loc["nightforce", "price_value"] if len(cnt_r) else float("nan"),
                   int(cnt_r.loc["nightforce", "price_value"]) if len(cnt_r) else 0))
        f.write("Finding 3 (build quality most-positive per brand): " + verdict[f3]
                + " | brands where true: %d/%d\n" % (sum(bq), len(bq)))
    log("[audit] wrote outputs/recency_summary.txt (+ 24mo matrices)")

    # ---- 6. stats for memo ----
    with open(os.path.join(OUT_DIR, "stats_update.txt"), "w", encoding="utf-8") as f:
        f.write("doc-level commit rate      : %d/%d = %.0f%% (other = %.0f%%)\n"
                % (doc_committed, doc_total, 100.0 * doc_committed / doc_total if doc_total else 0,
                   100.0 * (1 - doc_committed / doc_total) if doc_total else 0))
        f.write("attribution-level commit   : %d/%d = %.0f%%\n"
                % (att_committed, att_total, 100.0 * att_committed / att_total if att_total else 0))
        f.write("per-brand 'other' share    : " + json.dumps(per_brand_other.to_dict()) + "\n")
        f.write("24mo subset size           : %d docs (%.0f%% of corpus)\n"
                % (len(rec), 100.0 * len(rec) / doc_total if doc_total else 0))
    log("[audit] wrote outputs/stats_update.txt")


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
    if stage in ("audit", "all"):
        if not check_dependencies(need_plotting=True):
            return
        stage_audit()
    if stage == "all":
        print_surprises()


def main():
    global SINCE
    stage = STAGE
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--since":
            i += 1
            if i < len(args):
                SINCE = args[i].strip()
        elif a.startswith("--since="):
            SINCE = a.split("=", 1)[1].strip()
        elif not a.startswith("-"):
            stage = a.strip().lower()
        i += 1
    if SINCE and not re.match(r"^\d{4}-\d{2}-\d{2}$", SINCE):
        log("--since must be YYYY-MM-DD; got %r" % SINCE)
        return
    valid = {"collect", "corpus", "clouds", "classify", "analyze", "report", "audit", "all"}
    if stage not in valid:
        log("Unknown stage %r. Choose one of: %s" % (stage, ", ".join(sorted(valid))))
        return
    log("=== Nightforce Market Voice :: stage = %s%s ==="
        % (stage, (" since=%s" % SINCE) if SINCE else ""))
    run_stage(stage)
    log("=== done (stage = %s) ===" % stage)


if __name__ == "__main__":
    main()
