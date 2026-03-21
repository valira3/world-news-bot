#!/usr/bin/env python3
"""
World News Summary Bot — Telegram bot that aggregates world news,
generates AI-powered summaries, creates visual reports, and supports Q&A.

Single-file architecture for Railway deployment.
Python 3.11 compatible — NO backslashes inside f-string braces.
"""

import os
import sys
import json
import hashlib
import logging
import asyncio
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

import aiohttp
import feedparser
from aiohttp import web
from PIL import Image, ImageDraw, ImageFont
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from anthropic import AsyncAnthropic
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

# ─── Configuration ───────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "")
PORT = int(os.environ.get("PORT", "8080"))

# IST is UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("world-news-bot")

# ─── RSS Feeds ───────────────────────────────────────────────────────────────

RSS_FEEDS = {
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "AP News": "https://rsshub.app/apnews/topics/apf-topnews",
    "CNN": "http://rss.cnn.com/rss/edition_world.rss",
    "France24": "https://www.france24.com/en/rss",
    "DW News": "https://rss.dw.com/rdf/rss-en-world",
    "ABC News": "https://abcnews.go.com/abcnews/internationalheadlines",
    # Chicago local news
    "Chicago Sun-Times": "https://chicago.suntimes.com/rss/index.xml",
    "Block Club Chicago": "https://blockclubchicago.org/feed/",
    "WTTW Chicago": "https://news.wttw.com/rss.xml",
    "Chicago Reader": "https://chicagoreader.com/feed/",
}

# ─── Categories ──────────────────────────────────────────────────────────────

CATEGORIES = {
    "geopolitics": ["war", "diplomacy", "sanctions", "treaty", "nato", "un", "summit"],
    "economy": ["gdp", "inflation", "fed", "interest rate", "trade", "tariff", "recession"],
    "technology": ["ai", "tech", "startup", "cyber", "software", "chip", "semiconductor"],
    "climate": ["climate", "carbon", "weather", "hurricane", "flood", "wildfire", "emissions"],
    "conflict": ["military", "attack", "missile", "troops", "invasion", "ceasefire"],
    "science": ["study", "research", "space", "nasa", "discovery", "medical", "vaccine"],
    "markets": ["stock", "nasdaq", "dow", "crypto", "bitcoin", "oil", "gold"],
    "local": ["chicago", "illinois", "cook county", "mayor", "cta", "cubs", "bears", "bulls", "white sox", "blackhawks", "o'hare", "midway", "loop", "south side", "north side", "wrigley"],
}

CATEGORY_COLORS = {
    "geopolitics": "#3b82f6",
    "economy": "#22c55e",
    "technology": "#a855f7",
    "climate": "#06b6d4",
    "conflict": "#ef4444",
    "science": "#f59e0b",
    "markets": "#ec4899",
    "local": "#f97316",
    "general": "#6b7280",
}

CATEGORY_EMOJI = {
    "geopolitics": "\U0001f30d",
    "economy": "\U0001f4b0",
    "technology": "\U0001f4bb",
    "climate": "\U0001f30a",
    "conflict": "\u2694\ufe0f",
    "science": "\U0001f52c",
    "markets": "\U0001f4c8",
    "local": "\U0001f3d9\ufe0f",
    "general": "\U0001f4f0",
}

# ─── File Paths ──────────────────────────────────────────────────────────────

def get_path(filename):
    return os.path.join(DATA_DIR, filename)


NEWS_CACHE_FILE = "news_cache.jsonl"
USER_PREFS_FILE = "user_prefs.json"
SENT_ARTICLES_FILE = "sent_articles.json"
CONVERSATION_FILE = "conversation_history.json"
BOT_STATE_FILE = "bot_state.json"
ALERTS_SENT_FILE = "alerts_sent.json"
FOLLOW_ALERTS_FILE = "follow_alerts_sent.json"

# ─── Data Helpers ────────────────────────────────────────────────────────────

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_json(filename, default=None):
    path = get_path(filename)
    if default is None:
        default = {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(filename, data):
    ensure_data_dir()
    path = get_path(filename)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def load_news_cache():
    """Load all articles from the JSONL cache."""
    path = get_path(NEWS_CACHE_FILE)
    articles = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return articles


def append_news_cache(article):
    """Append a single article to the JSONL cache."""
    ensure_data_dir()
    path = get_path(NEWS_CACHE_FILE)
    with open(path, "a") as f:
        f.write(json.dumps(article, default=str) + "\n")


def trim_news_cache(days=14):
    """Remove articles older than N days from the cache."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    articles = load_news_cache()
    kept = []
    for a in articles:
        try:
            fetched = datetime.fromisoformat(a.get("fetched_at", "2000-01-01"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if fetched > cutoff:
                kept.append(a)
        except (ValueError, TypeError):
            kept.append(a)
    ensure_data_dir()
    path = get_path(NEWS_CACHE_FILE)
    with open(path, "w") as f:
        for a in kept:
            f.write(json.dumps(a, default=str) + "\n")
    logger.info("Cache trimmed: %d -> %d articles", len(articles), len(kept))


def hash_url(url):
    return hashlib.sha256(url.encode()).hexdigest()


def is_duplicate_title(title, existing_titles, threshold=0.85):
    """Check if title is too similar to any existing title."""
    for existing in existing_titles:
        ratio = SequenceMatcher(None, title.lower(), existing.lower()).ratio()
        if ratio > threshold:
            return True
    return False


def get_user_prefs(user_id):
    prefs = load_json(USER_PREFS_FILE, {})
    uid = str(user_id)
    if uid not in prefs:
        prefs[uid] = {
            "categories": list(CATEGORIES.keys()),
            "schedule": "07:00",
            "summary_length": "medium",
        }
        save_json(USER_PREFS_FILE, prefs)
    return prefs[uid]


def set_user_prefs(user_id, key, value):
    prefs = load_json(USER_PREFS_FILE, {})
    uid = str(user_id)
    if uid not in prefs:
        prefs[uid] = {
            "categories": list(CATEGORIES.keys()),
            "schedule": "07:00",
            "summary_length": "medium",
        }
    prefs[uid][key] = value
    save_json(USER_PREFS_FILE, prefs)


def get_conversation_history(user_id, limit=5):
    history = load_json(CONVERSATION_FILE, {})
    uid = str(user_id)
    return history.get(uid, [])[-limit:]


def add_conversation_exchange(user_id, question, answer):
    history = load_json(CONVERSATION_FILE, {})
    uid = str(user_id)
    if uid not in history:
        history[uid] = []
    history[uid].append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    history[uid] = history[uid][-10:]
    save_json(CONVERSATION_FILE, history)


def mark_sent(article_id):
    sent = load_json(SENT_ARTICLES_FILE, {})
    sent[article_id] = datetime.now(timezone.utc).isoformat()
    # Trim to 7 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    trimmed = {}
    for aid, ts in sent.items():
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > cutoff:
                trimmed[aid] = ts
        except (ValueError, TypeError):
            trimmed[aid] = ts
    save_json(SENT_ARTICLES_FILE, trimmed)


def is_sent(article_id):
    sent = load_json(SENT_ARTICLES_FILE, {})
    return article_id in sent


def mark_alert_sent(article_id):
    """Mark an article as sent via breaking news auto-alert."""
    alerts = load_json(ALERTS_SENT_FILE, {})
    alerts[article_id] = datetime.now(timezone.utc).isoformat()
    # Trim to 7 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    trimmed = {}
    for aid, ts in alerts.items():
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > cutoff:
                trimmed[aid] = ts
        except (ValueError, TypeError):
            trimmed[aid] = ts
    save_json(ALERTS_SENT_FILE, trimmed)


def is_alert_sent(article_id):
    """Check if an article has already been sent as a breaking alert."""
    alerts = load_json(ALERTS_SENT_FILE, {})
    return article_id in alerts


def mark_follow_alert_sent(user_id, article_id, keyword):
    """Mark a follow-alert as sent for a user+article+keyword combo."""
    alerts = load_json(FOLLOW_ALERTS_FILE, {})
    key = str(user_id) + "_" + article_id + "_" + keyword
    alerts[key] = datetime.now(timezone.utc).isoformat()
    # Trim to 7 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    trimmed = {}
    for k, ts in alerts.items():
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > cutoff:
                trimmed[k] = ts
        except (ValueError, TypeError):
            trimmed[k] = ts
    save_json(FOLLOW_ALERTS_FILE, trimmed)


def is_follow_alert_sent(user_id, article_id, keyword):
    """Check if a follow-alert was already sent."""
    alerts = load_json(FOLLOW_ALERTS_FILE, {})
    key = str(user_id) + "_" + article_id + "_" + keyword
    return key in alerts


# ─── Story Clustering ───────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "nor", "so",
    "yet", "both", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "all", "also", "am", "as", "because", "before", "between", "during",
    "he", "her", "here", "him", "his", "how", "i", "if", "into", "it",
    "its", "me", "my", "new", "now", "only", "our", "out", "over", "own",
    "s", "she", "that", "their", "them", "then", "there", "these", "they",
    "this", "up", "us", "we", "what", "when", "which", "who", "whom",
    "why", "you", "your", "says", "said", "say", "get", "got", "one",
    "two", "three", "after", "being", "going", "make", "many", "much",
    # Common news words that cause false-positive keyword matches
    "news", "report", "update", "latest", "breaking", "official", "officials",
    "government", "state", "country", "world", "global", "international",
    "national", "local", "today", "year", "years", "people", "back", "first",
    "time", "with", "from", "were", "have", "has", "had", "will", "would",
    "could", "also", "just", "more", "most", "some", "other", "into", "about",
    "than", "been", "not", "but", "can", "all",
}


def extract_keywords(title):
    """Extract significant keyword stems from a title, stripping stopwords.

    Applies naive stemming (strip trailing s/ed/ing) to improve matching
    across sources that use different word forms.
    """
    words = re.findall(r'[a-z]+', title.lower())
    stems = set()
    for w in words:
        if w in STOPWORDS or len(w) <= 2:
            continue
        # Naive suffix stripping for better cross-source matching
        stem = w
        if stem.endswith("ing") and len(stem) > 5:
            stem = stem[:-3]
        elif stem.endswith("ed") and len(stem) > 4:
            stem = stem[:-2]
        elif stem.endswith("s") and len(stem) > 3:
            stem = stem[:-1]
        stems.add(stem)
    return stems


def _articles_match(title_a, title_b, kw_a, kw_b):
    """Check if two articles are about the same event."""
    sim = SequenceMatcher(None, title_a.lower(), title_b.lower()).ratio()
    if sim > 0.5:
        return True
    if kw_a and kw_b:
        overlap = len(kw_a & kw_b)
        union = len(kw_a | kw_b)
        kw_ratio = overlap / union if union > 0 else 0
        if kw_ratio > 0.5:
            return True
        # Combined: moderate title similarity + some keyword overlap
        if sim > 0.35 and overlap >= 2:
            return True
        # Strong keyword overlap (3+ shared stems is a reliable signal)
        if overlap >= 3:
            return True
    return False


async def ai_dedup_articles(new_articles, recent_cache):
    """Use Claude Haiku to identify which new articles are duplicates of cached ones.

    Returns list of new articles that are NOT duplicates.
    """
    if not new_articles or not recent_cache:
        return new_articles

    # Build a numbered reference list of recent cache titles+sources
    cache_ref = []
    for i, a in enumerate(recent_cache[:50]):  # Cap at 50 recent
        cache_ref.append(str(i + 1) + ". [" + a.get("source", "") + "] " + a.get("title", ""))
    cache_text = "\n".join(cache_ref)

    # Check new articles in batches of 10
    unique_articles = []
    for batch_start in range(0, len(new_articles), 10):
        batch = new_articles[batch_start:batch_start + 10]
        new_ref = []
        for i, a in enumerate(batch):
            letter = chr(65 + i)
            new_ref.append(letter + ". [" + a.get("source", "") + "] " + a.get("title", ""))
        new_text = "\n".join(new_ref)

        prompt = (
            "You are a news deduplication system. Compare these NEW articles against the EXISTING articles.\n"
            "Two articles are DUPLICATES if they cover the SAME specific event or development, even if worded differently.\n"
            "Two articles are NOT duplicates if they cover different aspects/events within the same broad topic.\n\n"
            "EXISTING articles:\n" + cache_text + "\n\n"
            "NEW articles:\n" + new_text + "\n\n"
            "For each new article (A, B, C...), respond with ONLY:\n"
            "- 'X: UNIQUE' if it covers a new event not in the existing list\n"
            "- 'X: DUP of Y' if it duplicates existing article number Y\n"
            "One line per article, nothing else."
        )

        try:
            await rate_limited_api_call()
            client = get_anthropic_client()
            response = await client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = response.content[0].text.strip()

            # Parse response
            for i, a in enumerate(batch):
                letter = chr(65 + i)
                # Check if this article was marked as UNIQUE
                found = False
                for line in result_text.split("\n"):
                    line = line.strip()
                    if line.startswith(letter + ":") or line.startswith(letter + " :"):
                        found = True
                        if "UNIQUE" in line.upper():
                            unique_articles.append(a)
                        else:
                            logger.info("AI dedup: removed '%s' as duplicate", a.get("title", "")[:60])
                        break
                if not found:
                    # If not mentioned in response, assume unique (safe default)
                    unique_articles.append(a)
        except Exception as e:
            logger.warning("AI dedup failed: %s — keeping all articles", e)
            unique_articles.extend(batch)

    logger.info("AI dedup against cache: %d -> %d articles", len(new_articles), len(unique_articles))
    return unique_articles


async def ai_dedup_within_batch(articles):
    """Use Claude Haiku to deduplicate articles within the same batch.

    When multiple feeds report the same event, keeps only the best version.
    Returns deduplicated list.
    """
    if len(articles) <= 1:
        return articles

    # Build numbered list
    art_ref = []
    for i, a in enumerate(articles):
        art_ref.append(str(i + 1) + ". [" + a.get("source", "") + "] " + a.get("title", ""))
    art_text = "\n".join(art_ref)

    prompt = (
        "You are a news deduplication system. These articles were all fetched at the same time.\n"
        "Identify groups of articles that cover the SAME specific event or development.\n"
        "Two articles are DUPLICATES if they report on the same event, even if worded differently.\n\n"
        "Articles:\n" + art_text + "\n\n"
        "For each article, respond with ONLY:\n"
        "- 'N: UNIQUE' if it is not a duplicate of any other article in this list\n"
        "- 'N: DUP of M' if it duplicates article M (keep the one with the more detailed title)\n"
        "One line per article, nothing else."
    )

    try:
        await rate_limited_api_call()
        client = get_anthropic_client()
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text.strip()

        # Parse: keep articles marked UNIQUE or not mentioned
        unique_articles = []
        for i, a in enumerate(articles):
            num = str(i + 1)
            found = False
            for line in result_text.split("\n"):
                line = line.strip()
                if line.startswith(num + ":") or line.startswith(num + " :"):
                    found = True
                    if "UNIQUE" in line.upper():
                        unique_articles.append(a)
                    else:
                        logger.info("AI batch dedup: removed '%s' as duplicate", a.get("title", "")[:60])
                    break
            if not found:
                unique_articles.append(a)

        logger.info("AI batch dedup: %d -> %d articles", len(articles), len(unique_articles))
        return unique_articles
    except Exception as e:
        logger.warning("AI batch dedup failed: %s — keeping all articles", e)
        return articles


def cluster_articles(articles):
    """Group articles about the same event using title similarity + keyword overlap.

    Checks candidate articles against ALL members of a cluster, not just the first.
    Returns list of clusters:
    {"primary": article, "related": [article, ...], "sources": ["BBC", ...]}
    """
    if not articles:
        return []

    assigned = [False] * len(articles)
    clusters_idx = []  # list of lists of indices

    # Precompute keywords and lowercase titles for each article
    keywords_list = [extract_keywords(a.get("title", "")) for a in articles]
    titles_lower = [a.get("title", "").lower() for a in articles]

    for i in range(len(articles)):
        if assigned[i]:
            continue
        cluster_indices = [i]
        assigned[i] = True

        for j in range(i + 1, len(articles)):
            if assigned[j]:
                continue

            # Check against ALL articles already in this cluster
            matched = False
            for idx in cluster_indices:
                if _articles_match(
                    titles_lower[idx], titles_lower[j],
                    keywords_list[idx], keywords_list[j],
                ):
                    matched = True
                    break

            if matched:
                cluster_indices.append(j)
                assigned[j] = True

        clusters_idx.append(cluster_indices)

    # Build cluster dicts
    clusters = []
    for indices in clusters_idx:
        cluster_arts = [articles[idx] for idx in indices]
        cluster_arts.sort(
            key=lambda x: x.get("newsworthiness", 5), reverse=True
        )
        primary = cluster_arts[0]
        related = cluster_arts[1:]
        sources = list(dict.fromkeys(
            a.get("source", "Unknown") for a in cluster_arts
        ))
        clusters.append({
            "primary": primary,
            "related": related,
            "sources": sources,
        })

    return clusters


async def consolidate_cluster(cluster):
    """Call Claude Haiku to produce a consolidated summary for a multi-source cluster.

    Returns dict: headline, key_points (list of 3), summary, sources (list of dicts).
    """
    articles = [cluster["primary"]] + cluster["related"]
    articles_text = ""
    for a in articles:
        articles_text += (
            "Source: " + a.get("source", "Unknown") + "\n"
            "Title: " + a.get("title", "") + "\n"
            "Summary: " + a.get("ai_summary", a.get("summary", "")) + "\n\n"
        )

    prompt_text = (
        "You are a world-class news editor. Given these articles from different "
        "sources about the same event, produce a consolidated summary.\n\n"
        + articles_text
        + "Respond in JSON with these exact keys:\n"
        '{"headline": "...", "key_points": ["What happened: ...", '
        '"Why it matters: ...", "What\'s next: ..."], '
        '"summary": "2-3 sentence summary incorporating all perspectives"}'
    )

    try:
        await rate_limited_api_call()
        client = get_anthropic_client()
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt_text}],
        )
        text = response.content[0].text.strip()
        # Try to extract JSON — may have nested braces in key_points
        # Find the outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            data = json.loads(json_str)
            source_list = []
            for a in articles:
                source_list.append({
                    "name": a.get("source", "Unknown"),
                    "url": a.get("url", ""),
                })
            return {
                "headline": data.get("headline", ""),
                "key_points": data.get("key_points", [])[:3],
                "summary": data.get("summary", ""),
                "sources": source_list,
            }
    except Exception as e:
        logger.error("Consolidation error: %s", e)
    return None


def format_consolidated_caption(consolidated, primary_article):
    """Format a consolidated cluster message as a Telegram caption.

    Returns (short_caption, full_text) — if short_caption is under 1024 chars
    it can be used as a photo caption. Otherwise use short_caption for photo
    and full_text as a follow-up text message.
    """
    headline = consolidated.get("headline", "")
    key_points = consolidated.get("key_points", [])
    summary = consolidated.get("summary", "")
    sources = consolidated.get("sources", [])
    num_sources = len(sources)

    safe_headline = _escape_md(headline)
    safe_summary = _escape_md(summary)

    lines = [
        "*" + safe_headline + "*",
        "",
        "\U0001f4ca Covered by " + str(num_sources) + " sources",
        "",
    ]
    for kp in key_points:
        lines.append("\u25b8 " + _escape_md(kp))
    lines.append("")
    lines.append(safe_summary)
    lines.append("")
    lines.append("\U0001f4f0 Sources:")
    for s_info in sources:
        name = _escape_md(s_info.get("name", "Unknown"))
        url = s_info.get("url", "")
        lines.append("\u2022 " + name + " \u2014 [Read](" + url + ")")

    full_text = "\n".join(lines)

    if len(full_text) <= 1024:
        return full_text, None

    # Build short caption for photo
    short_lines = [
        "*" + safe_headline + "*",
        "",
        "\U0001f4ca Covered by " + str(num_sources) + " sources",
        "",
        safe_summary[:200],
    ]
    short_caption = "\n".join(short_lines)
    if len(short_caption) > 1024:
        short_caption = short_caption[:1021] + "..."
    return short_caption, full_text


async def _send_consolidated_story(bot, chat_id, consolidated, primary_article):
    """Send a consolidated multi-source story: photo + caption, optional follow-up."""
    short_caption, full_text = format_consolidated_caption(
        consolidated, primary_article
    )
    image_url = primary_article.get("_og_image_url")

    if image_url:
        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=image_url,
                caption=short_caption,
                parse_mode="Markdown",
            )
        except Exception:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        image_url,
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={"User-Agent": "Mozilla/5.0"},
                    ) as resp:
                        if resp.status == 200 and "image" in resp.content_type:
                            data = await resp.read()
                            if len(data) >= 500:
                                await bot.send_photo(
                                    chat_id=chat_id,
                                    photo=data,
                                    caption=short_caption,
                                    parse_mode="Markdown",
                                )
                            else:
                                await bot.send_message(
                                    chat_id=chat_id,
                                    text=short_caption,
                                    parse_mode="Markdown",
                                    disable_web_page_preview=True,
                                )
            except Exception:
                await bot.send_message(
                    chat_id=chat_id,
                    text=short_caption,
                    parse_mode="Markdown",
                    disable_web_page_preview=True,
                )
    else:
        await bot.send_message(
            chat_id=chat_id,
            text=short_caption,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )

    # Send full text as follow-up if caption was too long
    if full_text:
        await asyncio.sleep(0.3)
        if len(full_text) > 4096:
            full_text = full_text[:4093] + "..."
        await bot.send_message(
            chat_id=chat_id,
            text=full_text,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )


# ─── Perspectives Analysis ──────────────────────────────────────────────

async def analyze_perspectives(topic, articles):
    """Call Claude Sonnet to analyze how different sources cover a topic."""
    articles_text = ""
    for a in articles:
        articles_text += (
            "Source: " + a.get("source", "Unknown") + "\n"
            "Title: " + a.get("title", "") + "\n"
            "Summary: " + a.get("ai_summary", a.get("summary", "")) + "\n\n"
        )

    prompt_text = (
        "Analyze how these news sources cover the topic '" + topic + "'. "
        "For each source, write 1-2 sentences about their angle, emphasis, "
        "and what they highlight or downplay. Then provide a brief overall "
        "assessment of the coverage spectrum.\n\n"
        + articles_text
        + "Format your response exactly like this:\n"
        "SOURCE_NAME: analysis of their angle\n"
        "---\n"
        "OVERALL: brief assessment of the coverage spectrum with a recommendation"
    )

    try:
        await rate_limited_api_call()
        client = get_anthropic_client()
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error("Perspectives analysis error: %s", e)
        return None


# ─── Claude API ──────────────────────────────────────────────────────────────

anthropic_client = None
_api_call_times = []
API_RATE_LIMIT = 30  # max calls per minute


def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return anthropic_client


async def rate_limited_api_call():
    """Simple rate limiter: max 30 calls per minute."""
    global _api_call_times
    now = time.time()
    _api_call_times = [t for t in _api_call_times if now - t < 60]
    if len(_api_call_times) >= API_RATE_LIMIT:
        wait = 60 - (now - _api_call_times[0])
        if wait > 0:
            logger.info("Rate limit: waiting %.1fs", wait)
            await asyncio.sleep(wait)
    _api_call_times.append(time.time())


async def summarize_article(title, content, source):
    """Use Claude Haiku to summarize an article."""
    try:
        await rate_limited_api_call()
        client = get_anthropic_client()
        prompt_text = (
            "You are a world-class news analyst. Summarize this article in 2-3 concise sentences.\n"
            "Then provide a one-line \"So What?\" takeaway explaining why this matters to a global audience.\n"
            "Rate the newsworthiness from 1-10 (10 = most significant global event).\n"
            "Categorize into one of: geopolitics, economy, technology, climate, conflict, science, markets.\n\n"
            "Article title: " + title + "\n"
            "Article content: " + (content[:2000] if content else title) + "\n"
            "Source: " + source + "\n\n"
            "Respond in JSON:\n"
            '{"summary": "...", "takeaway": "...", "newsworthiness": 8, "category": "geopolitics"}'
        )
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt_text}],
        )
        text = response.content[0].text.strip()
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "summary": data.get("summary", ""),
                "takeaway": data.get("takeaway", ""),
                "newsworthiness": float(data.get("newsworthiness", 5)),
                "category": data.get("category", "general"),
            }
    except Exception as e:
        logger.error("Summarization error: %s", e)
    return None


async def deep_analysis(article, question=None):
    """Use Claude Sonnet for deep analysis."""
    try:
        await rate_limited_api_call()
        client = get_anthropic_client()
        if question:
            prompt_text = (
                "Provide a deep analysis of this news story. "
                "Cover: background context, key players, implications, "
                "what to watch next, and how it connects to broader trends.\n\n"
                "Article: " + article.get("title", "") + "\n"
                "Summary: " + article.get("ai_summary", article.get("summary", "")) + "\n"
                "Source: " + article.get("source", "") + "\n"
                "Category: " + article.get("category", "") + "\n\n"
                "Additional question: " + question
            )
        else:
            prompt_text = (
                "Provide a deep analysis of this news story. "
                "Cover: background context, key players, implications, "
                "what to watch next, and how it connects to broader trends.\n\n"
                "Article: " + article.get("title", "") + "\n"
                "Summary: " + article.get("ai_summary", article.get("summary", "")) + "\n"
                "Source: " + article.get("source", "") + "\n"
                "Category: " + article.get("category", "")
            )
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error("Deep analysis error: %s", e)
        return "Unable to generate deep analysis at this time."


async def ask_claude(question, articles_context, history):
    """Use Claude Sonnet for conversational Q&A."""
    try:
        await rate_limited_api_call()
        client = get_anthropic_client()
        history_text = ""
        for ex in history:
            history_text += "User: " + ex["question"] + "\n"
            history_text += "Assistant: " + ex["answer"] + "\n\n"

        prompt_text = (
            "You are a knowledgeable news analyst. Answer the user's question "
            "based ONLY on the provided news articles.\n"
            "Cite specific sources and dates. If the articles don't contain "
            "enough information, say so.\n"
            "Be conversational but authoritative. Keep answers concise "
            "(3-5 sentences for simple questions, more for complex).\n\n"
            "Previous conversation:\n" + history_text + "\n"
            "Available news articles:\n" + articles_context + "\n\n"
            "User question: " + question
        )
        response = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error("Ask error: %s", e)
        return "I'm having trouble answering right now. Please try again later."


# ─── News Fetching ───────────────────────────────────────────────────────────

async def fetch_rss_feed(session, source, url):
    """Fetch and parse a single RSS feed."""
    articles = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                text = await resp.text()
                feed = feedparser.parse(text)
                for entry in feed.entries[:15]:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "").strip()
                    if not title or not link:
                        continue
                    # Extract content
                    content = ""
                    if hasattr(entry, "summary"):
                        content = entry.summary
                    elif hasattr(entry, "description"):
                        content = entry.description
                    # Clean HTML tags
                    content = re.sub(r'<[^>]+>', '', content)

                    # Extract published date
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            import calendar
                            ts = calendar.timegm(entry.published_parsed)
                            published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                        except Exception:
                            pass
                    if not published:
                        published = datetime.now(timezone.utc).isoformat()

                    # Extract image
                    image_url = None
                    if hasattr(entry, "media_content"):
                        for media in entry.media_content:
                            if "url" in media:
                                image_url = media["url"]
                                break
                    if not image_url and hasattr(entry, "media_thumbnail"):
                        for thumb in entry.media_thumbnail:
                            if "url" in thumb:
                                image_url = thumb["url"]
                                break

                    articles.append({
                        "title": title,
                        "url": link,
                        "source": source,
                        "published": published,
                        "content": content[:3000],
                        "image_url": image_url,
                    })
    except Exception as e:
        logger.warning("RSS fetch failed for %s: %s", source, e)
    return articles


async def fetch_gnews(session):
    """Fetch from GNews API if key is available."""
    if not GNEWS_API_KEY:
        return []
    articles = []
    url = (
        "https://gnews.io/api/v4/top-headlines?"
        "category=general&lang=en&max=10&apikey=" + GNEWS_API_KEY
    )
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
                for item in data.get("articles", []):
                    title = item.get("title", "").strip()
                    link = item.get("url", "").strip()
                    if not title or not link:
                        continue
                    articles.append({
                        "title": title,
                        "url": link,
                        "source": "GNews: " + item.get("source", {}).get("name", "Unknown"),
                        "published": item.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                        "content": item.get("description", "")[:3000],
                        "image_url": item.get("image"),
                    })
    except Exception as e:
        logger.warning("GNews fetch failed: %s", e)
    return articles


async def fetch_all_news():
    """Fetch news from all sources, deduplicate, and summarize."""
    logger.info("Starting news fetch cycle...")
    all_raw = []
    async with aiohttp.ClientSession() as session:
        # Fetch RSS feeds concurrently
        tasks = []
        for source, url in RSS_FEEDS.items():
            tasks.append(fetch_rss_feed(session, source, url))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_raw.extend(result)

        # Fetch GNews
        gnews = await fetch_gnews(session)
        all_raw.extend(gnews)

    logger.info("Fetched %d raw articles", len(all_raw))

    # Deduplicate
    existing = load_news_cache()
    existing_ids = {a.get("id") for a in existing}
    existing_titles = [a.get("title", "") for a in existing]

    # Build a set of recent articles (last 24h) for semantic dedup
    now_utc = datetime.now(timezone.utc)
    recent_articles = []
    for a in existing:
        try:
            fetched = datetime.fromisoformat(a.get("fetched_at", "2000-01-01").replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if (now_utc - fetched).total_seconds() <= 86400:  # 24 hours
                recent_articles.append(a)
        except (ValueError, TypeError):
            pass

    new_articles = []
    for raw in all_raw:
        aid = hash_url(raw["url"])
        if aid in existing_ids:
            continue
        if is_duplicate_title(raw["title"], existing_titles):
            continue
        # Level 2: keyword+similarity match against recent cached articles (last 6h)
        raw_kw = extract_keywords(raw["title"])
        is_dup = False
        for recent in recent_articles:
            if _articles_match(raw["title"], recent.get("title", ""), raw_kw, extract_keywords(recent.get("title", ""))):
                is_dup = True
                break
        if is_dup:
            continue
        # Level 2b: check against other new articles collected so far in this batch
        for already in new_articles:
            if _articles_match(raw["title"], already.get("title", ""), raw_kw, extract_keywords(already.get("title", ""))):
                is_dup = True
                break
        if is_dup:
            continue
        existing_ids.add(aid)
        existing_titles.append(raw["title"])
        new_articles.append(raw)

    logger.info("New unique articles after string dedup: %d", len(new_articles))

    # AI-powered dedup: check new articles against recent cache
    if new_articles and recent_articles:
        new_articles = await ai_dedup_articles(new_articles, recent_articles)

    # AI-powered dedup: check new articles against each other within this batch
    if len(new_articles) > 1:
        new_articles = await ai_dedup_within_batch(new_articles)

    logger.info("New unique articles after AI dedup: %d", len(new_articles))

    # Summarize with AI (batch, with rate limiting)
    summarized = 0
    for raw in new_articles[:30]:  # Cap at 30 per cycle
        aid = hash_url(raw["url"])
        result = await summarize_article(raw["title"], raw.get("content", ""), raw["source"])
        article = {
            "id": aid,
            "title": raw["title"],
            "url": raw["url"],
            "source": raw["source"],
            "published": raw.get("published", datetime.now(timezone.utc).isoformat()),
            "summary": raw.get("content", "")[:500],
            "category": "general",
            "newsworthiness": 5.0,
            "image_url": raw.get("image_url"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ai_summary": "",
            "ai_takeaway": "",
        }
        if result:
            article["ai_summary"] = result["summary"]
            article["ai_takeaway"] = result["takeaway"]
            article["newsworthiness"] = result["newsworthiness"]
            article["category"] = result["category"]
            summarized += 1
        else:
            # Fallback: keyword-based categorization
            article["category"] = categorize_by_keywords(raw["title"] + " " + raw.get("content", ""))

        append_news_cache(article)

    logger.info("Summarized %d articles", summarized)
    return len(new_articles)


def categorize_by_keywords(text):
    """Fallback keyword-based categorization."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[cat] = score
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return "general"


# ─── Image Fetching ──────────────────────────────────────────────────────────

async def fetch_og_image(session, url):
    """Extract og:image from an article page."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8),
                               headers={"User-Agent": "Mozilla/5.0"}) as resp:
            if resp.status == 200:
                html = await resp.text(errors="ignore")
                html = html[:100000]
                match = re.search(
                    r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']'
                    r'(https?://[^"\']+)["\']', html
                )
                if match:
                    return match.group(1)
                match = re.search(
                    r'<meta[^>]*content=["\'](https?://[^"\']+)["\'][^>]*'
                    r'property=["\']og:image["\']', html
                )
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None


async def download_image(session, url, target_size=(280, 160)):
    """Download an image from URL and return as Pillow Image, resized."""
    if not url:
        return None
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10),
                               headers={"User-Agent": "Mozilla/5.0"}) as resp:
            if resp.status == 200 and "image" in resp.content_type:
                data = await resp.read()
                if len(data) < 500:
                    return None
                img = Image.open(BytesIO(data)).convert("RGB")
                # Crop to target aspect ratio then resize
                tw, th = target_size
                target_ratio = tw / th
                iw, ih = img.size
                img_ratio = iw / ih
                if img_ratio > target_ratio:
                    new_w = int(ih * target_ratio)
                    left = (iw - new_w) // 2
                    img = img.crop((left, 0, left + new_w, ih))
                else:
                    new_h = int(iw / target_ratio)
                    top = (ih - new_h) // 2
                    img = img.crop((0, top, iw, top + new_h))
                img = img.resize(target_size, Image.LANCZOS)
                return img
    except Exception:
        pass
    return None


async def fetch_og_image_urls(articles):
    """Fetch og:image URLs for articles. Updates articles in-place with '_og_image_url'."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for a in articles:
            async def _fetch(article, page_url):
                img_url = article.get("image_url")
                if not img_url and page_url:
                    img_url = await fetch_og_image(session, page_url)
                article["_og_image_url"] = img_url

            tasks.append(_fetch(a, a.get("url", "")))
        await asyncio.gather(*tasks, return_exceptions=True)


# ─── Visual Report Generation (Pillow) ──────────────────────────────────────

# Render at 2x for retina/mobile crispness
SCALE = 2

def get_font(size, bold=False):
    """Get DejaVu Sans font at given size (auto-scaled for hi-res)."""
    scaled = size * SCALE
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, scaled)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", scaled)
    except OSError:
        return ImageFont.load_default()


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def s(val):
    """Scale a pixel value by the global SCALE factor."""
    return int(val * SCALE)


def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = (current_line + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def truncate_url(url, max_len=50):
    """Shorten a URL for display on the image card."""
    short = url.replace("https://", "").replace("http://", "").replace("www.", "")
    if len(short) > max_len:
        short = short[:max_len - 3] + "..."
    return short



def _format_time_ago(published_str):
    """Format a published timestamp as a human-readable time-ago string."""
    if not published_str:
        return ""
    try:
        pub = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - pub
        minutes = int(diff.total_seconds() / 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return str(minutes) + "m ago"
        hours = minutes // 60
        if hours < 24:
            return str(hours) + "h ago"
        days = hours // 24
        return str(days) + "d ago"
    except (ValueError, TypeError, AttributeError):
        return ""


def format_story_caption(article):
    """Build a Telegram caption with all story info (no image text overlay).

    Format: category emoji + name, headline in bold, summary,
    source + time ago, clickable article link. Kept under 1024 chars.
    Python 3.11 safe — no backslashes inside f-string braces.
    """
    cat = article.get("category", "general")
    emoji = CATEGORY_EMOJI.get(cat, "\U0001f4f0")
    cat_name = cat.capitalize()

    title = article.get("title", "Untitled")
    summary = article.get("ai_summary", article.get("summary", ""))
    source = article.get("source", "Unknown")
    time_ago = _format_time_ago(article.get("published", ""))
    url = article.get("url", "")

    # Build source line
    source_parts = [source]
    if time_ago:
        source_parts.append(time_ago)
    source_line = " \u00b7 ".join(source_parts)

    # Escape markdown in dynamic content
    safe_title = _escape_md(title)
    safe_summary = _escape_md(summary)
    safe_source_line = _escape_md(source_line)

    # Assemble caption — headline first so it shows in push notifications
    parts = [
        "*" + safe_title + "*",
        "",
        emoji + " " + _escape_md(cat_name),
        "",
        safe_summary,
        "",
        "\U0001f4e1 " + safe_source_line,
        "\U0001f517 [Read full article](" + url + ")",
    ]
    caption = "\n".join(parts)

    # Ensure under 1024 chars (Telegram limit for photo captions)
    if len(caption) > 1024:
        # Calculate how much space summary can take
        without_summary = "\n".join(parts[:4] + ["", ""] + parts[6:])
        max_summary = 1024 - len(without_summary) - 10
        if max_summary > 20:
            safe_summary = safe_summary[:max_summary] + "..."
        else:
            safe_summary = ""
        parts[4] = safe_summary
        caption = "\n".join(parts)

    # Final safety truncation
    if len(caption) > 1024:
        caption = caption[:1021] + "..."

    return caption


# ─── Telegram Command Handlers ──────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message + setup categories."""
    user = update.effective_user
    _ = get_user_prefs(user.id)
    welcome = (
        "\U0001f30d *World News Bot*\n\n"
        "Welcome! I aggregate news from 10+ global\n"
        "sources, summarize with AI, and deliver\n"
        "visual briefings.\n\n"
        "*Quick Start:*\n"
        "/news \u2014 Top 5 latest stories\n"
        "/briefing \u2014 Visual news report\n"
        "/breaking \u2014 High-impact stories\n"
        "/ask \u2014 Ask about the news\n"
        "/help \u2014 All commands\n\n"
        "Set your categories with /categories"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)


async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Top 5 latest stories with summaries."""
    await update.message.reply_text("\U0001f50d Fetching latest news...")
    articles = load_news_cache()
    if not articles:
        # Try fetching
        await fetch_all_news()
        articles = load_news_cache()
    if not articles:
        await update.message.reply_text("No news articles available yet. Try again shortly.")
        return

    # Sort by newsworthiness and recency
    now = datetime.now(timezone.utc)
    for a in articles:
        try:
            pub = datetime.fromisoformat(a.get("published", "2000-01-01").replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            hours_old = (now - pub).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_old = 999
        a["_score"] = a.get("newsworthiness", 5) - (hours_old * 0.1)

    articles.sort(key=lambda x: x.get("_score", 0), reverse=True)
    top = articles[:15]  # Take more, then cluster down

    # Cluster to remove duplicates in display
    clusters = cluster_articles(top)
    display_clusters = clusters[:5]

    # Store primary articles in context for /deep reference
    context.user_data["last_news"] = [c["primary"] for c in display_clusters]

    lines = ["\U0001f4f0 *Top News Stories*\n"]
    for i, c in enumerate(display_clusters):
        a = c["primary"]
        num = str(i + 1)
        emoji = CATEGORY_EMOJI.get(a.get("category", "general"), "\U0001f4f0")
        title = a.get("title", "Untitled")
        source = a.get("source", "Unknown")
        summary = a.get("ai_summary", a.get("summary", ""))
        takeaway = a.get("ai_takeaway", "")
        nw = a.get("newsworthiness", 5)

        extra_count = len(c["related"])
        source_suffix = ""
        if extra_count > 0:
            source_suffix = " (+" + str(extra_count) + " more sources)"

        text = emoji + " *" + num + ". " + _escape_md(title) + "*\n"
        text += "\U0001f4cd " + _escape_md(source) + source_suffix + " | "
        text += "\u2b50 " + str(nw) + "/10\n"
        if summary:
            text += _escape_md(summary[:200]) + "\n"
        if takeaway:
            text += "\U0001f4a1 _" + _escape_md(takeaway[:150]) + "_\n"
        text += "[Read more](" + a.get("url", "") + ")\n"
        lines.append(text)

    lines.append("\n\U0001f50e Use /deep [1-5] for analysis")
    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def _send_story(bot, chat_id, article):
    """Send a single story: photo with caption if og:image available, else text only."""
    caption = format_story_caption(article)
    image_url = article.get("_og_image_url")
    if image_url:
        try:
            # Try passing URL directly — Telegram can fetch it
            await bot.send_photo(
                chat_id=chat_id,
                photo=image_url,
                caption=caption,
                parse_mode="Markdown",
            )
            return
        except Exception:
            # Fall back to downloading bytes
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        image_url,
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={"User-Agent": "Mozilla/5.0"},
                    ) as resp:
                        if resp.status == 200 and "image" in resp.content_type:
                            data = await resp.read()
                            if len(data) >= 500:
                                await bot.send_photo(
                                    chat_id=chat_id,
                                    photo=data,
                                    caption=caption,
                                    parse_mode="Markdown",
                                )
                                return
            except Exception:
                pass
    # No image or image failed — send text only
    await bot.send_message(
        chat_id=chat_id,
        text=caption,
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


async def _send_clustered_briefing(bot, chat_id, articles):
    """Send articles as clustered briefing. Multi-source clusters get consolidated messages."""
    clusters = cluster_articles(articles)

    # Collect all articles for og:image fetching
    all_arts = []
    for c in clusters:
        all_arts.append(c["primary"])
        all_arts.extend(c["related"])
    await fetch_og_image_urls(all_arts)

    for c in clusters:
        try:
            if c["related"]:
                # Multi-source cluster — consolidate
                consolidated = await consolidate_cluster(c)
                if consolidated:
                    await _send_consolidated_story(
                        bot, chat_id, consolidated, c["primary"]
                    )
                else:
                    # Fallback: send primary as normal
                    await _send_story(bot, chat_id, c["primary"])
            else:
                # Singleton — send as normal
                await _send_story(bot, chat_id, c["primary"])
        except Exception as e:
            logger.warning("Failed to send cluster story: %s", e)
        await asyncio.sleep(0.3)


async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Visual briefing: individual story cards sent as separate messages."""
    await update.message.reply_text("\U0001f3a8 Generating visual briefing...")
    articles = load_news_cache()
    if not articles:
        await fetch_all_news()
        articles = load_news_cache()
    if not articles:
        await update.message.reply_text("No news available yet.")
        return

    # Sort by score
    now = datetime.now(timezone.utc)
    for a in articles:
        try:
            pub = datetime.fromisoformat(a.get("published", "2000-01-01").replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            hours_old = (now - pub).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_old = 999
        a["_score"] = a.get("newsworthiness", 5) - (hours_old * 0.1)

    articles.sort(key=lambda x: x.get("_score", 0), reverse=True)
    top = articles[:7]

    # Header message
    now_ist = datetime.now(IST)
    date_str = now_ist.strftime("%B %d, %Y")
    source_count = len(set(a.get("source", "") for a in top))
    header = (
        "\U0001f4f0 *Morning Briefing* \u2014 " + date_str + "\n"
        + str(len(top)) + " top stories from " + str(source_count) + " sources"
    )
    await update.message.reply_text(header, parse_mode=ParseMode.MARKDOWN)

    # Send stories with clustering
    chat_id = update.effective_chat.id
    await _send_clustered_briefing(context.bot, chat_id, top)

    # Footer message
    await update.message.reply_text(
        "\U0001f50e Use /topic <keyword> to dive deeper into any story",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_breaking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Breaking news: individual story cards for high-impact stories."""
    articles = load_news_cache()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    breaking = []
    for a in articles:
        if a.get("newsworthiness", 0) < 8:
            continue
        try:
            fetched = datetime.fromisoformat(a.get("fetched_at", "2000-01-01").replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if fetched > cutoff:
                breaking.append(a)
        except (ValueError, TypeError):
            continue

    if not breaking:
        await update.message.reply_text(
            "\u2705 No breaking news in the last 2 hours.\n"
            "That's a good thing! Use /news for latest stories."
        )
        return

    breaking.sort(key=lambda x: x.get("newsworthiness", 0), reverse=True)
    top_breaking = breaking[:3]

    # Header
    await update.message.reply_text(
        "\U0001f534 *Breaking News*",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Fetch og:image URLs for articles
    await fetch_og_image_urls(top_breaking)

    # Send each breaking story as photo + caption (or text-only fallback)
    chat_id = update.effective_chat.id
    for a in top_breaking:
        try:
            await _send_story(context.bot, chat_id, a)
        except Exception as e:
            logger.warning("Failed to send breaking story: %s", e)
        await asyncio.sleep(0.3)


async def cmd_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Search cached articles for a topic."""
    if not context.args:
        await update.message.reply_text("Usage: /topic [keyword]\nExample: /topic ukraine")
        return

    keyword = " ".join(context.args).lower()
    articles = load_news_cache()
    matches = []
    for a in articles:
        searchable = (
            a.get("title", "").lower() + " " +
            a.get("ai_summary", "").lower() + " " +
            a.get("summary", "").lower() + " " +
            a.get("category", "").lower()
        )
        if keyword in searchable:
            matches.append(a)

    if not matches:
        await update.message.reply_text("No articles found for: " + keyword)
        return

    matches.sort(key=lambda x: x.get("newsworthiness", 0), reverse=True)
    top = matches[:15]  # Take more, then cluster down

    # Cluster to remove duplicates in display
    clusters = cluster_articles(top)
    display_clusters = clusters[:5]

    lines = ["\U0001f50d *Results for: " + _escape_md(keyword) + "*\n"]
    for i, c in enumerate(display_clusters):
        a = c["primary"]
        num = str(i + 1)
        emoji = CATEGORY_EMOJI.get(a.get("category", "general"), "\U0001f4f0")
        title = a.get("title", "")
        source = a.get("source", "")
        extra_count = len(c["related"])
        source_suffix = ""
        if extra_count > 0:
            source_suffix = " (+" + str(extra_count) + " more sources)"
        lines.append(
            emoji + " *" + num + ".* " + _escape_md(title) + "\n"
            + _escape_md(source) + source_suffix + " | \u2b50 " + str(a.get("newsworthiness", 5)) + "/10\n"
            + "[Read](" + a.get("url", "") + ")\n"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def cmd_perspectives(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Multi-source viewpoint comparison for a topic."""
    if not context.args:
        await update.message.reply_text("Usage: /perspectives <topic>\nExample: /perspectives Ukraine war")
        return

    query = " ".join(context.args).lower()
    await update.message.reply_text("\U0001f50d Analyzing perspectives on: " + _escape_md(query) + "...", parse_mode=ParseMode.MARKDOWN)

    articles = load_news_cache()
    # Find matching articles
    matches = []
    for a in articles:
        searchable = (
            a.get("title", "").lower() + " " +
            a.get("ai_summary", "").lower() + " " +
            a.get("summary", "").lower() + " " +
            a.get("category", "").lower()
        )
        if query in searchable:
            matches.append(a)

    # Group by source, pick best from each
    by_source = {}
    for a in matches:
        src = a.get("source", "Unknown")
        if src not in by_source:
            by_source[src] = a
        else:
            # Keep higher newsworthiness
            if a.get("newsworthiness", 0) > by_source[src].get("newsworthiness", 0):
                by_source[src] = a

    if len(by_source) < 2:
        await update.message.reply_text(
            "Not enough sources covering this topic for comparison.\n"
            "Need at least 2 different sources. Try a broader topic."
        )
        return

    # Take up to 5 sources
    source_articles = list(by_source.values())[:5]

    # Call Claude to analyze perspectives
    analysis = await analyze_perspectives(query, source_articles)
    if not analysis:
        await update.message.reply_text("Unable to generate perspective analysis right now. Try again later.")
        return

    # Parse and format the analysis
    display_query = " ".join(context.args)
    source_count = str(len(source_articles))
    lines = [
        "\U0001f50d *Perspectives: " + _escape_md(display_query) + "*",
        "",
        "Analyzing coverage from " + source_count + " sources...",
        "",
    ]

    # Parse structured response
    parts = analysis.split("---")
    source_analysis = parts[0].strip() if parts else analysis
    overall = parts[1].strip() if len(parts) > 1 else ""

    # Format source analyses
    for source_line in source_analysis.split("\n"):
        source_line = source_line.strip()
        if not source_line:
            continue
        if ":" in source_line:
            src_name, src_text = source_line.split(":", 1)
            lines.append(
                "\U0001f4f0 *" + _escape_md(src_name.strip()) + "*"
            )
            lines.append(_escape_md(src_text.strip()))
            lines.append("")
        else:
            lines.append(_escape_md(source_line))

    if overall:
        # Strip "OVERALL:" prefix if present
        overall_text = overall
        if overall_text.upper().startswith("OVERALL:"):
            overall_text = overall_text[8:]
        lines.append("\u2696\ufe0f *Overall*: " + _escape_md(overall_text.strip()))

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:4000] + "..."
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)


async def cmd_deep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Deep AI analysis of a story."""
    if not context.args:
        await update.message.reply_text("Usage: /deep [number]\nUse story numbers from /news (1-5)")
        return

    try:
        num = int(context.args[0]) - 1
    except ValueError:
        await update.message.reply_text("Please provide a valid story number.")
        return

    last_news = context.user_data.get("last_news", [])
    if not last_news:
        await update.message.reply_text("Run /news first, then use /deep [1-5]")
        return

    if num < 0 or num >= len(last_news):
        await update.message.reply_text("Invalid number. Use 1-" + str(len(last_news)))
        return

    article = last_news[num]
    await update.message.reply_text("\U0001f9e0 Generating deep analysis...")

    question = " ".join(context.args[1:]) if len(context.args) > 1 else None
    analysis = await deep_analysis(article, question)

    title = article.get("title", "")
    text = (
        "\U0001f9e0 *Deep Analysis*\n\n"
        "*" + _escape_md(title) + "*\n"
        + _escape_md(article.get("source", "")) + "\n\n"
        + _escape_md(analysis)
    )
    # Split if too long
    if len(text) > 4000:
        text = text[:4000] + "..."
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Conversational Q&A grounded in cached news."""
    if not context.args:
        await update.message.reply_text(
            "Usage: /ask [question]\n"
            "Example: /ask What's happening with AI regulation?"
        )
        return

    question = " ".join(context.args)
    user_id = update.effective_user.id
    await update.message.reply_text("\U0001f914 Thinking...")

    # Search articles for relevant context
    articles = load_news_cache()
    keywords = question.lower().split()
    scored = []
    for a in articles:
        searchable = (
            a.get("title", "").lower() + " " +
            a.get("ai_summary", "").lower() + " " +
            a.get("summary", "").lower() + " " +
            a.get("category", "").lower()
        )
        score = sum(1 for kw in keywords if len(kw) > 2 and kw in searchable)
        if score > 0:
            scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [a for _, a in scored[:10]]

    # Build context
    articles_context = ""
    for a in relevant:
        articles_context += "Title: " + a.get("title", "") + "\n"
        articles_context += "Source: " + a.get("source", "") + "\n"
        articles_context += "Published: " + a.get("published", "") + "\n"
        articles_context += "Summary: " + a.get("ai_summary", a.get("summary", "")) + "\n"
        articles_context += "Takeaway: " + a.get("ai_takeaway", "") + "\n\n"

    if not articles_context:
        articles_context = "No relevant articles found in the cache."

    history = get_conversation_history(user_id)
    answer = await ask_claude(question, articles_context, history)
    add_conversation_exchange(user_id, question, answer)

    text = "\U0001f4ac *News Q&A*\n\n" + _escape_md(answer)
    if relevant:
        text += "\n\n\U0001f4ce _Based on " + str(len(relevant)) + " articles_"
    if len(text) > 4000:
        text = text[:4000] + "..."
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View/set preferred categories."""
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    user_cats = prefs.get("categories", list(CATEGORIES.keys()))

    keyboard = []
    for cat in CATEGORIES:
        emoji = CATEGORY_EMOJI.get(cat, "")
        check = "\u2705 " if cat in user_cats else "\u274c "
        keyboard.append([InlineKeyboardButton(
            check + emoji + " " + cat.capitalize(),
            callback_data="cat_toggle_" + cat,
        )])
    keyboard.append([InlineKeyboardButton("\u2705 Done", callback_data="cat_done")])

    await update.message.reply_text(
        "\U0001f3af *Select Your Categories*\n"
        "Tap to toggle on/off:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle category toggle callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "cat_done":
        await query.edit_message_text("\u2705 Categories saved!")
        return

    if data.startswith("cat_toggle_"):
        cat = data.replace("cat_toggle_", "")
        user_id = update.effective_user.id
        prefs = get_user_prefs(user_id)
        user_cats = prefs.get("categories", list(CATEGORIES.keys()))

        if cat in user_cats:
            user_cats.remove(cat)
        else:
            user_cats.append(cat)

        set_user_prefs(user_id, "categories", user_cats)

        # Rebuild keyboard
        keyboard = []
        for c in CATEGORIES:
            emoji = CATEGORY_EMOJI.get(c, "")
            check = "\u2705 " if c in user_cats else "\u274c "
            keyboard.append([InlineKeyboardButton(
                check + emoji + " " + c.capitalize(),
                callback_data="cat_toggle_" + c,
            )])
        keyboard.append([InlineKeyboardButton("\u2705 Done", callback_data="cat_done")])

        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))


async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set daily briefing time (IST)."""
    if not context.args:
        prefs = get_user_prefs(update.effective_user.id)
        current = prefs.get("schedule", "07:00")
        await update.message.reply_text(
            "\u23f0 *Daily Briefing Schedule*\n\n"
            "Current: " + current + " IST\n\n"
            "Set new time: /schedule HH:MM\n"
            "Example: /schedule 08:30",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    time_str = context.args[0]
    if not re.match(r'^\d{2}:\d{2}$', time_str):
        await update.message.reply_text("Invalid format. Use HH:MM (e.g., 08:30)")
        return

    try:
        hour, minute = map(int, time_str.split(":"))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError()
    except ValueError:
        await update.message.reply_text("Invalid time. Use HH:MM (00:00 - 23:59)")
        return

    set_user_prefs(update.effective_user.id, "schedule", time_str)
    await update.message.reply_text(
        "\u2705 Daily briefing set to " + time_str + " IST"
    )


async def cmd_sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show active news sources and status."""
    articles = load_news_cache()
    source_counts = {}
    for a in articles:
        src = a.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    lines = ["\U0001f4e1 *News Sources*\n"]
    for source in RSS_FEEDS:
        count = source_counts.get(source, 0)
        status = "\U0001f7e2" if count > 0 else "\U0001f534"
        lines.append(status + " " + source + " (" + str(count) + " articles)")

    if GNEWS_API_KEY:
        gnews_count = sum(v for k, v in source_counts.items() if k.startswith("GNews"))
        status = "\U0001f7e2" if gnews_count > 0 else "\U0001f7e1"
        lines.append(status + " GNews API (" + str(gnews_count) + " articles)")
    else:
        lines.append("\u26aa GNews API (not configured)")

    lines.append("\n\U0001f4ca Total cached: " + str(len(articles)) + " articles")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Overall news sentiment gauge."""
    articles = load_news_cache()
    if not articles:
        await update.message.reply_text("No articles cached yet.")
        return

    # Recent articles (last 24h)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = []
    for a in articles:
        try:
            fetched = datetime.fromisoformat(a.get("fetched_at", "2000-01-01").replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if fetched > cutoff:
                recent.append(a)
        except (ValueError, TypeError):
            continue

    if not recent:
        recent = articles[-20:]

    # Analyze by category
    cat_counts = {}
    total_nw = 0
    conflict_count = 0
    for a in recent:
        cat = a.get("category", "general")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        total_nw += a.get("newsworthiness", 5)
        if cat in ("conflict", "geopolitics"):
            conflict_count += 1

    avg_nw = total_nw / len(recent) if recent else 5

    # Sentiment gauge
    if conflict_count > len(recent) * 0.5:
        mood = "\U0001f534 Tense"
        bar = "\u2588" * 8 + "\u2591" * 2
    elif conflict_count > len(recent) * 0.3:
        mood = "\U0001f7e0 Cautious"
        bar = "\u2588" * 6 + "\u2591" * 4
    elif avg_nw > 7:
        mood = "\U0001f7e1 Eventful"
        bar = "\u2588" * 5 + "\u2591" * 5
    else:
        mood = "\U0001f7e2 Calm"
        bar = "\u2588" * 3 + "\u2591" * 7

    lines = [
        "\U0001f4ca *News Sentiment Gauge*\n",
        "Mood: " + mood,
        "Intensity: [" + bar + "] " + "{:.1f}".format(avg_nw) + "/10",
        "",
        "*Category Breakdown:*",
    ]
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats:
        emoji = CATEGORY_EMOJI.get(cat, "\U0001f4f0")
        pct = count * 100 // len(recent)
        bar_len = max(1, count * 10 // len(recent))
        lines.append(
            emoji + " " + cat.capitalize() + ": "
            + "\u2588" * bar_len + " " + str(pct) + "%"
        )

    lines.append("\n\U0001f4f0 Based on " + str(len(recent)) + " recent articles")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Adjust preferences."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "\u2699\ufe0f *Settings*\n\n"
            "/set summary short|medium|long\n"
            "/set schedule HH:MM\n"
            "/set alerts on|off\n"
            "\nCurrent preferences:\n"
            + _format_prefs(get_user_prefs(update.effective_user.id)),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    param = context.args[0].lower()
    value = context.args[1].lower()

    if param == "summary":
        if value not in ("short", "medium", "long"):
            await update.message.reply_text("Options: short, medium, long")
            return
        set_user_prefs(update.effective_user.id, "summary_length", value)
        await update.message.reply_text("\u2705 Summary length set to: " + value)
    elif param == "schedule":
        if not re.match(r'^\d{2}:\d{2}$', value):
            await update.message.reply_text("Format: HH:MM")
            return
        set_user_prefs(update.effective_user.id, "schedule", value)
        await update.message.reply_text("\u2705 Schedule set to: " + value + " IST")
    elif param == "alerts":
        if value not in ("on", "off"):
            await update.message.reply_text("Options: on, off")
            return
        set_user_prefs(update.effective_user.id, "alerts", value == "on")
        status = "enabled" if value == "on" else "disabled"
        await update.message.reply_text("\u2705 Breaking news alerts " + status)
    else:
        await update.message.reply_text("Unknown setting: " + param)


async def cmd_follow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Subscribe to keyword alerts."""
    if not context.args:
        await update.message.reply_text("Usage: /follow <keyword>\nExample: /follow Ukraine")
        return

    keyword = " ".join(context.args).strip()
    if not keyword:
        await update.message.reply_text("Please provide a keyword to follow.")
        return

    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    follows = prefs.get("follows", [])

    # Check if already following (case-insensitive)
    for f in follows:
        if f.lower() == keyword.lower():
            await update.message.reply_text("You're already following '" + keyword + "'.")
            return

    follows.append(keyword)
    set_user_prefs(user_id, "follows", follows)
    await update.message.reply_text(
        "\u2705 Now following '" + keyword + "'.\n"
        "You'll get alerts for matching stories."
    )


async def cmd_unfollow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unsubscribe from keyword alerts."""
    if not context.args:
        await update.message.reply_text("Usage: /unfollow <keyword>\nExample: /unfollow Ukraine")
        return

    keyword = " ".join(context.args).strip()
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    follows = prefs.get("follows", [])

    # Case-insensitive removal
    new_follows = [f for f in follows if f.lower() != keyword.lower()]
    if len(new_follows) == len(follows):
        await update.message.reply_text("You weren't following '" + keyword + "'.")
        return

    set_user_prefs(user_id, "follows", new_follows)
    await update.message.reply_text("Unfollowed '" + keyword + "'.")


async def cmd_following(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List current keyword follows."""
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    follows = prefs.get("follows", [])

    if not follows:
        await update.message.reply_text(
            "You're not following any topics.\n"
            "Use /follow <keyword> to start."
        )
        return

    follows_str = ", ".join(follows)
    await update.message.reply_text("You're following: " + follows_str)


def _matches_follow_keyword(keyword, article):
    """Check if a keyword matches an article's title or summary.

    For multi-word keywords, all words must appear (not necessarily adjacent).
    """
    text = (
        article.get("title", "").lower() + " " +
        article.get("ai_summary", "").lower() + " " +
        article.get("summary", "").lower()
    )
    words = keyword.lower().split()
    return all(w in text for w in words)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command list (64-char width for mobile)."""
    help_text = (
        "\U0001f30d *World News Bot \u2014 Commands*\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "/news     \u2014 Top 5 stories with AI\n"
        "            summaries\n"
        "/briefing \u2014 Clustered news report\n"
        "            (groups related stories)\n"
        "/breaking \u2014 High-impact stories\n"
        "            (newsworthiness 8+)\n"
        "/topic    \u2014 Search by keyword\n"
        "            Usage: /topic ukraine\n"
        "/deep     \u2014 Deep analysis of a\n"
        "            story from /news\n"
        "/ask      \u2014 Ask about the news\n"
        "            (AI-powered Q&A)\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "/perspectives \u2014 Compare how sources\n"
        "    cover the same topic differently\n"
        "    Usage: /perspectives AI regulation\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "/follow   \u2014 Get alerts for a keyword\n"
        "            Usage: /follow Ukraine\n"
        "/unfollow \u2014 Stop keyword alerts\n"
        "/following \u2014 List followed keywords\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "/categories \u2014 Set preferred topics\n"
        "/schedule   \u2014 Daily briefing time\n"
        "/sources    \u2014 View news sources\n"
        "/sentiment  \u2014 News mood gauge\n"
        "/set        \u2014 Adjust preferences\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "\U0001f514 *Auto-Alerts:* Breaking news (8+)\n"
        "is auto-sent every 30 min.\n"
        "Toggle: /set alerts on|off\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "Powered by Claude AI + 14 sources"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


# ─── Utility Functions ───────────────────────────────────────────────────────

def _escape_md(text):
    """Escape markdown special chars for Telegram."""
    if not text:
        return ""
    # Only escape characters that cause issues in Markdown V1
    for ch in ['_', '*', '[', ']', '`']:
        text = text.replace(ch, '\\' + ch)
    return text


def _format_prefs(prefs):
    cats = prefs.get("categories", [])
    cat_str = ", ".join(cats) if cats else "all"
    alerts_status = "on" if prefs.get("alerts", True) else "off"
    return (
        "Categories: " + cat_str + "\n"
        "Schedule: " + prefs.get("schedule", "07:00") + " IST\n"
        "Summary: " + prefs.get("summary_length", "medium") + "\n"
        "Alerts: " + alerts_status
    )


# ─── Scheduled Tasks ────────────────────────────────────────────────────────

async def scheduled_fetch(app):
    """Periodic news fetch + auto-send breaking alerts + follow keyword alerts."""
    try:
        count = await fetch_all_news()
        logger.info("Scheduled fetch completed: %d new articles", count)

        # Auto-send breaking news alerts (newsworthiness >= 8)
        articles = load_news_cache()
        new_breaking = []
        for a in articles:
            aid = a.get("id", "")
            if aid and a.get("newsworthiness", 0) >= 8 and not is_alert_sent(aid):
                new_breaking.append(a)
                mark_alert_sent(aid)
            if len(new_breaking) >= 5:
                break

        if new_breaking:
            # Cluster breaking alerts to avoid sending near-duplicate stories
            breaking_clusters = cluster_articles(new_breaking)
            deduped_breaking = [c["primary"] for c in breaking_clusters]
            logger.info("Auto-alerting %d breaking stories (%d after dedup)", len(new_breaking), len(deduped_breaking))
            await fetch_og_image_urls(deduped_breaking)
            prefs = load_json(USER_PREFS_FILE, {})
            for uid, user_prefs in prefs.items():
                if not user_prefs.get("alerts", True):
                    continue
                try:
                    chat_id = int(uid)
                    for a in deduped_breaking:
                        try:
                            await _send_story(app.bot, chat_id, a)
                        except Exception as e:
                            logger.warning("Alert send failed for %s: %s", uid, e)
                        await asyncio.sleep(0.3)
                except Exception as e:
                    logger.warning("Alert failed for user %s: %s", uid, e)

        # Follow keyword alerts
        prefs = load_json(USER_PREFS_FILE, {})
        # Collect all follow-alert articles to fetch og:images for
        follow_alert_articles = []
        user_alerts = {}  # uid -> list of (article, keyword)

        for uid, user_prefs in prefs.items():
            follows = user_prefs.get("follows", [])
            if not follows:
                continue
            matched = []
            for a in articles:
                aid = a.get("id", "")
                if not aid:
                    continue
                for kw in follows:
                    if is_follow_alert_sent(uid, aid, kw):
                        continue
                    if _matches_follow_keyword(kw, a):
                        matched.append((a, kw))
                        if len(matched) >= 3:
                            break
                if len(matched) >= 3:
                    break
            if matched:
                # Cluster matched articles to avoid sending near-duplicate stories
                matched_articles_only = [a for a, _kw in matched]
                follow_clusters = cluster_articles(matched_articles_only)
                primary_ids = {c["primary"].get("id") for c in follow_clusters}
                deduped_matched = [(a, kw) for a, kw in matched if a.get("id") in primary_ids]
                user_alerts[uid] = deduped_matched
                for a, _kw in deduped_matched:
                    follow_alert_articles.append(a)

        if follow_alert_articles:
            await fetch_og_image_urls(follow_alert_articles)

        for uid, matched in user_alerts.items():
            try:
                chat_id = int(uid)
                for a, kw in matched:
                    try:
                        # Send with a follow-alert header
                        header = "\U0001f514 *Keyword Alert:* " + _escape_md(kw)
                        await app.bot.send_message(
                            chat_id=chat_id,
                            text=header,
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        await _send_story(app.bot, chat_id, a)
                        mark_follow_alert_sent(uid, a.get("id", ""), kw)
                    except Exception as e:
                        logger.warning("Follow alert failed for %s: %s", uid, e)
                    await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning("Follow alert failed for user %s: %s", uid, e)
    except Exception as e:
        logger.error("Scheduled fetch error: %s", e)


async def scheduled_briefing(app):
    """Send scheduled briefings as photo + caption messages."""
    try:
        articles = load_news_cache()
        if not articles:
            return

        now = datetime.now(timezone.utc)
        for a in articles:
            try:
                pub = datetime.fromisoformat(a.get("published", "2000-01-01").replace("Z", "+00:00"))
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                hours_old = (now - pub).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_old = 999
            a["_score"] = a.get("newsworthiness", 5) - (hours_old * 0.1)

        articles.sort(key=lambda x: x.get("_score", 0), reverse=True)
        top = articles[:7]

        prefs = load_json(USER_PREFS_FILE, {})
        now_ist = datetime.now(IST)
        current_time = now_ist.strftime("%H:%M")
        date_str = now_ist.strftime("%B %d, %Y")

        # Fetch og:image URLs once for all users
        await fetch_og_image_urls(top)

        source_count = len(set(a.get("source", "") for a in top))

        for uid, user_prefs in prefs.items():
            schedule = user_prefs.get("schedule", "07:00")
            # Check if within 15 min window
            try:
                sh, sm = map(int, schedule.split(":"))
                ch, cm = map(int, current_time.split(":"))
                diff = abs((sh * 60 + sm) - (ch * 60 + cm))
                if diff > 15:
                    continue
            except ValueError:
                continue

            try:
                chat_id = int(uid)
                # Header message
                header = (
                    "\U0001f4f0 *Scheduled Briefing* \u2014 " + date_str + "\n"
                    + str(len(top)) + " top stories from " + str(source_count) + " sources"
                )
                await app.bot.send_message(
                    chat_id=chat_id,
                    text=header,
                    parse_mode=ParseMode.MARKDOWN,
                )

                # Send stories with clustering
                await _send_clustered_briefing(app.bot, chat_id, top)

                # Footer
                await app.bot.send_message(
                    chat_id=chat_id,
                    text="\U0001f50e Use /topic <keyword> to dive deeper into any story",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception as e:
                logger.warning("Failed to send briefing to %s: %s", uid, e)
    except Exception as e:
        logger.error("Scheduled briefing error: %s", e)


def cleanup_cache_duplicates():
    """Cluster the entire cache and remove duplicates, keeping only primary articles."""
    articles = load_news_cache()
    if len(articles) < 2:
        return

    clusters = cluster_articles(articles)
    # Keep only the primary (highest newsworthiness) from each cluster
    kept = []
    removed = 0
    for c in clusters:
        kept.append(c["primary"])
        removed += len(c["related"])

    if removed == 0:
        logger.info("Cache dedup: no duplicates found in %d articles", len(articles))
        return

    # Rewrite cache with only primary articles
    ensure_data_dir()
    path = get_path(NEWS_CACHE_FILE)
    with open(path, "w") as f:
        for a in kept:
            f.write(json.dumps(a, default=str) + "\n")
    logger.info("Cache dedup: %d -> %d articles (%d duplicates removed)", len(articles), len(kept), removed)


async def scheduled_cleanup():
    """Daily cache cleanup + dedup."""
    try:
        cleanup_cache_duplicates()
        trim_news_cache(14)
        logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error("Cleanup error: %s", e)


# ─── Health Check HTTP Server ───────────────────────────────────────────────

async def health_check(request):
    articles = load_news_cache()
    return web.json_response({
        "status": "ok",
        "articles_cached": len(articles),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def start_health_server():
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info("Health check server running on port %d", PORT)


# ─── Bot Setup & Main ───────────────────────────────────────────────────────

async def post_init(application):
    """Post-initialization: set commands, start scheduler, start health server."""
    # Set bot commands
    commands = [
        BotCommand("start", "Welcome + setup"),
        BotCommand("news", "Top 5 stories"),
        BotCommand("briefing", "Clustered news report"),
        BotCommand("breaking", "High-impact stories"),
        BotCommand("topic", "Search by keyword"),
        BotCommand("perspectives", "Multi-source viewpoint comparison"),
        BotCommand("deep", "Deep analysis"),
        BotCommand("ask", "Ask about the news"),
        BotCommand("follow", "Follow a keyword for alerts"),
        BotCommand("unfollow", "Stop following a keyword"),
        BotCommand("following", "List followed keywords"),
        BotCommand("categories", "Set topics"),
        BotCommand("schedule", "Daily briefing time"),
        BotCommand("sources", "News sources"),
        BotCommand("sentiment", "Sentiment gauge"),
        BotCommand("set", "Preferences"),
        BotCommand("help", "All commands"),
    ]
    await application.bot.set_my_commands(commands)

    # Start health server
    await start_health_server()

    # Start scheduler
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        scheduled_fetch, "interval", minutes=30, args=[application],
        id="fetch_news", replace_existing=True,
    )
    # Morning briefing: 7:00 AM IST = 1:30 UTC
    scheduler.add_job(
        scheduled_briefing, "cron", hour=1, minute=30, args=[application],
        id="morning_briefing", replace_existing=True,
    )
    # Evening briefing: 6:00 PM IST = 12:30 UTC
    scheduler.add_job(
        scheduled_briefing, "cron", hour=12, minute=30, args=[application],
        id="evening_briefing", replace_existing=True,
    )
    # Cleanup: midnight IST = 18:30 UTC
    scheduler.add_job(
        scheduled_cleanup, "cron", hour=18, minute=30,
        id="cleanup", replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started")

    # Initial fetch
    ensure_data_dir()
    asyncio.create_task(fetch_all_news())


def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set!")
        sys.exit(1)
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — AI features will be limited")

    ensure_data_dir()

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Register handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("news", cmd_news))
    application.add_handler(CommandHandler("briefing", cmd_briefing))
    application.add_handler(CommandHandler("breaking", cmd_breaking))
    application.add_handler(CommandHandler("topic", cmd_topic))
    application.add_handler(CommandHandler("perspectives", cmd_perspectives))
    application.add_handler(CommandHandler("deep", cmd_deep))
    application.add_handler(CommandHandler("ask", cmd_ask))
    application.add_handler(CommandHandler("follow", cmd_follow))
    application.add_handler(CommandHandler("unfollow", cmd_unfollow))
    application.add_handler(CommandHandler("following", cmd_following))
    application.add_handler(CommandHandler("categories", cmd_categories))
    application.add_handler(CommandHandler("schedule", cmd_schedule))
    application.add_handler(CommandHandler("sources", cmd_sources))
    application.add_handler(CommandHandler("sentiment", cmd_sentiment))
    application.add_handler(CommandHandler("set", cmd_set))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CallbackQueryHandler(category_callback, pattern=r"^cat_"))

    logger.info("Starting World News Bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
