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

HAIKU_MODEL = "claude-3-5-haiku-20241022"
SONNET_MODEL = "claude-sonnet-4-20250514"

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
}

CATEGORY_COLORS = {
    "geopolitics": "#3b82f6",
    "economy": "#22c55e",
    "technology": "#a855f7",
    "climate": "#06b6d4",
    "conflict": "#ef4444",
    "science": "#f59e0b",
    "markets": "#ec4899",
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

    new_articles = []
    for raw in all_raw:
        aid = hash_url(raw["url"])
        if aid in existing_ids:
            continue
        if is_duplicate_title(raw["title"], existing_titles):
            continue
        existing_ids.add(aid)
        existing_titles.append(raw["title"])
        new_articles.append(raw)

    logger.info("New unique articles: %d", len(new_articles))

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


async def fetch_article_images(articles):
    """Download thumbnail images for articles. Updates articles in-place."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for a in articles:
            img_url = a.get("image_url")
            article_url = a.get("url", "")

            async def _fetch(article, image_url, page_url):
                # Try image_url first, then og:image from page
                img = None
                if image_url:
                    img = await download_image(session, image_url)
                if img is None and page_url:
                    og_url = await fetch_og_image(session, page_url)
                    if og_url:
                        article["image_url"] = og_url
                        img = await download_image(session, og_url)
                article["_pil_image"] = img

            tasks.append(_fetch(a, img_url, article_url))
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


def render_story_card(article, scale=2):
    """Render a clean individual story card image (800x420 at 2x = 1600x840px).

    Layout: dark bg, thumbnail on the left, headline + source + category badge
    + newsworthiness dots on the right. Returns BytesIO PNG.
    """
    old_scale = globals().get("SCALE", 2)
    sc = scale

    def _s(val):
        return int(val * sc)

    WIDTH = _s(800)
    HEIGHT = _s(420)
    PADDING = _s(30)
    THUMB_W = _s(260)
    THUMB_H = _s(220)
    THUMB_RADIUS = _s(12)
    BG_COLOR = hex_to_rgb("#1a1a2e")
    CARD_BG = hex_to_rgb("#16213e")
    ACCENT_COLOR = hex_to_rgb("#0f3460")

    font_headline = get_font(18, bold=True)
    font_body = get_font(14)
    font_small = get_font(12)
    font_badge = get_font(11, bold=True)

    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Card background with rounded corners
    draw.rounded_rectangle(
        [_s(10), _s(10), WIDTH - _s(10), HEIGHT - _s(10)],
        radius=_s(16),
        fill=CARD_BG,
    )

    # --- Left side: thumbnail ---
    thumb_x = PADDING
    thumb_y = (HEIGHT - THUMB_H) // 2
    has_img = article.get("_pil_image") is not None

    if has_img:
        pil_img = article["_pil_image"]
        thumb = pil_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        mask = Image.new("L", (THUMB_W, THUMB_H), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([0, 0, THUMB_W, THUMB_H], radius=THUMB_RADIUS, fill=255)
        img.paste(thumb, (thumb_x, thumb_y), mask)
    else:
        # Colored fallback block based on category
        cat = article.get("category", "general")
        fallback_color = hex_to_rgb(CATEGORY_COLORS.get(cat, "#6b7280"))
        # Darken the color slightly for the block
        dark_color = tuple(max(0, c - 40) for c in fallback_color)
        draw.rounded_rectangle(
            [thumb_x, thumb_y, thumb_x + THUMB_W, thumb_y + THUMB_H],
            radius=THUMB_RADIUS,
            fill=dark_color,
        )
        # Draw category emoji/label centered in fallback
        cat_label = cat.upper()
        bbox = draw.textbbox((0, 0), cat_label, font=font_badge)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        draw.text(
            (thumb_x + (THUMB_W - lw) // 2, thumb_y + (THUMB_H - lh) // 2),
            cat_label, font=font_badge, fill=(255, 255, 255, 180),
        )

    # --- Right side: text content ---
    text_x = thumb_x + THUMB_W + _s(25)
    text_w = WIDTH - text_x - PADDING
    cy = PADDING + _s(10)

    # Category badge (colored pill)
    cat = article.get("category", "general")
    cat_color = hex_to_rgb(CATEGORY_COLORS.get(cat, "#6b7280"))
    badge_text = cat.upper()
    bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw = bbox[2] - bbox[0] + _s(14)
    bh = _s(22)
    draw.rounded_rectangle(
        [text_x, cy, text_x + bw, cy + bh],
        radius=_s(11),
        fill=cat_color,
    )
    draw.text((text_x + _s(7), cy + _s(3)), badge_text, font=font_badge, fill=(255, 255, 255))

    # Newsworthiness dots next to badge
    nw = article.get("newsworthiness", 5)
    dots_x = text_x + bw + _s(15)
    for d in range(10):
        dot_color = (255, 200, 50) if d < int(nw) else (60, 60, 80)
        dx = dots_x + d * _s(11)
        draw.ellipse([dx, cy + _s(5), dx + _s(8), cy + _s(13)], fill=dot_color)

    cy += bh + _s(12)

    # Headline (white, bold, 2-3 lines max with ellipsis)
    title = article.get("title", "Untitled")
    headline_lines = wrap_text(title, font_headline, text_w, draw)
    for line in headline_lines[:3]:
        draw.text((text_x, cy), line, font=font_headline, fill=(255, 255, 255))
        cy += _s(24)
    if len(headline_lines) > 3:
        # Add ellipsis to last visible line
        last_line = headline_lines[2]
        if len(last_line) > 3:
            last_line = last_line[:-3] + "..."
        # Redraw last line with ellipsis
        draw.rectangle(
            [text_x, cy - _s(24), text_x + text_w, cy],
            fill=CARD_BG,
        )
        draw.text((text_x, cy - _s(24)), last_line, font=font_headline, fill=(255, 255, 255))
    cy += _s(10)

    # Summary (1-2 lines, gray)
    summary = article.get("ai_summary", article.get("summary", ""))
    if summary:
        summary_lines = wrap_text(summary, font_body, text_w, draw)
        for line in summary_lines[:2]:
            draw.text((text_x, cy), line, font=font_body, fill=(180, 180, 200))
            cy += _s(20)
        cy += _s(8)

    # Source + time ago (gray, smaller)
    source = article.get("source", "Unknown")
    time_ago = _format_time_ago(article.get("published", ""))
    source_line = source
    if time_ago:
        source_line = source + "  |  " + time_ago
    draw.text((text_x, cy), source_line, font=font_small, fill=(120, 120, 150))

    # Bottom accent line
    draw.line(
        [(_s(10), HEIGHT - _s(14)), (WIDTH - _s(10), HEIGHT - _s(14))],
        fill=ACCENT_COLOR,
        width=_s(2),
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


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
    """Build a Telegram caption for a story card photo message.

    Format: category emoji + name, 1-2 line summary, source + time ago,
    clickable 'Read full article' link. Kept under 1024 chars.
    Python 3.11 safe — no backslashes inside f-string braces.
    """
    cat = article.get("category", "general")
    emoji = CATEGORY_EMOJI.get(cat, "\U0001f4f0")
    cat_name = cat.capitalize()

    summary = article.get("ai_summary", article.get("summary", ""))
    # Truncate summary to ~300 chars to stay under 1024 total
    if len(summary) > 300:
        summary = summary[:297] + "..."

    source = article.get("source", "Unknown")
    time_ago = _format_time_ago(article.get("published", ""))
    url = article.get("url", "")

    # Build source line
    source_parts = [source]
    if time_ago:
        source_parts.append(time_ago)
    source_line = " \u00b7 ".join(source_parts)

    # Escape markdown in dynamic content
    safe_summary = _escape_md(summary)
    safe_source_line = _escape_md(source_line)

    # Assemble caption — no backslashes inside f-string braces
    parts = [
        emoji + " *" + _escape_md(cat_name) + "*",
        "",
        safe_summary,
        "",
        "\U0001f4e1 " + safe_source_line,
        "\U0001f517 [Read full article](" + url + ")",
    ]
    caption = "\n".join(parts)

    # Ensure under 1024 chars (Telegram limit for photo captions)
    if len(caption) > 1024:
        # Shorten summary further
        max_summary = 1024 - (len(caption) - len(safe_summary)) - 20
        if max_summary > 0:
            safe_summary = safe_summary[:max_summary] + "..."
        parts[2] = safe_summary
        caption = "\n".join(parts)

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
    top = articles[:5]

    # Store in context for /deep reference
    context.user_data["last_news"] = top

    lines = ["\U0001f4f0 *Top News Stories*\n"]
    for i, a in enumerate(top):
        num = str(i + 1)
        emoji = CATEGORY_EMOJI.get(a.get("category", "general"), "\U0001f4f0")
        title = a.get("title", "Untitled")
        source = a.get("source", "Unknown")
        summary = a.get("ai_summary", a.get("summary", ""))
        takeaway = a.get("ai_takeaway", "")
        nw = a.get("newsworthiness", 5)

        text = emoji + " *" + num + ". " + _escape_md(title) + "*\n"
        text += "\U0001f4cd " + _escape_md(source) + " | "
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

    # Fetch article images
    await fetch_article_images(top)

    # Header message
    now_ist = datetime.now(IST)
    date_str = now_ist.strftime("%B %d, %Y")
    source_count = len(set(a.get("source", "") for a in top))
    header = (
        "\U0001f4f0 *Morning Briefing* \u2014 " + date_str + "\n"
        + str(len(top)) + " top stories from " + str(source_count) + " sources"
    )
    await update.message.reply_text(header, parse_mode=ParseMode.MARKDOWN)

    # Send each story as an individual photo + caption
    chat_id = update.effective_chat.id
    for a in top:
        try:
            img_buf = render_story_card(a, scale=2)
            caption = format_story_caption(a)
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=img_buf,
                caption=caption,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning("Failed to send story card: %s", e)
        await asyncio.sleep(0.3)

    # Footer message
    await update.message.reply_text(
        "\U0001f50e Use /topic <keyword> to dive deeper into any story",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Cleanup pil images from memory
    for a in top:
        a.pop("_pil_image", None)


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

    # Fetch article images
    await fetch_article_images(top_breaking)

    # Send each breaking story as individual photo + caption
    chat_id = update.effective_chat.id
    for a in top_breaking:
        try:
            img_buf = render_story_card(a, scale=2)
            caption = format_story_caption(a)
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=img_buf,
                caption=caption,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning("Failed to send breaking card: %s", e)
        a.pop("_pil_image", None)
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
    top = matches[:5]

    lines = ["\U0001f50d *Results for: " + _escape_md(keyword) + "*\n"]
    for i, a in enumerate(top):
        num = str(i + 1)
        emoji = CATEGORY_EMOJI.get(a.get("category", "general"), "\U0001f4f0")
        title = a.get("title", "")
        source = a.get("source", "")
        lines.append(
            emoji + " *" + num + ".* " + _escape_md(title) + "\n"
            + _escape_md(source) + " | \u2b50 " + str(a.get("newsworthiness", 5)) + "/10\n"
            + "[Read](" + a.get("url", "") + ")\n"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


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
            "\nCurrent preferences:\n"
            + _format_prefs(get_user_prefs(update.effective_user.id)),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    param = context.args[0].lower()
    value = context.args[1]

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
    else:
        await update.message.reply_text("Unknown setting: " + param)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command list (64-char width for mobile)."""
    help_text = (
        "\U0001f30d *World News Bot \u2014 Commands*\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "/news     \u2014 Top 5 stories with AI\n"
        "            summaries\n"
        "/briefing \u2014 Visual news report\n"
        "            (infographic image)\n"
        "/breaking \u2014 High-impact stories\n"
        "            (newsworthiness 8+)\n"
        "/topic    \u2014 Search by keyword\n"
        "            Usage: /topic ukraine\n"
        "/deep     \u2014 Deep analysis of a\n"
        "            story from /news\n"
        "/ask      \u2014 Ask about the news\n"
        "            (AI-powered Q&A)\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "/categories \u2014 Set preferred topics\n"
        "/schedule   \u2014 Daily briefing time\n"
        "/sources    \u2014 View news sources\n"
        "/sentiment  \u2014 News mood gauge\n"
        "/set        \u2014 Adjust preferences\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "Powered by Claude AI + 10 sources"
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
    return (
        "Categories: " + cat_str + "\n"
        "Schedule: " + prefs.get("schedule", "07:00") + " IST\n"
        "Summary: " + prefs.get("summary_length", "medium")
    )


# ─── Scheduled Tasks ────────────────────────────────────────────────────────

async def scheduled_fetch(app):
    """Periodic news fetch."""
    try:
        count = await fetch_all_news()
        logger.info("Scheduled fetch completed: %d new articles", count)
    except Exception as e:
        logger.error("Scheduled fetch error: %s", e)


async def scheduled_briefing(app):
    """Send scheduled briefings as individual story cards."""
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

        # Fetch article images once for all users
        await fetch_article_images(top)

        # Pre-render all story cards once
        rendered_cards = []
        for a in top:
            try:
                img_buf = render_story_card(a, scale=2)
                caption = format_story_caption(a)
                rendered_cards.append((img_buf, caption))
            except Exception as e:
                logger.warning("Failed to render story card: %s", e)

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

                # Send each story card
                for img_buf, caption in rendered_cards:
                    img_buf.seek(0)
                    await app.bot.send_photo(
                        chat_id=chat_id,
                        photo=img_buf,
                        caption=caption,
                        parse_mode="Markdown",
                    )
                    await asyncio.sleep(0.3)

                # Footer
                await app.bot.send_message(
                    chat_id=chat_id,
                    text="\U0001f50e Use /topic <keyword> to dive deeper into any story",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception as e:
                logger.warning("Failed to send briefing to %s: %s", uid, e)

        # Cleanup pil images
        for a in top:
            a.pop("_pil_image", None)
    except Exception as e:
        logger.error("Scheduled briefing error: %s", e)


async def scheduled_cleanup():
    """Daily cache cleanup."""
    try:
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
        BotCommand("briefing", "Visual news report"),
        BotCommand("breaking", "High-impact stories"),
        BotCommand("topic", "Search by keyword"),
        BotCommand("deep", "Deep analysis"),
        BotCommand("ask", "Ask about the news"),
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
    application.add_handler(CommandHandler("deep", cmd_deep))
    application.add_handler(CommandHandler("ask", cmd_ask))
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
