# World News Bot

A Telegram bot that aggregates world news from 10+ global sources, generates AI-powered summaries with Claude, creates visual infographic reports, and supports interactive Q&A.

## Features

- **Multi-source aggregation**: Reuters, BBC, Al Jazeera, CNN, NPR, The Guardian, AP News, France24, DW News, ABC News + optional GNews API
- **AI-powered summaries**: Claude Haiku for fast summarization, Claude Sonnet for deep analysis
- **Visual briefings**: Dark-themed infographic cards generated with Pillow
- **Breaking news alerts**: Auto-detect high-impact stories (newsworthiness 8+)
- **Conversational Q&A**: Ask questions about current news with context-aware AI responses
- **Smart categorization**: 7 categories (geopolitics, economy, technology, climate, conflict, science, markets)
- **Scheduled briefings**: Configurable daily briefing times (default 7AM/6PM IST)
- **Deduplication**: SHA-256 URL hashing + fuzzy title matching

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message + setup |
| `/news` | Top 5 stories with AI summaries |
| `/briefing` | Visual infographic report |
| `/breaking` | High-impact stories (8+/10) |
| `/topic [keyword]` | Search by keyword |
| `/deep [1-5]` | Deep analysis of a story |
| `/ask [question]` | AI-powered news Q&A |
| `/categories` | Set preferred topics |
| `/schedule [HH:MM]` | Daily briefing time (IST) |
| `/sources` | View news source status |
| `/sentiment` | News mood gauge |
| `/set [param] [value]` | Adjust preferences |
| `/help` | Command list |

## Deployment (Railway)

1. Create a new Railway project
2. Connect this GitHub repo
3. Add environment variables:
   - `TELEGRAM_TOKEN` — Telegram bot token from @BotFather
   - `ANTHROPIC_API_KEY` — Claude API key
   - `GNEWS_API_KEY` — (optional) GNews API key
   - `DATA_DIR` — `/data` (attach a Railway volume)
4. Deploy — nixpacks handles the rest

## Tech Stack

- Python 3.11
- python-telegram-bot v20+ (async)
- Anthropic Claude API (Haiku + Sonnet)
- Pillow (visual reports)
- feedparser (RSS)
- aiohttp (async HTTP + health check)
- APScheduler (scheduled briefings)

## Architecture

Single-file (`bot.py`) for simplicity. Data persisted to `DATA_DIR`:
- `news_cache.jsonl` — article cache (auto-trimmed to 14 days)
- `user_prefs.json` — per-user preferences
- `sent_articles.json` — dedup tracker
- `conversation_history.json` — Q&A context
- `bot_state.json` — general state
