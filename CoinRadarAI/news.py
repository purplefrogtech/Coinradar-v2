import aiohttp
from datetime import datetime
from urllib.parse import quote
from config import NEWS_API_KEY, t, logger

async def get_market_news(lang) -> list:
    if lang == "tr":
        query = quote('ekonomi OR finans OR "iş dünyası"')
        url = f"https://newsapi.org/v2/everything?q={query}&language=tr&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        pub_text = "Yayınlandı:"
        read_more = "Devamını oku"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data.get("articles", [])
                        news_list = []
                        for art in articles[:5]:
                            title = art.get("title", "Başlık Yok")
                            link = art.get("url", "")
                            pub_at = art.get("publishedAt", "")
                            try:
                                dt = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
                                pub_on = dt.strftime("%Y-%m-%d %H:%M")
                            except Exception:
                                pub_on = pub_at
                            news_list.append(f"*{title}*\n{pub_text} {pub_on}\n[{read_more}]({link})")
                        return news_list
                    else:
                        logger.error(f"Error fetching Turkish news: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Error in Turkish news: {e}")
            return []
    else:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        pub_text = "Published:"
        read_more = "Read more"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("Data", [])
                        news_list = []
                        for item in items[:5]:
                            title = item.get("title", "No Title")
                            link = item.get("url", "")
                            pub_on = datetime.utcfromtimestamp(item.get("published_on", 0)).strftime("%Y-%m-%d %H:%M")
                            news_list.append(f"*{title}*\n{pub_text} {pub_on}\n[{read_more}]({link})")
                        return news_list
                    else:
                        logger.error(f"Error fetching English news: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Error in English news: {e}")
            return []

async def news_command(update, context) -> None:
    from config import user_language, ALLOWED_USERS
    from utils import update_user_activity
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if update.effective_user.username not in ALLOWED_USERS:
        await update.message.reply_text(t("no_permission", lang))
        return
    await update.message.reply_text(t("fetching_news", lang))
    news_items = await get_market_news(lang)
    if news_items:
        header = t("news_header", lang)
        text = header + "\n\n" + "\n\n".join(news_items)
        await update.message.reply_text(text, parse_mode='Markdown', disable_web_page_preview=True)
    else:
        await update.message.reply_text(t("no_news", lang))
