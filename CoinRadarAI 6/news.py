import aiohttp
from datetime import datetime
from urllib.parse import quote
from config import NEWS_API_KEY, t, logger

async def get_market_news(lang) -> list:
    """
    Haberleri getirir ve her haber için:
      {
        "title": "...",
        "link": "...",
        "published": "...",
        "image_url": "...",  # Eğer yoksa None
      }
    şeklinde bir liste döndürür.
    """
    if lang == "tr":
        query = quote('ekonomi OR finans OR "iş dünyası"')
        url = f"https://newsapi.org/v2/everything?q={query}&language=tr&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
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
                            image_url = art.get("urlToImage")  # NewsAPI'de resim linki
                            # Yayınlanma tarihi formatlaması
                            try:
                                dt = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
                                pub_on = dt.strftime("%Y-%m-%d %H:%M")
                            except Exception:
                                pub_on = pub_at

                            news_list.append({
                                "title": title,
                                "link": link,
                                "published": pub_on,
                                "image_url": image_url if image_url else None
                            })
                        return news_list
                    else:
                        logger.error(f"Error fetching Turkish news: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Error in Turkish news: {e}")
            return []
    else:
        # CryptoCompare API
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
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
                            # Bazı item'larda 'imageurl' alanı boş veya None olabilir
                            raw_image_url = item.get("imageurl", None)
                            # CryptoCompare'da resim linki tam olmayabiliyor, başına domain ekliyoruz
                            if raw_image_url and raw_image_url.startswith("http"):
                                image_url = raw_image_url
                            elif raw_image_url:
                                image_url = "https://www.cryptocompare.com" + raw_image_url
                            else:
                                image_url = None

                            pub_ts = item.get("published_on", 0)
                            pub_on = datetime.utcfromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")

                            news_list.append({
                                "title": title,
                                "link": link,
                                "published": pub_on,
                                "image_url": image_url
                            })
                        return news_list
                    else:
                        logger.error(f"Error fetching English news: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Error in English news: {e}")
            return []

async def news_command(update, context) -> None:
    """
    Kullanıcıya en son haberleri daha şık ve mümkünse resimli bir şekilde gönderir.
    """
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
    if not news_items:
        await update.message.reply_text(t("no_news", lang))
        return

    # Her haber için ayrı bir mesaj gönderiyoruz.
    # Resim varsa "send_photo", yoksa normal text mesajı.
    pub_text = "Yayınlandı:" if lang == "tr" else "Published:"
    read_more = "Devamını oku" if lang == "tr" else "Read more"

    for article in news_items:
        title = article["title"]
        link = article["link"]
        published = article["published"]
        image_url = article["image_url"]

        # Telegram mesajı için "caption" oluşturuyoruz
        caption = (
            f"*{title}*\n"
            f"{pub_text} {published}\n"
            f"[{read_more}]({link})"
        )

        try:
            if image_url:
                # Resimli mesaj gönder
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=image_url,
                    caption=caption,
                    parse_mode='Markdown'
                )
            else:
                # Sadece metin mesajı gönder
                await update.message.reply_text(
                    caption,
                    parse_mode='Markdown',
                    disable_web_page_preview=False  # Link önizlemesi açık kalsın
                )
        except Exception as e:
            # Resim linki bozuk veya başka bir hata olabilir
            logger.error(f"Error sending news item with image: {e}")
            # Yedek olarak metin mesajı
            await update.message.reply_text(
                caption,
                parse_mode='Markdown',
                disable_web_page_preview=False
            )
