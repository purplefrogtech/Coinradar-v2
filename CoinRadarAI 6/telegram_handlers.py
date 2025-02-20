from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler, CommandHandler
from config import user_language, t, logger
from utils import update_user_activity, normalize_coin_name
from analysis import coin_analysis_by_term, long_signals_by_term, short_signals_by_term, trend_analysis_by_term
from charts import generate_chart, generate_adv_chart
from data_fetch import get_technical_indicators
from notifications import is_user_allowed
from ai_analysis import interpret_chart

# Başlangıç komutu
async def start(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    if user_id not in user_language:
        keyboard = [
            [InlineKeyboardButton("English", callback_data="lang_en")],
            [InlineKeyboardButton("Türkçe", callback_data="lang_tr")]
        ]
        markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(t("choose_language", "en"), reply_markup=markup)
        return
    lang = user_language[user_id]
    if is_user_allowed(update):
        await update.message.reply_text(t("welcome_message", lang))
    else:
        await update.message.reply_text(f"{t('no_permission', lang)}\n{t('join_community', lang)}")

# Dil seçimi için callback
async def language_callback(update: Update, context):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    if query.data == "lang_en":
        user_language[user_id] = "en"
        msg = t("language_set_en", "en")
    elif query.data == "lang_tr":
        user_language[user_id] = "tr"
        msg = t("language_set_tr", "tr")
    else:
        msg = "Error: Unknown language selection."
    await query.edit_message_text(msg)

# /lang komutu
async def set_language(update: Update, context):
    update_user_activity(update)
    if context.args:
        new_lang = context.args[0].lower()
        if new_lang in ["en", "tr"]:
            user_language[update.effective_user.id] = new_lang
            await update.message.reply_text(f"Language set to {new_lang}.")
        else:
            await update.message.reply_text("Invalid language code. Use 'en' or 'tr'.")
    else:
        await update.message.reply_text("Usage: /lang <en/tr>")

# /coin komutu
async def coin(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(f"{t('no_permission', lang)}\n{t('join_community', lang)}")
        return
    if not context.args:
        await update.message.reply_text(t("specify_coin", lang))
        return
    coin_input = " ".join(context.args)
    symbol = normalize_coin_name(coin_input)
    keyboard = [
        [InlineKeyboardButton("Short Term", callback_data=f"analysis:coin:{symbol}:short")],
        [InlineKeyboardButton("Medium Term", callback_data=f"analysis:coin:{symbol}:medium")],
        [InlineKeyboardButton("Long Term", callback_data=f"analysis:coin:{symbol}:long")]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select term for analysis:", reply_markup=markup)

# /long komutu
async def long_signals(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(f"{t('no_permission', lang)}\n{t('join_community', lang)}")
        return
    keyboard = [
        [InlineKeyboardButton("Short Term", callback_data="analysis:long::short")],
        [InlineKeyboardButton("Medium Term", callback_data="analysis:long::medium")],
        [InlineKeyboardButton("Long Term", callback_data="analysis:long::long")]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select term for long signals analysis:", reply_markup=markup)

# /short komutu
async def sell_signals(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(f"{t('no_permission', lang)}\n{t('join_community', lang)}")
        return
    keyboard = [
        [InlineKeyboardButton("Short Term", callback_data="analysis:short::short")],
        [InlineKeyboardButton("Medium Term", callback_data="analysis:short::medium")],
        [InlineKeyboardButton("Long Term", callback_data="analysis:short::long")]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select term for short signals analysis:", reply_markup=markup)

# /trend komutu
async def trend(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(f"{t('no_permission', lang)}\n{t('join_community', lang)}")
        return
    keyboard = [
        [InlineKeyboardButton("Short Term", callback_data="analysis:trend::short")],
        [InlineKeyboardButton("Medium Term", callback_data="analysis:trend::medium")],
        [InlineKeyboardButton("Long Term", callback_data="analysis:trend::long")]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select term for trend analysis:", reply_markup=markup)

# Callback query ile analiz komutlarını işleme
async def analysis_callback(update: Update, context):
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":")
    if len(parts) != 4:
        await query.edit_message_text("Invalid callback data.")
        return
    analysis_type = parts[1]
    symbol = parts[2]  # coin komutunda sembol; boş olabilir
    term = parts[3]
    lang = user_language.get(query.from_user.id, "en")
    final_msg = ""
    final_photo = None
    if analysis_type == "coin":
        final_msg, final_photo = await coin_analysis_by_term(symbol, term, lang, query.from_user.username)
    elif analysis_type == "long":
        final_msg = await long_signals_by_term(term, lang)
    elif analysis_type == "short":
        final_msg = await short_signals_by_term(term, lang)
    elif analysis_type == "trend":
        final_msg, final_photo = await trend_analysis_by_term(term, lang)
    else:
        final_msg = "Unknown analysis type."
    await query.message.delete()
    if final_photo:
        await context.bot.send_photo(chat_id=query.message.chat_id, photo=final_photo, caption=final_msg, parse_mode="Markdown")
    else:
        await context.bot.send_message(chat_id=query.message.chat_id, text=final_msg, parse_mode="Markdown")

# /realtime komutu
async def realtime(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    if not context.args:
        await update.message.reply_text(t("realtime_usage", lang))
        return
    symbol = normalize_coin_name(context.args[0])
    await update.message.reply_text(f"{t('connecting_realtime', lang)}{symbol}...")
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    import aiohttp
    from datetime import datetime, timezone
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                count = 0
                start = datetime.now(timezone.utc)
                while count < 10 and (datetime.now(timezone.utc) - start).total_seconds() < 30:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        price = float(data.get('p', 0))
                        tms = data.get('T')
                        trade_time = datetime.utcfromtimestamp(tms/1000).strftime("%H:%M:%S")
                        await update.message.reply_text(f"Real-time update for {symbol}:\nPrice: {price}\nTime: {trade_time}")
                        count += 1
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    except Exception as e:
        logger.error(f"Realtime error: {e}")
        await update.message.reply_text(t("error_realtime", lang))

# /chart komutu
async def chart(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    if not context.args:
        await update.message.reply_text(t("chart_usage", lang))
        return
    symbol = normalize_coin_name(context.args[0])
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    await update.message.reply_text(t("chart_wait", lang))
    img = await generate_chart(symbol, timeframe)
    if img:
        await update.message.reply_photo(photo=img)
    else:
        await update.message.reply_text(t("chart_error", lang))

# /adv_analysis komutu
async def adv_analysis(update: Update, context):
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    symbol = normalize_coin_name(" ".join(context.args)) if context.args else "BTCUSDT"
    await update.message.reply_text(f"{t('adv_analysis_wait', lang)}{symbol}...")
    indicators = await get_technical_indicators(symbol, "1h")
    chart_img = await generate_adv_chart(symbol, "1h")
    ai_comment = ""
    final_caption = f"Advanced Technical Analysis for {symbol}\n\n{ai_comment}"
    if chart_img:
        await update.message.reply_photo(photo=chart_img, caption=final_caption, parse_mode='Markdown')
    else:
        await update.message.reply_text(final_caption)

# /setfavorites komutu
async def set_favorites(update: Update, context):
    from config import user_favorites
    update_user_activity(update)
    user = update.effective_user
    lang = user_language.get(user.id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    if not context.args:
        await update.message.reply_text(t("setfavorites_usage", lang))
        return
    favs = [fav.strip().upper() for fav in " ".join(context.args).split(',')]
    user_favorites[user.username] = favs
    await update.message.reply_text(t("favorites_set", lang) + ", ".join(favs))

# /getfavorites komutu
async def get_favorites(update: Update, context):
    from config import user_favorites
    update_user_activity(update)
    user = update.effective_user
    lang = user_language.get(user.id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    favs = user_favorites.get(user.username, [])
    if not favs:
        await update.message.reply_text(t("no_favorites", lang))
    else:
        await update.message.reply_text(t("your_favorites", lang) + ", ".join(favs))

# Güncellenmiş /setrisk komutu
async def set_risk(update: Update, context):
    from config import user_risk_settings
    update_user_activity(update)
    user = update.effective_user
    lang = user_language.get(user.id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    if not context.args:
        await update.message.reply_text(t("setrisk_usage", lang))
        return
    try:
        risk = float(context.args[0])
        user_risk_settings[user.username] = risk
        await update.message.reply_text(f"{t('risk_set', lang)}{risk}%")
    except Exception:
        await update.message.reply_text(t("invalid_risk", lang))

# Güncellenmiş /getrisk komutu
async def get_risk(update: Update, context):
    from config import user_risk_settings
    update_user_activity(update)
    user = update.effective_user
    lang = user_language.get(user.id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    risk = user_risk_settings.get(user.username)
    if risk is None:
        await update.message.reply_text(t("no_risk", lang))
    else:
        await update.message.reply_text(f"{t('your_risk', lang)}{risk}%")
