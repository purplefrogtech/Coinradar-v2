import os
import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from config import TOKEN
from telegram_handlers import (
    start, language_callback, set_language, coin, long_signals, sell_signals,
    trend, analysis_callback, realtime, chart, adv_analysis, set_favorites, get_favorites, set_risk, get_risk
)
from news import news_command

# Sohbet mesajlarına yanıt veren handler
async def chat_handler(update, context):
    # Sadece komut olmayan normal sohbet mesajlarına yanıt ver.
    user_text = update.message.text
    # Finans konularıyla ilgili basit bir sohbet yanıtı. (Gelişmiş bir sohbet sistemi entegre edilebilir.)
    reply = "Finans hakkında sorularınızı bekliyorum. Ancak lütfen komutları kullanarak işlem yapınız."
    await update.message.reply_text(reply)

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CallbackQueryHandler(language_callback, pattern="^lang_"))
    app.add_handler(CallbackQueryHandler(analysis_callback, pattern="^analysis:"))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("coin", coin))
    app.add_handler(CommandHandler("long", long_signals))
    app.add_handler(CommandHandler("short", sell_signals))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("setfavorites", set_favorites))
    app.add_handler(CommandHandler("getfavorites", get_favorites))
    app.add_handler(CommandHandler("setrisk", set_risk))
    app.add_handler(CommandHandler("getrisk", get_risk))
    app.add_handler(CommandHandler("realtime", realtime))
    app.add_handler(CommandHandler("adv_analysis", adv_analysis))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("lang", set_language))
    
    # Komut dışı sohbet mesajları için handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
    
    app.run_polling()

if __name__ == "__main__":
    main()
