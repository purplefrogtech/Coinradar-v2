import os
import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from config import TOKEN
from telegram_handlers import (
    start, language_callback, set_language, coin, long_signals, sell_signals,
    trend, analysis_callback, realtime, chart, adv_analysis, set_favorites, get_favorites, set_risk, get_risk
)
from news import news_command

# Logging yapılandırması
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Komut dışı sohbet mesajlarına yanıt veren handler
async def chat_handler(update, context):
    message = update.effective_message
    if not message or not message.text:
        return
    user_text = message.text
    # Basit sohbet yanıtı; istenirse daha gelişmiş bir sohbet sistemi entegre edilebilir.
    reply = "Finans hakkında sorularınızı bekliyorum. Ancak lütfen komutları kullanarak işlem yapınız."
    await message.reply_text(reply)

# Hata yakalayıcı handler (hem loglama hem de opsiyonel kullanıcı bilgilendirmesi için)
async def error_handler(update, context):
    logger.error("Güncelleme işlenirken hata oluştu:", exc_info=context.error)
    # İsteğe bağlı: Kullanıcıya hata mesajı gönderilebilir.
    # if update and update.effective_message:
    #     await update.effective_message.reply_text("Beklenmedik bir hata oluştu. Lütfen daha sonra tekrar deneyiniz.")

def main():
    if not TOKEN:
        logger.error("TOKEN bulunamadı! Lütfen config dosyanızda TOKEN bilgisini ayarlayınız.")
        return

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
    
    # Hata handler'ını ekliyoruz
    app.add_error_handler(error_handler)
    
    logger.info("Bot başlatılıyor...")
    app.run_polling()

if __name__ == "__main__":
    main()
