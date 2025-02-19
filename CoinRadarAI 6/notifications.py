import random
from datetime import datetime, timezone

# Projede zaten var olduğunu varsaydığımız değişken ve fonksiyonları import ediyoruz.
from config import (
    ALLOWED_CHAT_IDS,
    daily_notification_data,
    INACTIVITY_THRESHOLD,
    user_last_active,
    t,
    logger
)
from utils import format_price

# Güncelleme bilgisi kontrolü için kullanılabilecek basit bir örnek.
# config.py içinde LATEST_VERSION = 1.1 gibi bir değer olduğunu varsayıyoruz.
# current_version parametresi ise main.py'de veya başka bir yerde set edilebilir.
def check_for_updates(current_version: float) -> bool:
    try:
        from config import LATEST_VERSION  # Örnek: LATEST_VERSION = 1.1
        if current_version < LATEST_VERSION:
            logger.info(f"Yeni bir versiyon mevcut: {LATEST_VERSION}. Güncelleme yapmanız önerilir.")
            return True
        return False
    except ImportError:
        logger.warning("config.py içinde LATEST_VERSION tanımlı değil.")
        return False

# Rastgele sürpriz mesajlar (isteğe göre çoğaltılabilir/değiştirilebilir).
SURPRISE_MESSAGES = [
    "Şans kapında, bu sinyali kaçırma!",
    "Bugün senin günün, harekete geç!",
    "Sürpriz bir kazanç seni bekliyor!",
    "Bu sinyal belki de günün fırsatı!",
    "Ekstra şans ve bol kazanç dileriz!"
]

def is_user_allowed(update) -> bool:
    """Kullanıcı yetkili mi?"""
    from config import ALLOWED_USERS  # Örnek: ALLOWED_USERS = ["myTelegramUsername", ...]
    return update.effective_user.username in ALLOWED_USERS

def check_daily_limit() -> bool:
    """
    Günlük bildirim limitini kontrol eder.
    Limit aşıldıysa False döner, değilse True.
    """
    today = datetime.now(timezone.utc).date()
    if daily_notification_data.get('date') != today:
        daily_notification_data['date'] = today
        daily_notification_data['count'] = 0

    if daily_notification_data['count'] >= 8:
        logger.info("Daily notification limit reached.")
        return False
    return True

def increment_daily_count():
    """Her başarılı gönderimde günlük bildirim sayısını 1 artırır."""
    daily_notification_data['count'] += 1

async def send_trade_notification(context, symbol, direction, entry_price, tp, sl, lang):
    """
    Trade sinyallerini gönderir.
    """
    if not check_daily_limit():
        return

    # Sürpriz mesaj seçimi
    surprise_line = random.choice(SURPRISE_MESSAGES)

    msg = (
        f"🪬 *{t('trade_signal', lang)}*\n\n"
        f"*{t('coin', lang)}*: {symbol}\n"
        f"*{t('direction', lang)}*: {direction.upper()}\n\n"
        f"*{t('entry', lang)}*: {format_price(entry_price)}\n"
        f"*{t('take_profit', lang)}*: {format_price(tp)}\n"
        f"*{t('stop_loss', lang)}*: {format_price(sl)}\n\n"
        f"_{surprise_line}_"  # Sürpriz cümleyi ekliyoruz
    )

    # Tüm yetkili chat_id'lere gönder
    for chat_id in ALLOWED_CHAT_IDS:
        last_active = user_last_active.get(chat_id)
        # Kullanıcı uzun süredir pasifse veya hiç kaydı yoksa gönder
        if last_active is None or (datetime.now(timezone.utc) - last_active) > INACTIVITY_THRESHOLD:
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending trade notification to {chat_id}: {e}")

    increment_daily_count()

async def send_reversal_notification(context, symbol, old_direction, new_direction, price, lang):
    """
    Fiyatın veya trendin yön değiştirdiğini (reversal) bildiren sinyalleri gönderir.
    Örneğin: LONG'dan SHORT'a, SHORT'tan LONG'a geçiş.
    """
    if not check_daily_limit():
        return

    surprise_line = random.choice(SURPRISE_MESSAGES)

    msg = (
        f"🔄 *{t('reversal_signal', lang)}*\n\n"
        f"*{t('coin', lang)}*: {symbol}\n"
        f"*{t('old_direction', lang)}*: {old_direction.upper()}\n"
        f"*{t('new_direction', lang)}*: {new_direction.upper()}\n\n"
        f"*{t('price', lang)}*: {format_price(price)}\n\n"
        f"_{surprise_line}_"
    )

    for chat_id in ALLOWED_CHAT_IDS:
        last_active = user_last_active.get(chat_id)
        if last_active is None or (datetime.now(timezone.utc) - last_active) > INACTIVITY_THRESHOLD:
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending reversal notification to {chat_id}: {e}")

    increment_daily_count()

def map_term_to_interval(term):
    """
    Kısa/Orta/Uzun vade gibi terimleri süreye dönüştürür.
    short -> 1h, medium -> 4h, aksi -> 1d
    """
    term = term.lower()
    if term == "short":
        return "1h"
    elif term == "medium":
        return "4h"
    else:
        return "1d"
