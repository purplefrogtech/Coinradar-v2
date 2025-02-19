from datetime import datetime, timezone
from config import ALLOWED_CHAT_IDS, daily_notification_data, INACTIVITY_THRESHOLD, user_last_active, t, logger

from utils import format_price

def is_user_allowed(update) -> bool:
    from config import ALLOWED_USERS
    return update.effective_user.username in ALLOWED_USERS

async def send_trade_notification(context, symbol, direction, entry_price, tp, sl, lang):
    today = datetime.now(timezone.utc).date()
    if daily_notification_data.get('date') != today:
        daily_notification_data['date'] = today
        daily_notification_data['count'] = 0
    if daily_notification_data['count'] >= 8:
        logger.info("Daily notification limit reached.")
        return
    msg = (f"🪬 *{t('trade_signal', lang)}*\n\n"
           f"*{t('coin', lang)}*: {symbol}\n"
           f"*{t('direction', lang)}*: {direction.upper()}\n\n"
           f"*{t('entry', lang)}*: {format_price(entry_price)}\n"
           f"*{t('take_profit', lang)}*: {format_price(tp)}\n"
           f"*{t('stop_loss', lang)}*: {format_price(sl)}\n")
    for chat_id in ALLOWED_CHAT_IDS:
        last = user_last_active.get(chat_id)
        if last is None or (datetime.now(timezone.utc) - last) > INACTIVITY_THRESHOLD:
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending notification to {chat_id}: {e}")
    daily_notification_data['count'] += 1

def map_term_to_interval(term):
    term = term.lower()
    if term == "short":
        return "1h"
    elif term == "medium":
        return "4h"
    else:
        return "1d"
