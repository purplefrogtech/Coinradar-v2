from datetime import datetime, timezone
from config import user_last_active

def format_price(price):
    if price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    elif price >= 0.0001:
        return f"{price:.8f}"
    else:
        return f"{price:.10f}"

def update_user_activity(update):
    if update.effective_chat:
        user_last_active[update.effective_chat.id] = datetime.now(timezone.utc)

def normalize_coin_name(name: str) -> str:
    mapping = {
        "BTC": "BTCUSDT", "BITCOIN": "BTCUSDT",
        "ETH": "ETHUSDT", "ETHEREUM": "ETHUSDT",
        "BNB": "BNBUSDT", "BINANCE": "BNBUSDT"
    }
    key = name.strip().upper()
    return mapping.get(key, key)
