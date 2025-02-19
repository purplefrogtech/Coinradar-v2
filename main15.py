import pandas as pd
import ta
import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Update, Message, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import io
from urllib.parse import quote

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

# Telegram bot token (kendi tokenınızı kullanın)
TOKEN = '7183903208:AAExk376WQTmjhi9k3qabDSDYijkEEEljDE'

# Haberler için NewsAPI anahtarı (kendi API anahtarınızı ekleyin)
NEWS_API_KEY = '58878f135c9c4d609f856aa552d2d12d'

# Logging ayarları
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Allowed users and chat IDs
ALLOWED_USERS = [
    'paraloperceo', 'LaunchControll', 'ensalgz', 'gorkemk6',
    'WOULTHERR', 'MacqTrulz', 'janexander', 'mmmmonur', 'Ern5716',
    'Lord1334', 'thebatyroff', 'M_Senol24'
]
ALLOWED_CHAT_IDS = [5124738136, 5633085280, 1332756927, 5140980618]

TARGET_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT'
]

daily_notification_data = {'date': None, 'count': 0}
user_last_active = {}
user_favorites = {}      
user_risk_settings = {}  
INACTIVITY_THRESHOLD = timedelta(minutes=10)

# Kullanıcı dil tercihleri (user_id -> "en" veya "tr")
user_language = {}

# Çeviri sözlüğü
translations = {
    "choose_language": {"en": "Please choose your language:", "tr": "Lütfen dilinizi seçiniz:"},
    "language_set_en": {"en": "Language set to English.", "tr": "Dil İngilizce olarak ayarlandı."},
    "language_set_tr": {"en": "Language set to Turkish.", "tr": "Dil Türkçe olarak ayarlandı."},
    "welcome_message": {
        "en": (
            "Welcome to CoinRadar AI! 🎉\n\n"
            "To receive a signal, use the command /coin <symbol> (e.g. /coin BTCUSDT).\n"
            "Or scan the market with:\n"
            "/long - for long signals\n"
            "/short - for short signals\n"
            "/trend - for general market trend analysis\n\n"
            "Additional commands:\n"
            "/chart <symbol> [timeframe] - Get a technical chart (e.g., /chart BTCUSDT 4h)\n"
            "/setfavorites <coin1,coin2,...> - Set your favorite coins\n"
            "/getfavorites - Show your favorite coins\n"
            "/setrisk <risk_percentage> - Set your risk percentage\n"
            "/getrisk - Show your risk setting\n"
            "/realtime <symbol> - Get real-time price updates\n"
            "/adv_analysis <symbol> - Advanced technical analysis\n"
            "/news - Get latest market news"
        ),
        "tr": (
            "CoinRadar AI'ya hoşgeldiniz! 🎉\n\n"
            "Sinyal almak için /coin <sembol> komutunu kullanın (örn. /coin BTCUSDT).\n"
            "Ya da piyasayı taramak için:\n"
            "/long - uzun sinyaller\n"
            "/short - kısa sinyaller\n"
            "/trend - genel piyasa trend analizi\n\n"
            "Ek komutlar:\n"
            "/chart <sembol> [zaman dilimi] - Teknik grafik (örn. /chart BTCUSDT 4h)\n"
            "/setfavorites <coin1,coin2,...> - Favori coinlerinizi ayarlayın\n"
            "/getfavorites - Favori coinlerinizi görüntüleyin\n"
            "/setrisk <risk_yüzdesi> - Risk ayarınızı belirleyin\n"
            "/getrisk - Risk ayarınızı görüntüleyin\n"
            "/realtime <sembol> - Gerçek zamanlı fiyat güncellemeleri\n"
            "/adv_analysis <sembol> - Gelişmiş teknik analiz\n"
            "/news - En güncel piyasa haberleri"
        )
    },
    "news_header": {"en": "Latest Market News:", "tr": "En Güncel Piyasa Haberleri:"},
    "no_permission": {"en": "You don't have access permission for the AI trader.", "tr": "AI trader'a erişim izniniz yok."},
    "join_community": {"en": "Join Global Community: @coinradarsinyal", "tr": "Global Community'e katılın: @coinradarsinyal"},
    "trend_analysis_wait": {"en": "Performing trend analysis, please wait...", "tr": "Trend analizi yapılıyor, lütfen bekleyiniz..."},
    "chart_wait": {"en": "Generating chart, please wait...", "tr": "Grafik oluşturuluyor, lütfen bekleyiniz..."},
    "chart_usage": {"en": "Usage: /chart <symbol> [timeframe]. Example: /chart BTCUSDT 4h", "tr": "Kullanım: /chart <sembol> [zaman dilimi]. Örnek: /chart BTCUSDT 4h"},
    "chart_error": {"en": "Chart could not be generated. Please check coin symbol and timeframe.", "tr": "Grafik oluşturulamadı. Lütfen coin sembolünü ve zaman dilimini kontrol ediniz."},
    "setfavorites_usage": {"en": "Usage: /setfavorites BTCUSDT,ETHUSDT,...", "tr": "Kullanım: /setfavorites BTCUSDT,ETHUSDT,..."},
    "favorites_set": {"en": "Favorites set: ", "tr": "Favoriler ayarlandı: "},
    "no_favorites": {"en": "No favorites set.", "tr": "Favori ayarlanmamış."},
    "your_favorites": {"en": "Your favorites: ", "tr": "Favorileriniz: "},
    "setrisk_usage": {"en": "Usage: /setrisk <risk_percentage>", "tr": "Kullanım: /setrisk <risk_yüzdesi>"},
    "risk_set": {"en": "Risk setting set to ", "tr": "Risk ayarı "},
    "invalid_risk": {"en": "Invalid risk value.", "tr": "Geçersiz risk değeri."},
    "no_risk": {"en": "No risk setting found.", "tr": "Risk ayarı bulunamadı."},
    "your_risk": {"en": "Your risk setting: ", "tr": "Risk ayarınız: "},
    "start_analysis": {"en": "Starting analysis...", "tr": "Analiz başlatılıyor..."},
    "specify_coin": {"en": "Please specify a coin, e.g. /coin BTCUSDT", "tr": "Lütfen bir coin belirtin, örn. /coin BTCUSDT"},
    "no_signal": {"en": "No Clear Signal at the moment", "tr": "Şu anda net bir sinyal yok"},
    "risk_management": {"en": "Risk Management:", "tr": "Risk Yönetimi:"},
    "realtime_usage": {"en": "Usage: /realtime <symbol>", "tr": "Kullanım: /realtime <sembol>"},
    "connecting_realtime": {"en": "Connecting to real-time data for ", "tr": "Gerçek zamanlı veriye bağlanılıyor: "},
    "error_realtime": {"en": "Error connecting to real-time data.", "tr": "Gerçek zamanlı veriye bağlanırken hata oluştu."},
    "adv_analysis_wait": {"en": "Performing advanced analysis for ", "tr": "Gelişmiş analiz yapılıyor: "},
    "adv_analysis_error": {"en": "Could not generate advanced analysis chart.", "tr": "Gelişmiş analiz grafiği oluşturulamadı."},
    "fetching_news": {"en": "Fetching latest market news...", "tr": "En güncel piyasa haberleri getiriliyor..."},
    "no_news": {"en": "Could not fetch news at this time.", "tr": "Şu anda haberler getirilemedi."},
    "trade_signal": {"en": "Trade Signal", "tr": "Ticaret Sinyali"},
    "coin": {"en": "Coin", "tr": "Coin"},
    "direction": {"en": "Direction", "tr": "Yön"},
    "entry": {"en": "Entry", "tr": "Giriş"},
    "take_profit": {"en": "Take Profit", "tr": "Kar Al"},
    "stop_loss": {"en": "Stop Loss", "tr": "Zarar Durdur"},
    "long": {"en": "LONG", "tr": "UZUN"},
    "short": {"en": "SHORT", "tr": "KISA"},
    "scanning_long": {"en": "Scanning market for long signals...", "tr": "Uzun sinyaller için piyasada tarama yapılıyor..."},
    "scanning_short": {"en": "Scanning market for short signals...", "tr": "Kısa sinyaller için piyasada tarama yapılıyor..."},
    "top_long_signals": {"en": "Top LONG Signals", "tr": "En İyi UZUN Sinyaller"},
    "top_short_signals": {"en": "Top SHORT Signals", "tr": "En İyi KISA Sinyaller"},
    "analyzing": {"en": "Analyzing...", "tr": "Analiz ediliyor..."},
    "signal_found": {"en": "Signal found!", "tr": "Sinyal bulundu!"}
}

def t(key: str, lang: str) -> str:
    return translations.get(key, {}).get(lang, "")

def format_price(price: float) -> str:
    if price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    elif price >= 0.0001:
        return f"{price:.8f}"
    else:
        return f"{price:.10f}"

def update_user_activity(update: Update) -> None:
    if update.effective_chat:
        user_last_active[update.effective_chat.id] = datetime.utcnow()

async def get_crypto_data(symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    try:
        url = f"{BINANCE_API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error fetching data: {response.status}, {await response.text()}")
                    return pd.DataFrame()
                data = await response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df.astype(float)
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_pivot_points(data: pd.DataFrame) -> dict:
    last_period = data.iloc[-2]
    pivot = (last_period['high'] + last_period['low'] + last_period['close']) / 3
    return {
        'pivot': pivot,
        'support1': (2 * pivot) - last_period['high'],
        'resistance1': (2 * pivot) - last_period['low'],
        'support2': pivot - (last_period['high'] - last_period['low']),
        'resistance2': pivot + (last_period['high'] - last_period['low'])
    }

def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    macd = ta.trend.MACD(close=data['close'], window_slow=12, window_fast=6, window_sign=3)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    return data

def calculate_atr(data: pd.DataFrame, period: int = 7) -> float:
    atr = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=period)
    return atr.average_true_range().iloc[-1]

def calculate_entry_price(data: pd.DataFrame) -> float:
    return data['open'].iloc[-1]

async def get_technical_indicators(symbol: str, timeframe: str = '1h') -> dict:
    try:
        data = await get_crypto_data(symbol, interval=timeframe, limit=100)
        if data.empty:
            raise ValueError("No data received.")
        pivots = calculate_pivot_points(data)
        data = calculate_macd(data)
        atr = calculate_atr(data, period=7)
        current_price = data['close'].iloc[-1]
        entry_price = calculate_entry_price(data)
        data['ema_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()
        data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
        return {
            'pivot_points': pivots,
            'current_price': current_price,
            'atr': atr,
            'entry_price': entry_price,
            'data': data
        }
    except Exception as e:
        logger.error(f"Error in technical indicators for {symbol}: {e}")
        return {}

def generate_signals(indicators: dict) -> dict:
    cp = indicators.get('current_price')
    pivots = indicators.get('pivot_points')
    atr = indicators.get('atr')
    data = indicators.get('data')
    ema_long = cp > data['ema_20'].iloc[-1]
    ema_short = cp < data['ema_20'].iloc[-1]
    obv_long = data['obv'].iloc[-1] > data['obv'].iloc[-2] if len(data) > 1 else False
    obv_short = data['obv'].iloc[-1] < data['obv'].iloc[-2] if len(data) > 1 else False
    rsi = data['rsi'].iloc[-1]
    rsi_long = rsi < 70
    rsi_short = rsi > 30
    buy_strength = (data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]) if (cp > pivots['pivot'] and ema_long and obv_long and rsi_long) else None
    sell_strength = (data['macd_signal'].iloc[-1] - data['macd'].iloc[-1]) if (cp < pivots['pivot'] and ema_short and obv_short and rsi_short) else None
    return {
        'buy_signal': buy_strength is not None,
        'sell_signal': sell_strength is not None,
        'buy_signal_strength': buy_strength,
        'sell_signal_strength': sell_strength,
        'tp_long': cp + (2 * atr),
        'sl_long': cp - (1 * atr),
        'tp_short': cp - (2 * atr),
        'sl_short': cp + (1 * atr)
    }

def is_user_allowed(update: Update) -> bool:
    return update.effective_user.username in ALLOWED_USERS

async def send_trade_notification(context: ContextTypes.DEFAULT_TYPE, symbol: str, direction: str,
                                  entry_price: float, tp: float, sl: float, lang: str) -> None:
    global daily_notification_data
    today = datetime.utcnow().date()
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
        last_active = user_last_active.get(chat_id)
        if last_active is None or (datetime.utcnow() - last_active) > INACTIVITY_THRESHOLD:
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending notification to {chat_id}: {e}")
    daily_notification_data['count'] += 1

# Vade haritalaması: short -> 1h, medium -> 4h, long -> 1d
def map_term_to_interval(term: str) -> str:
    term = term.lower()
    if term == "short":
        return "1h"
    elif term == "medium":
        return "4h"
    else:
        return "1d"

# --- ANALYSIS FUNCTIONS (vade seçimine göre) ---

async def coin_analysis_by_term(symbol: str, term: str, lang: str, username: str) -> (str, io.BytesIO):
    tf = map_term_to_interval(term)
    indicators = await get_technical_indicators(symbol, tf)
    if not indicators:
        return (f"⚠️ {symbol} - lütfen sembolü kontrol edin ya da daha sonra tekrar deneyin.", None)
    signals = generate_signals(indicators)
    msg = f"🪬 *{symbol} ({term.capitalize()} Term)*\n\n"
    if signals['buy_signal']:
        msg += (f"🚀 *{t('direction', lang)}*: {t('long', lang)}\n\n"
                f"*{t('entry', lang)}*: {format_price(indicators['entry_price'])}\n"
                f"*{t('take_profit', lang)}*: {format_price(signals['tp_long'])}\n"
                f"*{t('stop_loss', lang)}*: {format_price(signals['sl_long'])}\n")
        direction = t("long", lang)
    elif signals['sell_signal']:
        msg += (f"🩸 *{t('direction', lang)}*: {t('short', lang)}\n\n"
                f"*{t('entry', lang)}*: {format_price(indicators['entry_price'])}\n"
                f"*{t('take_profit', lang)}*: {format_price(signals['tp_short'])}\n"
                f"*{t('stop_loss', lang)}*: {format_price(signals['sl_short'])}\n")
        direction = t("short", lang)
    else:
        msg += f"🤚🏼 *{t('no_signal', lang)}*\n"
        direction = None
    risk = user_risk_settings.get(username)
    if risk is not None and direction is not None:
        if direction == t("long", lang):
            rd = indicators['entry_price'] - signals['sl_long']
        else:
            rd = signals['sl_short'] - indicators['entry_price']
        if rd > 0:
            lev = round((indicators['entry_price'] * (risk / 100)) / rd, 1)
            msg += (f"\n*{t('risk_management', lang)}*\n"
                    f"{t('your_risk', lang)} {risk}%\n"
                    f"Recommended Leverage: {lev}x\n"
                    f"Allocate {risk}% of your capital as margin.")
    chart = await generate_chart(symbol, tf)
    return msg, chart

async def long_signals_by_term(term: str, lang: str) -> str:
    tf = map_term_to_interval(term)
    sigs = []
    tasks = [get_technical_indicators(s, tf) for s in TARGET_COINS]
    results = await asyncio.gather(*tasks)
    for sym, indicators in zip(TARGET_COINS, results):
        if indicators:
            s = generate_signals(indicators)
            if s['buy_signal']:
                sigs.append((sym, s['buy_signal_strength'], s, indicators['entry_price']))
    if sigs:
        sigs = sorted(sigs, key=lambda x: x[1], reverse=True)[:7]
        msg = f"🚀 *{t('top_long_signals', lang)} ({term.capitalize()} Term)*\n\n"
        for sym, strength, s, ep in sigs:
            msg += (f"🪬*{sym}*\n\n"
                    f"   {t('direction', lang)}: {t('long', lang)}\n"
                    f"   {t('entry', lang)}: {format_price(ep)}\n"
                    f"   {t('take_profit', lang)}: {format_price(s['tp_long'])}\n"
                    f"   {t('stop_loss', lang)}: {format_price(s['sl_long'])}\n\n")
        return msg
    else:
        return "No strong long signals found at this time."

async def short_signals_by_term(term: str, lang: str) -> str:
    tf = map_term_to_interval(term)
    sigs = []
    tasks = [get_technical_indicators(s, tf) for s in TARGET_COINS]
    results = await asyncio.gather(*tasks)
    for sym, indicators in zip(TARGET_COINS, results):
        if indicators:
            s = generate_signals(indicators)
            if s['sell_signal']:
                sigs.append((sym, s['sell_signal_strength'], s, indicators['entry_price']))
    if sigs:
        sigs = sorted(sigs, key=lambda x: x[1], reverse=True)[:7]
        msg = f"🩸 *{t('top_short_signals', lang)} ({term.capitalize()} Term)*\n\n"
        for sym, strength, s, ep in sigs:
            msg += (f"🪬*{sym}*\n\n"
                    f"   {t('direction', lang)}: {t('short', lang)}\n"
                    f"   {t('entry', lang)}: {format_price(ep)}\n"
                    f"   {t('take_profit', lang)}: {format_price(s['tp_short'])}\n"
                    f"   {t('stop_loss', lang)}: {format_price(s['sl_short'])}\n\n")
        return msg
    else:
        return "No strong short signals found at this time."

async def trend_analysis_by_term(term: str, lang: str) -> (str, io.BytesIO):
    tf = map_term_to_interval(term)
    fng = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/?limit=1") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fng = int(data["data"][0]["value"])
                else:
                    logger.error(f"FNG fetch error: {resp.status}")
    except Exception as e:
        logger.error(f"Error fetching FNG: {e}")
    btc_ema_20 = btc_ema_50 = None
    try:
        btc_data = await get_crypto_data("BTCUSDT", interval=tf, limit=60)
        if not btc_data.empty:
            closes = btc_data["close"].tolist()
            if len(closes) >= 50:
                btc_ema_20 = sum(closes[-20:]) / 20
                btc_ema_50 = sum(closes[-50:]) / 50
    except Exception as e:
        logger.error(f"BTC EMA error: {e}")
    if fng is not None and btc_ema_20 is not None and btc_ema_50 is not None:
        if fng > 60 and btc_ema_20 > btc_ema_50:
            trend = "TREND Long 🚀"
        elif fng < 40 and btc_ema_20 < btc_ema_50:
            trend = "TREND Short 🩸"
        else:
            trend = "TREND Nötr 🤚🏼"
    else:
        trend = "Trend analizi için gerekli veriler alınamadı."
    chart = await generate_trend_chart_custom(tf)
    return trend, chart

async def generate_trend_chart_custom(interval: str) -> io.BytesIO:
    data = await get_crypto_data("BTCUSDT", interval=interval, limit=60)
    if data.empty:
        return None
    data['ema_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
    data['ema_50'] = ta.trend.EMAIndicator(close=data['close'], window=50).ema_indicator()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['close'], label="BTCUSDT Close", color="blue")
    ax.plot(data.index, data['ema_20'], label="EMA 20", color="orange")
    ax.plot(data.index, data['ema_50'], label="EMA 50", color="green")
    ax.set_title("BTCUSDT Trend Analizi")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat")
    ax.legend(loc='upper left')
    ax.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# --- CHART GENERATION ---
async def generate_chart(symbol: str, timeframe: str = "1h") -> io.BytesIO:
    data = await get_crypto_data(symbol, interval=timeframe, limit=100)
    if data.empty:
        return None
    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()
    data['bb_middle'] = bb.bollinger_mavg()
    fib_data = data[-60:] if len(data) >= 60 else data
    high = fib_data['high'].max()
    low = fib_data['low'].min()
    diff = high - low
    fib_levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100.0%': low
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(data.index, data['close'], label="Close", color="blue")
    ax1.plot(data.index, data['bb_upper'], label="BB Upper", color="red", linestyle="--")
    ax1.plot(data.index, data['bb_lower'], label="BB Lower", color="green", linestyle="--")
    ax1.plot(data.index, data['bb_middle'], label="BB Middle", color="orange", linestyle="--")
    for lvl, val in fib_levels.items():
        ax1.axhline(val, linestyle="--", label=f"Fib {lvl}")
    ax1.set_title(f"{symbol} Price Chart ({timeframe})")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
    ax2.plot(data.index, data['rsi'], label="RSI", color="purple")
    ax2.axhline(70, linestyle="--", color="red")
    ax2.axhline(30, linestyle="--", color="green")
    ax2.set_title("RSI")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# --- NEWS ---
async def get_market_news(lang: str) -> list:
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

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
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

# --- COIN NAME NORMALIZATION ---
def normalize_coin_name(name: str) -> str:
    mapping = {
        "BTC": "BTCUSDT", "BITCOIN": "BTCUSDT",
        "ETH": "ETHUSDT", "ETHEREUM": "ETHUSDT",
        "BNB": "BNBUSDT", "BINANCE": "BNBUSDT"
        # Gerekirse diğer coinleri ekleyin
    }
    key = name.strip().upper()
    return mapping.get(key, key)

# --- COMMANDS WITH TERM SELECTION ---
# /start: dil seçimi ve hoşgeldiniz mesajı
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

# Callback for language selection
async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

# /coin command: kullanıcı istediği coin adını (BTC, bitcoin, vb.) yazabilsin, ardından vade seçimi yapsın
async def coin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

# /long command: vade seçimi ile uzun sinyaller
async def long_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

# /short command: vade seçimi ile kısa sinyaller
async def sell_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

# /trend command: vade seçimi ile trend analizi
async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

# Callback handler for analysis selections – final output tek mesaj olarak gönderilir.
async def analysis_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    await query.message.delete()  # Inline mesajı sil
    if final_photo:
        await context.bot.send_photo(chat_id=query.message.chat_id, photo=final_photo, caption=final_msg, parse_mode="Markdown")
    else:
        await context.bot.send_message(chat_id=query.message.chat_id, text=final_msg, parse_mode="Markdown")

# --- REALTIME, CHART, ADV_ANALYSIS, NEWS, FAVORITES, RISK ---
async def realtime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                count = 0
                start = datetime.utcnow()
                while count < 10 and (datetime.utcnow() - start).total_seconds() < 30:
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

async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def adv_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    update_user_activity(update)
    user_id = update.effective_user.id
    lang = user_language.get(user_id, "en")
    if not is_user_allowed(update):
        await update.message.reply_text(t("no_permission", lang))
        return
    symbol = normalize_coin_name(context.args[0]) if context.args else "BTCUSDT"
    await update.message.reply_text(f"{t('adv_analysis_wait', lang)}{symbol}...")
    img = await generate_adv_chart(symbol, "1h")
    if img:
        await update.message.reply_photo(photo=img, caption=f"Advanced Technical Analysis for {symbol}", parse_mode='Markdown')
    else:
        await update.message.reply_text(t("adv_analysis_error", lang))

async def generate_adv_chart(symbol: str, timeframe: str = "1h") -> io.BytesIO:
    data = await get_crypto_data(symbol, interval=timeframe, limit=100)
    if data.empty:
        return None
    try:
        ichimoku = ta.trend.IchimokuIndicator(high=data['high'], low=data['low'], window1=9, window2=26, window3=52)
        data['ichimoku_a'] = ichimoku.ichimoku_a()
        data['ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception as e:
        logger.error(f"Ichimoku error: {e}")
    try:
        stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14, smooth_window=3)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
    except Exception as e:
        logger.error(f"Stochastic error: {e}")
    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax1.plot(data.index, data['close'], label="Close", color="blue")
    if 'ichimoku_a' in data.columns and 'ichimoku_b' in data.columns:
        ax1.plot(data.index, data['ichimoku_a'], label="Ichimoku A", color="green", linestyle="--")
        ax1.plot(data.index, data['ichimoku_b'], label="Ichimoku B", color="red", linestyle="--")
        ax1.fill_between(data.index, data['ichimoku_a'], data['ichimoku_b'], color='gray', alpha=0.3)
    ax1.set_title(f"{symbol} Price with Ichimoku Cloud ({timeframe})")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2.plot(data.index, data['stoch_k'], label="Stoch %K", color="blue")
    ax2.plot(data.index, data['stoch_d'], label="Stoch %D", color="orange")
    ax2.axhline(80, linestyle="--", color="red")
    ax2.axhline(20, linestyle="--", color="green")
    ax2.set_title("Stochastic Oscillator")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax3.plot(data.index, data['rsi'], label="RSI", color="purple")
    ax3.axhline(70, linestyle="--", color="red")
    ax3.axhline(30, linestyle="--", color="green")
    ax3.set_title("RSI")
    ax3.legend(loc='upper left')
    ax3.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

async def set_favorites(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def get_favorites(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def get_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

def main() -> None:
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
    app.add_handler(CommandHandler("news", news))
    app.run_polling()

if __name__ == "__main__":
    main()
