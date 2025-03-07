import pandas as pd
import ta
import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Update, Message
from telegram.ext import Application, CommandHandler, ContextTypes
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import io

# Makine öğrenimi için scikit-learn
from sklearn.linear_model import LinearRegression

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

# Telegram bot token (kendi tokenınızı kullanın)
TOKEN = '7183903208:AAExk376WQTmjhi9k3qabDSDYijkEEEljDE'

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# List of allowed usernames (for /start access) and allowed chat IDs (for notifications)
ALLOWED_USERS = [
    'paraloperceo', 'LaunchControll', 'ensalgz', 'gorkemk6',
    'WOULTHERR', 'MacqTrulz', 'janexander', 'mmmmonur', 'Ern5716',
    'Lord1334', 'thebatyroff', 'M_Senol24'
]
ALLOWED_CHAT_IDS = [5124738136, 5633085280, 1332756927, 5140980618]

# Target coins for which notifications will be sent
TARGET_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT'
]

# Global daily notification tracker
daily_notification_data = {'date': None, 'count': 0}

# Global dictionary to track the last activity time for each chat
user_last_active = {}

# Kullanıcı favorileri ve risk ayarları
user_favorites = {}      # Örn: { "username": ["BTCUSDT", "ETHUSDT"] }
user_risk_settings = {}  # Örn: { "username": 2.0 }  -> risk % değeri

# Inactivity threshold (örneğin 10 dakika)
INACTIVITY_THRESHOLD = timedelta(minutes=10)


def format_price(price: float) -> str:
    """Fiyata göre dinamik formatlama."""
    if price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    elif price >= 0.0001:
        return f"{price:.8f}"
    else:
        return f"{price:.10f}"


def update_user_activity(update: Update) -> None:
    """Komut veren kullanıcının son aktif zamanını günceller."""
    if update.effective_chat:
        chat_id = update.effective_chat.id
        user_last_active[chat_id] = datetime.utcnow()


async def get_crypto_data(symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    """Belirtilen zaman diliminde Binance API'den mum verilerini alır."""
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
                df = df.astype(float)
                return df
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def calculate_pivot_points(data: pd.DataFrame) -> dict:
    """Önceki mum verilerine dayalı pivot noktalarını hesaplar."""
    last_period = data.iloc[-2]
    pivot = (last_period['high'] + last_period['low'] + last_period['close']) / 3
    support1 = (2 * pivot) - last_period['high']
    resistance1 = (2 * pivot) - last_period['low']
    support2 = pivot - (last_period['high'] - last_period['low'])
    resistance2 = pivot + (last_period['high'] - last_period['low'])
    return {'pivot': pivot, 'support1': support1, 'resistance1': resistance1,
            'support2': support2, 'resistance2': resistance2}


def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    """MACD ve ilgili göstergeleri hesaplar."""
    macd_indicator = ta.trend.MACD(close=data['close'], window_slow=12, window_fast=6, window_sign=3)
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()
    data['macd_diff'] = macd_indicator.macd_diff()
    return data


def calculate_atr(data: pd.DataFrame, period: int = 7) -> float:
    """ATR hesaplar."""
    atr_indicator = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=period)
    return atr_indicator.average_true_range().iloc[-1]


def calculate_entry_price(data: pd.DataFrame) -> float:
    """En son mumun açılış fiyatını hesaplar."""
    return data['open'].iloc[-1]


async def get_technical_indicators(symbol: str) -> dict:
    """1h zaman diliminde teknik göstergeleri hesaplar."""
    try:
        data = await get_crypto_data(symbol, interval='1h', limit=100)
        if data.empty:
            raise ValueError("No data received from Binance API.")
        pivot_points = calculate_pivot_points(data)
        data = calculate_macd(data)
        atr = calculate_atr(data, period=7)
        current_price = data['close'].iloc[-1]
        entry_price = calculate_entry_price(data)
        # Ek trend onayları
        data['ema_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()
        return {'pivot_points': pivot_points, 'current_price': current_price, 'atr': atr,
                'entry_price': entry_price, 'data': data}
    except Exception as e:
        logger.error(f"Error in technical indicators for {symbol}: {e}")
        return {}


def generate_signals(indicators: dict) -> dict:
    """MACD, pivot noktaları ve ek trend onaylarına göre sinyal üretir."""
    current_price = indicators.get('current_price')
    pivot_points = indicators.get('pivot_points')
    atr = indicators.get('atr')
    data = indicators.get('data')
    ema_condition_long = current_price > data['ema_20'].iloc[-1]
    ema_condition_short = current_price < data['ema_20'].iloc[-1]
    obv_condition_long = data['obv'].iloc[-1] > data['obv'].iloc[-2] if len(data) > 1 else False
    obv_condition_short = data['obv'].iloc[-1] < data['obv'].iloc[-2] if len(data) > 1 else False
    buy_signal_strength = (data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]) if (current_price > pivot_points['pivot'] and ema_condition_long and obv_condition_long) else None
    sell_signal_strength = (data['macd_signal'].iloc[-1] - data['macd'].iloc[-1]) if (current_price < pivot_points['pivot'] and ema_condition_short and obv_condition_short) else None
    tp_long = current_price + (2 * atr)
    sl_long = current_price - (1 * atr)
    tp_short = current_price - (2 * atr)
    sl_short = current_price + (1 * atr)
    return {'buy_signal': buy_signal_strength is not None,
            'sell_signal': sell_signal_strength is not None,
            'buy_signal_strength': buy_signal_strength,
            'sell_signal_strength': sell_signal_strength,
            'pivot_points': pivot_points,
            'tp_long': tp_long,
            'sl_long': sl_long,
            'tp_short': tp_short,
            'sl_short': sl_short}


def is_user_allowed(update: Update) -> bool:
    """Kullanıcının yetkisini kontrol eder."""
    user = update.effective_user
    return user.username in ALLOWED_USERS


async def send_trade_notification(context: ContextTypes.DEFAULT_TYPE, symbol: str, direction: str,
                                  entry_price: float, tp: float, sl: float) -> None:
    """Onaylanmış chat ID'lerine ticaret sinyalini gönderir."""
    global daily_notification_data
    today = datetime.utcnow().date()
    if daily_notification_data.get('date') != today:
        daily_notification_data['date'] = today
        daily_notification_data['count'] = 0
    if daily_notification_data['count'] >= 8:
        logger.info("Daily notification limit reached.")
        return
    message_text = (f"🪬 *Trade Signal*\n\n"
                    f"*Coin*: {symbol}\n"
                    f"*Direction*: {direction.upper()}\n\n"
                    f"*Entry*: {format_price(entry_price)}\n"
                    f"*Take Profit*: {format_price(tp)}\n"
                    f"*Stop Loss*: {format_price(sl)}\n")
    for chat_id in ALLOWED_CHAT_IDS:
        last_active = user_last_active.get(chat_id)
        if last_active is None or (datetime.utcnow() - last_active) > INACTIVITY_THRESHOLD:
            try:
                await context.bot.send_message(chat_id=chat_id, text=message_text, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending notification to chat {chat_id}: {e}")
        else:
            logger.info(f"User in chat {chat_id} is active; skipping notification.")
    daily_notification_data['count'] += 1
    logger.info(f"Total notifications today: {daily_notification_data['count']}")


async def analyze_general_trend() -> str:
    """Fear & Greed Index ve BTCUSDT günlük EMA'larına göre genel trend analizi yapar."""
    fng_index = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/?limit=1") as response:
                if response.status == 200:
                    data = await response.json()
                    fng_index = int(data["data"][0]["value"])
                else:
                    logger.error(f"Failed to fetch F&G index, status: {response.status}")
    except Exception as e:
        logger.error(f"Error fetching F&G index: {e}")

    btc_ema_20 = None
    btc_ema_50 = None
    try:
        btc_data = await get_crypto_data("BTCUSDT", interval="1d", limit=60)
        if not btc_data.empty:
            btc_closes = btc_data["close"].tolist()
            if len(btc_closes) >= 50:
                btc_ema_20 = sum(btc_closes[-20:]) / 20
                btc_ema_50 = sum(btc_closes[-50:]) / 50
    except Exception as e:
        logger.error(f"Error calculating BTC EMA: {e}")

    if fng_index is not None and btc_ema_20 is not None and btc_ema_50 is not None:
        if fng_index > 60 and btc_ema_20 > btc_ema_50:
            return "TREND Long 🚀"
        elif fng_index < 40 and btc_ema_20 < btc_ema_50:
            return "TREND Short 🩸"
        else:
            return "TREND Nötr 🤚🏼"
    else:
        return "Trend analizi için gerekli veriler alınamadı."


async def generate_trend_chart() -> io.BytesIO:
    """BTCUSDT günlük verileri ile trend analizi grafiği oluşturur."""
    data = await get_crypto_data("BTCUSDT", interval="1d", limit=60)
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


async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /trend komutuyla genel trend analizini yapar. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.\nJoin Global Community: @coinradarsinyal")
        return
    await update.message.reply_text("Trend analizi yapılıyor, lütfen bekleyiniz...")
    trend_result = await analyze_general_trend()
    trend_chart = await generate_trend_chart()
    if trend_chart:
        await update.message.reply_photo(photo=trend_chart, caption=trend_result, parse_mode='Markdown')
    else:
        await update.message.reply_text(trend_result)


async def set_favorites(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /setfavorites komutuyla favori coin listesini ayarlar. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    if len(context.args) == 0:
        await update.message.reply_text("Usage: /setfavorites BTCUSDT,ETHUSDT,...")
        return
    favs = [fav.strip().upper() for fav in context.args[0].split(',')]
    user_favorites[update.effective_user.username] = favs
    await update.message.reply_text(f"Favorites set: {', '.join(favs)}")


async def get_favorites(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /getfavorites komutuyla favori coin listesini gösterir. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    favs = user_favorites.get(update.effective_user.username, [])
    if not favs:
        await update.message.reply_text("No favorites set.")
    else:
        await update.message.reply_text("Your favorites: " + ", ".join(favs))


async def set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /setrisk komutuyla risk ayarını (yüzde) belirler. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /setrisk <risk_percentage>")
        return
    try:
        risk = float(context.args[0])
        user_risk_settings[update.effective_user.username] = risk
        await update.message.reply_text(f"Risk setting set to {risk}%")
    except Exception as e:
        await update.message.reply_text("Invalid risk value.")


async def get_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /getrisk komutuyla mevcut risk ayarını gösterir. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    risk = user_risk_settings.get(update.effective_user.username, None)
    if risk is None:
        await update.message.reply_text("No risk setting found.")
    else:
        await update.message.reply_text(f"Your risk setting: {risk}%")


async def generate_chart(symbol: str, timeframe: str = "1h") -> io.BytesIO:
    """Belirtilen zaman diliminde RSI, Bollinger ve Fibonacci göstergeli grafik oluşturur."""
    data = await get_crypto_data(symbol, interval=timeframe, limit=100)
    if data.empty:
        return None
    rsi_indicator = ta.momentum.RSIIndicator(close=data['close'], window=14)
    data['rsi'] = rsi_indicator.rsi()
    bb_indicator = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_upper'] = bb_indicator.bollinger_hband()
    data['bb_lower'] = bb_indicator.bollinger_lband()
    data['bb_middle'] = bb_indicator.bollinger_mavg()
    fib_data = data[-60:] if len(data) >= 60 else data
    highest = fib_data['high'].max()
    lowest = fib_data['low'].min()
    diff = highest - lowest
    fib_levels = {'0.0%': highest,
                  '23.6%': highest - 0.236 * diff,
                  '38.2%': highest - 0.382 * diff,
                  '50.0%': highest - 0.5 * diff,
                  '61.8%': highest - 0.618 * diff,
                  '100.0%': lowest}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(data.index, data['close'], label="Close", color="blue")
    ax1.plot(data.index, data['bb_upper'], label="BB Upper", color="red", linestyle="--")
    ax1.plot(data.index, data['bb_lower'], label="BB Lower", color="green", linestyle="--")
    ax1.plot(data.index, data['bb_middle'], label="BB Middle", color="orange", linestyle="--")
    for level, value in fib_levels.items():
        ax1.axhline(value, linestyle='--', label=f"Fib {level}")
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


async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /chart komutuyla grafik raporu oluşturur. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    if len(context.args) == 0:
        await update.message.reply_text("Usage: /chart <symbol> [timeframe]. Example: /chart BTCUSDT 4h")
        return
    symbol = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    await update.message.reply_text("Grafik oluşturuluyor, lütfen bekleyiniz...")
    chart_image = await generate_chart(symbol, timeframe)
    if chart_image is None:
        await update.message.reply_text("Grafik oluşturulamadı. Lütfen coin sembolünü ve zaman dilimini kontrol ediniz.")
    else:
        await update.message.reply_photo(photo=chart_image)


async def realtime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /realtime komutuyla gerçek zamanlı fiyat güncellemeleri gönderir. """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    if len(context.args) == 0:
        await update.message.reply_text("Usage: /realtime <symbol>")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"Connecting to real-time data for {symbol}...")
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                count = 0
                start_time = datetime.utcnow()
                while count < 10 and (datetime.utcnow() - start_time).total_seconds() < 30:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        price = float(data.get('p', 0))
                        time_ms = data.get('T')
                        trade_time = datetime.utcfromtimestamp(time_ms / 1000).strftime("%H:%M:%S")
                        update_text = f"Real-time update for {symbol}:\nPrice: {price}\nTime: {trade_time}"
                        await update.message.reply_text(update_text)
                        count += 1
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    except Exception as e:
        logger.error(f"Error in realtime data: {e}")
        await update.message.reply_text("Error connecting to real-time data.")


async def generate_adv_chart(symbol: str, timeframe: str = "1h") -> io.BytesIO:
    """Gelişmiş teknik analiz: Ichimoku, Stochastic Oscillator ve RSI içeren grafik oluşturur."""
    data = await get_crypto_data(symbol, interval=timeframe, limit=100)
    if data.empty:
        return None
    try:
        ichimoku = ta.trend.IchimokuIndicator(high=data['high'], low=data['low'], window1=9, window2=26, window3=52)
        data['ichimoku_a'] = ichimoku.ichimoku_a()
        data['ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception as e:
        logger.error(f"Error computing Ichimoku: {e}")
    try:
        stochastic = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14, smooth_window=3)
        data['stoch_k'] = stochastic.stoch()
        data['stoch_d'] = stochastic.stoch_signal()
    except Exception as e:
        logger.error(f"Error computing Stochastic Oscillator: {e}")
    rsi_indicator = ta.momentum.RSIIndicator(close=data['close'], window=14)
    data['rsi'] = rsi_indicator.rsi()
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


async def adv_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /adv_analysis komutuyla gelişmiş teknik analiz grafiği oluşturur."""
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    symbol = context.args[0].upper() if context.args else "BTCUSDT"
    await update.message.reply_text(f"Performing advanced analysis for {symbol}...")
    chart_image = await generate_adv_chart(symbol, "1h")
    if chart_image:
        await update.message.reply_photo(photo=chart_image, caption=f"Advanced Technical Analysis for {symbol}", parse_mode='Markdown')
    else:
        await update.message.reply_text("Could not generate advanced analysis chart.")


async def get_market_news() -> list:
    """CryptoCompare API kullanarak en güncel piyasa haberlerini getirir."""
    news_url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(news_url) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = data.get("Data", [])
                    top_news = []
                    for item in news_items[:5]:
                        title = item.get("title", "No Title")
                        url = item.get("url", "")
                        published_on = datetime.utcfromtimestamp(item.get("published_on", 0)).strftime("%Y-%m-%d %H:%M")
                        top_news.append(f"*{title}*\nPublished: {published_on}\n[Read more]({url})")
                    return top_news
                else:
                    logger.error("Error fetching news")
                    return []
    except Exception as e:
        logger.error(f"Error in market news: {e}")
        return []


async def news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ /news komutuyla en güncel piyasa haberlerini gönderir."""
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    await update.message.reply_text("Fetching latest market news...")
    news_items = await get_market_news()
    if news_items:
        message_text = "*Latest Market News:*\n\n" + "\n\n".join(news_items)
        await update.message.reply_text(message_text, parse_mode='Markdown', disable_web_page_preview=True)
    else:
        await update.message.reply_text("Could not fetch news at this time.")


async def ml_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /ml komutu: Makine öğrenimi modeli kullanarak geçmiş verilere dayalı sinyal üretir.
    Kullanım: /ml <symbol> [timeframe] (varsayılan: 1h)
    """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.")
        return
    if len(context.args) == 0:
        await update.message.reply_text("Usage: /ml <symbol> [timeframe]")
        return
    symbol = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    await update.message.reply_text(f"Running ML analysis for {symbol} on {timeframe} timeframe...")
    
    # Geçmiş verileri al (örneğin 200 mum)
    data = await get_crypto_data(symbol, interval=timeframe, limit=200)
    if data.empty:
        await update.message.reply_text("No data available for ML analysis.")
        return

    # Teknik göstergeler hesaplama (örneğin RSI, EMA20, MACD)
    try:
        data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
        data['ema_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
        data['macd'] = ta.trend.MACD(close=data['close'], window_slow=12, window_fast=6, window_sign=3).macd()
    except Exception as e:
        logger.error(f"Error computing technical indicators for ML: {e}")
        await update.message.reply_text("Error computing technical indicators for ML analysis.")
        return

    # Hedef değişken: Bir sonraki mumdaki getiri
    data['target'] = (data['close'].shift(-1) - data['close']) / data['close']
    data = data.dropna()
    
    # Özellikler ve hedefi belirle
    features = data[['rsi', 'ema_20', 'macd']].values
    target = data['target'].values

    # Basit Linear Regression modeli eğit
    try:
        model = LinearRegression()
        model.fit(features, target)
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        await update.message.reply_text("Error training ML model.")
        return

    # Son veri noktasını kullanarak tahmin yap
    last_features = features[-1].reshape(1, -1)
    predicted_return = model.predict(last_features)[0]

    if predicted_return > 0:
        ml_signal_text = f"LONG signal predicted with expected return of {predicted_return*100:.2f}%"
    else:
        ml_signal_text = f"SHORT signal predicted with expected return of {predicted_return*100:.2f}%"
    
    await update.message.reply_text(f"ML Analysis for {symbol} ({timeframe}):\n{ml_signal_text}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    update_user_activity(update)
    if is_user_allowed(update):
        await update.message.reply_text(
            'Welcome to CoinRadar AI! 🎉\n\n'
            'To receive a signal, use the command /coin <symbol> (e.g. /coin BTCUSDT).\n'
            'Or scan the market with:\n'
            '/long - for long signals\n'
            '/short - for short signals\n'
            '/trend - for general market trend analysis\n\n'
            'Additional commands:\n'
            '/chart <symbol> [timeframe] - Get a technical chart (e.g., /chart BTCUSDT 4h)\n'
            '/setfavorites <coin1,coin2,...> - Set your favorite coins\n'
            '/getfavorites - Show your favorite coins\n'
            '/setrisk <risk_percentage> - Set your risk percentage\n'
            '/getrisk - Show your risk setting\n'
            '/realtime <symbol> - Get real-time price updates\n'
            '/adv_analysis <symbol> - Advanced technical analysis\n'
            '/news - Get latest market news\n'
            '/ml <symbol> [timeframe] - ML based signal analysis'
        )
    else:
        await update.message.reply_text("You don't have access permission for the AI trader.\nJoin Global Community: @coinradarsinyal")


async def loading_bar(message: Message):
    """Yüzdelik ilerleme gösteren yükleniyor animasyonu."""
    for i in range(0, 101, 10):
        await asyncio.sleep(0.3)
        await message.edit_text(f'Analyzing... {i}%')
    await message.edit_text('Signal found!')


async def coin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /coin komutu: Teknik göstergeleri hesaplar, sinyal üretir,
    risk ayarına göre kaldıraç önerir ve grafik raporu gönderir.
    """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.\nJoin Global Community: @coinradarsinyal")
        return
    message: Message = await update.message.reply_text('Starting analysis...')
    loading_task = asyncio.create_task(loading_bar(message))
    try:
        if len(context.args) == 0:
            await message.edit_text('Please specify a coin, e.g. /coin BTCUSDT')
            return
        symbol = context.args[0].upper()
        indicators = await get_technical_indicators(symbol)
        await loading_task
        if indicators:
            signals = generate_signals(indicators)
            signal_message = f"🪬 *{symbol}*\n\n"
            if signals['buy_signal']:
                signal_message += (
                    f"🚀 *Direction*: LONG\n\n"
                    f"*Entry*: {format_price(indicators['entry_price'])}\n"
                    f"*Take Profit*: {format_price(signals['tp_long'])}\n"
                    f"*Stop Loss*: {format_price(signals['sl_long'])}\n"
                )
                direction = 'Long'
                entry_price = indicators['entry_price']
                tp = signals['tp_long']
                sl = signals['sl_long']
            elif signals['sell_signal']:
                signal_message += (
                    f"🩸 *Direction*: SHORT\n\n"
                    f"*Entry*: {format_price(indicators['entry_price'])}\n"
                    f"*Take Profit*: {format_price(signals['tp_short'])}\n"
                    f"*Stop Loss*: {format_price(signals['sl_short'])}\n"
                )
                direction = 'Short'
                entry_price = indicators['entry_price']
                tp = signals['tp_short']
                sl = signals['sl_short']
            else:
                signal_message += "🤚🏼 *No Clear Signal at the moment*\n"
                direction = None
            risk_setting = user_risk_settings.get(update.effective_user.username, None)
            if risk_setting is not None and direction is not None:
                if direction == 'Long':
                    risk_distance = indicators['entry_price'] - signals['sl_long']
                else:
                    risk_distance = signals['sl_short'] - indicators['entry_price']
                if risk_distance > 0:
                    recommended_leverage = (indicators['entry_price'] * (risk_setting / 100)) / risk_distance
                    recommended_leverage = round(recommended_leverage, 1)
                    signal_message += (
                        "\n*Risk Management:*\n"
                        f"Your risk setting: {risk_setting}%\n"
                        f"Recommended Leverage: {recommended_leverage}x\n"
                        f"Allocate {risk_setting}% of your capital as margin."
                    )
            chart_image = await generate_chart(symbol, "1h")
            if chart_image:
                await update.message.reply_photo(photo=chart_image, caption=signal_message, parse_mode='Markdown')
            else:
                await update.message.reply_text(signal_message, parse_mode='Markdown')
            if symbol in TARGET_COINS and direction is not None:
                await send_trade_notification(context, symbol, direction, entry_price, tp, sl)
        else:
            await update.message.reply_text(f"⚠️ {symbol} - check the symbol or try again later.")
    except Exception as e:
        logger.error(f"Error in /coin command: {e}")
        await update.message.reply_text("An error occurred during analysis.")
    finally:
        loading_task.cancel()


async def sell_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /short komutuyla kısa sinyalleri tarar ve en iyi 7 sinyali bildirir.
    """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.\nJoin Global Community: @coinradarsinyal")
        return
    message: Message = await update.message.reply_text('Scanning market for short signals...')
    loading_task = asyncio.create_task(loading_bar(message))
    try:
        symbols = TARGET_COINS
        sell_signals_list = []
        tasks = [get_technical_indicators(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        for symbol, indicators in zip(symbols, results):
            if indicators:
                signals = generate_signals(indicators)
                if signals['sell_signal']:
                    sell_signals_list.append((symbol, signals['sell_signal_strength'], signals, indicators['entry_price']))
        await loading_task
        if sell_signals_list:
            sell_signals_list = sorted(sell_signals_list, key=lambda x: x[1], reverse=True)[:7]
            signals_message = "🩸 *Top SHORT Signals*\n\n"
            for symbol, strength, sig, entry_price in sell_signals_list:
                signals_message += (
                    f"🪬*{symbol}*\n\n"
                    f"   Direction: SHORT\n"
                    f"   Entry: {format_price(entry_price)}\n"
                    f"   TP: {format_price(sig['tp_short'])}\n"
                    f"   SL: {format_price(sig['sl_short'])}\n\n"
                )
            await update.message.reply_text(signals_message, parse_mode='Markdown')
        else:
            await update.message.reply_text('No strong short signals found at this time.')
    except Exception as e:
        logger.error(f"Error in /short command: {e}")
        await update.message.reply_text("An error occurred while scanning for short signals.")
    finally:
        loading_task.cancel()


async def long_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /long komutuyla uzun sinyalleri tarar ve en iyi 7 sinyali bildirir.
    """
    update_user_activity(update)
    if not is_user_allowed(update):
        await update.message.reply_text("You don't have access permission for the AI trader.\nJoin Global Community: @coinradarsinyal")
        return
    message: Message = await update.message.reply_text('Scanning market for long signals...')
    loading_task = asyncio.create_task(loading_bar(message))
    try:
        symbols = TARGET_COINS
        long_signals_list = []
        tasks = [get_technical_indicators(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        for symbol, indicators in zip(symbols, results):
            if indicators:
                signals = generate_signals(indicators)
                if signals['buy_signal']:
                    long_signals_list.append((symbol, signals['buy_signal_strength'], signals, indicators['entry_price']))
        await loading_task
        if long_signals_list:
            long_signals_list = sorted(long_signals_list, key=lambda x: x[1], reverse=True)[:7]
            signals_message = "🚀 *Top LONG Signals*\n\n"
            for symbol, strength, sig, entry_price in long_signals_list:
                signals_message += (
                    f"🪬*{symbol}*\n\n"
                    f"   Direction: LONG\n"
                    f"   Entry: {format_price(entry_price)}\n"
                    f"   TP: {format_price(sig['tp_long'])}\n"
                    f"   SL: {format_price(sig['sl_long'])}\n\n"
                )
            await update.message.reply_text(signals_message, parse_mode='Markdown')
        else:
            await update.message.reply_text('No strong long signals found at this time.')
    except Exception as e:
        logger.error(f"Error in /long command: {e}")
        await update.message.reply_text("An error occurred while scanning for long signals.")
    finally:
        loading_task.cancel()


async def get_all_symbols() -> list:
    """USDT paritesiyle işlem gören tüm sembolleri getirir."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.binance.com/api/v3/exchangeInfo') as response:
                response.raise_for_status()
                data = await response.json()
                return [item['symbol'] for item in data['symbols']
                        if item['status'] == 'TRADING' and item['quoteAsset'] == 'USDT']
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching all symbols: {e}")
        return []


def main() -> None:
    application = Application.builder().token(TOKEN).build()
    # Varsayılan sinyal komutları
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("coin", coin))
    application.add_handler(CommandHandler("long", long_signals))
    application.add_handler(CommandHandler("short", sell_signals))
    application.add_handler(CommandHandler("trend", trend))
    # Ek özellik komutları
    application.add_handler(CommandHandler("chart", chart))
    application.add_handler(CommandHandler("setfavorites", set_favorites))
    application.add_handler(CommandHandler("getfavorites", get_favorites))
    application.add_handler(CommandHandler("setrisk", set_risk))
    application.add_handler(CommandHandler("getrisk", get_risk))
    # Yeni eklenen özellikler
    application.add_handler(CommandHandler("realtime", realtime))
    application.add_handler(CommandHandler("adv_analysis", adv_analysis))
    application.add_handler(CommandHandler("news", news))
    application.add_handler(CommandHandler("ml", ml_signal))
    application.run_polling()


if __name__ == "__main__":
    main()
