import pandas as pd
import ta
import aiohttp
import asyncio
import logging
from datetime import datetime
from telegram import Update, Message
from telegram.ext import Application, CommandHandler, ContextTypes

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

# Telegram bot token
TOKEN = '6366643634:AAGegP6shTT5_XCBSgUBA_VxtVgRc-aNm_Y'

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# List of allowed usernames (for /start access) and allowed chat IDs (for notifications)
ALLOWED_USERS = [
    'paraloperceo', 'LaunchControll', 'ensalgz', 'gorkemk6',
    'WOULTHERR', 'MacqTrulz', 'janexander', 'mmmmonur', 'Ern5716'
]
# Dummy chat IDs of authorized users for notifications – replace with actual chat IDs.
ALLOWED_CHAT_IDS = [5124738136, 5633085280, 1332756927]

# Target coins for which notifications will be sent
TARGET_COINS = ['BTCUSDT', 'XRPUSDT', 'AVAXUSDT', 'ETHUSDT']

# Global daily notification tracker
daily_notification_data = {
    'date': None,
    'count': 0
}

async def get_crypto_data(symbol: str, interval: str = '4h', limit: int = 100) -> pd.DataFrame:
    """
    Fetch historical candlestick data from Binance using a 4h timeframe for medium-term analysis.
    """
    try:
        url = f"{BINANCE_API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error fetching data from Binance API: {response.status}, {await response.text()}")
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
        logger.error(f"Error fetching data from Binance API: {e}")
        return pd.DataFrame()

def calculate_pivot_points(data: pd.DataFrame) -> dict:
    """
    Calculate pivot points based on the second-to-last candle.
    """
    last_period = data.iloc[-2]
    pivot = (last_period['high'] + last_period['low'] + last_period['close']) / 3
    support1 = (2 * pivot) - last_period['high']
    resistance1 = (2 * pivot) - last_period['low']
    support2 = pivot - (last_period['high'] - last_period['low'])
    resistance2 = pivot + (last_period['high'] - last_period['low'])
    
    return {
        'pivot': pivot,
        'support1': support1,
        'resistance1': resistance1,
        'support2': support2,
        'resistance2': resistance2
    }

def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MACD, MACD signal, and MACD difference using the TA library.
    """
    macd_indicator = ta.trend.MACD(close=data['close'])
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()
    data['macd_diff'] = macd_indicator.macd_diff()
    return data

def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) to gauge market volatility.
    """
    atr_indicator = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=period)
    return atr_indicator.average_true_range().iloc[-1]

async def get_technical_indicators(symbol: str) -> dict:
    """
    Fetch market data and compute technical indicators including pivot points, MACD, ATR, EMA, and OBV.
    Uses a 4h timeframe for a medium-term view.
    """
    try:
        data = await get_crypto_data(symbol, interval='4h', limit=100)
        if data.empty:
            raise ValueError("No data received from Binance API.")
        
        pivot_points = calculate_pivot_points(data)
        data = calculate_macd(data)
        atr = calculate_atr(data)
        current_price = data['close'].iloc[-1]

        # Additional trend filters: 50-period EMA and OBV for volume confirmation.
        data['ema_50'] = ta.trend.EMAIndicator(close=data['close'], window=50).ema_indicator()
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()

        return {
            'pivot_points': pivot_points,
            'current_price': current_price,
            'atr': atr,
            'data': data
        }
    except Exception as e:
        logger.error(f"Error in technical indicators calculation: {e}")
        return {}

def generate_signals(indicators: dict) -> dict:
    """
    Generate buy or sell signals based on MACD, pivot points, and additional trend confirmation filters.
    Also calculates dynamic TP/SL levels with an increased distance for medium-term trades.
    """
    current_price = indicators.get('current_price')
    pivot_points = indicators.get('pivot_points')
    atr = indicators.get('atr')
    data = indicators.get('data')

    # Trend confirmation conditions using EMA and OBV
    ema_condition_long = current_price > data['ema_50'].iloc[-1]
    ema_condition_short = current_price < data['ema_50'].iloc[-1]
    obv_condition_long = data['obv'].iloc[-1] > data['obv'].iloc[-2] if len(data) > 1 else False
    obv_condition_short = data['obv'].iloc[-1] < data['obv'].iloc[-2] if len(data) > 1 else False

    buy_signal_strength = (data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]) if (current_price > pivot_points['pivot'] and ema_condition_long and obv_condition_long) else None
    sell_signal_strength = (data['macd_signal'].iloc[-1] - data['macd'].iloc[-1]) if (current_price < pivot_points['pivot'] and ema_condition_short and obv_condition_short) else None

    # Dynamic TP/SL levels with extended distances for medium-term trades.
    tp_long = current_price + (3 * atr)  # For example, risk–reward ratio of ~1:3 for long positions
    sl_long = current_price - (1 * atr)
    tp_short = current_price - (3 * atr)  # Similarly for short positions
    sl_short = current_price + (1 * atr)

    return {
        'buy_signal': buy_signal_strength is not None,
        'sell_signal': sell_signal_strength is not None,
        'buy_signal_strength': buy_signal_strength,
        'sell_signal_strength': sell_signal_strength,
        'pivot_points': pivot_points,
        'tp_long': tp_long,
        'sl_long': sl_long,
        'tp_short': tp_short,
        'sl_short': sl_short
    }

def is_user_allowed(update: Update) -> bool:
    """
    Check if the user is allowed to use the bot.
    """
    user = update.effective_user
    return user.username in ALLOWED_USERS

async def send_trade_notification(context: ContextTypes.DEFAULT_TYPE, symbol: str, direction: str, entry_price: float, tp: float, sl: float) -> None:
    """
    Send a trade notification via Telegram to all approved chat IDs,
    but only if the daily notification limit (4 per day) has not been exceeded.
    """
    global daily_notification_data

    # Get current UTC date
    today = datetime.utcnow().date()
    # Reset the daily counter if the day has changed
    if daily_notification_data.get('date') != today:
        daily_notification_data['date'] = today
        daily_notification_data['count'] = 0

    if daily_notification_data['count'] >= 4:
        logger.info("Daily notification limit reached. Notification not sent.")
        return

    # Compose the notification message
    message_text = (
        f"🚨 *Trade Alert for {symbol}* 🚨\n\n"
        f"*Direction*: {direction}\n"
        f"*Entry Price*: {entry_price:.2f}\n"
        f"*Take Profit (TP)*: {tp:.2f}\n"
        f"*Stop Loss (SL)*: {sl:.2f}\n"
    )
    # Send message to each approved chat ID
    for chat_id in ALLOWED_CHAT_IDS:
        try:
            await context.bot.send_message(chat_id=chat_id, text=message_text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error sending notification to chat {chat_id}: {e}")

    daily_notification_data['count'] += 1
    logger.info(f"Notification sent. Total notifications today: {daily_notification_data['count']}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if is_user_allowed(update):
        await update.message.reply_text(
            'Welcome to CoinRadar AI! 🎉\n\n'
            'To receive signals, use the command /coin <symbol>\n\n'
            'For the AI to scan the entire market for buy and sell signals, you can give the command:\n\n'
            '/long or /short'
        )
    else:
        await update.message.reply_text(
            'You dont have access permission for the AI trader\n\n'
            'To register on the server: @paraloperceo\n\n'
            'To join the Global Community: @paraloper'
        )

async def loading_bar(message: Message):
    """
    Simulate a loading bar by updating the message with progress.
    """
    for i in range(0, 101, 10):
        await asyncio.sleep(0.5)
        await message.edit_text(f'Analyzing... {i}%')
    await message.edit_text('I found!')

async def coin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Process the /coin command, fetch technical indicators, generate signals,
    and return a formatted message. Also, if the coin is in the target list and
    a valid entry signal is identified, send a Telegram notification.
    """
    if not is_user_allowed(update):
        await update.message.reply_text(
            'You dont have access permission for the AI trader\n\n'
            'To register on the server: @paraloperceo\n\n'
            'To join the Global Community: @paraloper'
        )
        return

    message: Message = await update.message.reply_text('I am starting')
    loading_task = asyncio.create_task(loading_bar(message))
    
    try:
        if len(context.args) == 0:
            await message.edit_text('You need to tell me something specific, like /coin BTCUSDT, because I cant understand you otherwise.')
            return
        
        symbol = context.args[0].upper()
        indicators = await get_technical_indicators(symbol)
        
        await loading_task

        if indicators:
            signals = generate_signals(indicators)
            signal_message = f"🪬  *{symbol}*\n\n"

            # Determine the type of signal and associated trade details
            if signals['buy_signal']:
                signal_message += f"🚀 *Long*\n\nTP: {signals['tp_long']:.2f}\nSL: {signals['sl_long']:.2f}\n\n"
                direction = 'Long'
                entry_price = indicators['current_price']
                tp = signals['tp_long']
                sl = signals['sl_long']
            elif signals['sell_signal']:
                signal_message += f"🩸 *Short*\n\nTP: {signals['tp_short']:.2f}\nSL: {signals['sl_short']:.2f}\n\n"
                direction = 'Short'
                entry_price = indicators['current_price']
                tp = signals['tp_short']
                sl = signals['sl_short']
            else:
                signal_message += "🤚🏼 *Wait!*\n\n"
                direction = None

            await update.message.reply_text(signal_message, parse_mode='Markdown')

            # If this coin is among the targeted ones and there is a valid entry signal, send a notification.
            if symbol in TARGET_COINS and direction is not None:
                # Send the trade notification (asynchronously)
                await send_trade_notification(context, symbol, direction, entry_price, tp, sl)
        else:
            await update.message.reply_text(f"⚠️ {symbol} are you sure this is correct?")
    except Exception as e:
        logger.error(f"Error handling /coin command: {e}")
        await update.message.reply_text(f"An error occurred: {e}")
    finally:
        loading_task.cancel()

async def sell_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Scan the market for short signals and return the top 7 based on signal strength.
    """
    if not is_user_allowed(update):
        await update.message.reply_text(
            'You dont have access permission for the AI trader\n\n'
            'To register on the server: @paraloperceo\n\n'
            'To join the Global Community: @paraloper'
        )
        return

    message: Message = await update.message.reply_text('I am starting')
    loading_task = asyncio.create_task(loading_bar(message))

    try:
        symbols = await get_all_symbols()
        if not symbols:
            await message.edit_text('Failed to retrieve coin symbols from the Binance API.')
            return

        sell_signals_list = []
        tasks = [get_technical_indicators(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        for symbol, indicators in zip(symbols, results):
            if indicators:
                signals = generate_signals(indicators)
                if signals['sell_signal']:
                    sell_signals_list.append((symbol, signals['sell_signal_strength']))

        await loading_task

        if sell_signals_list:
            sell_signals_list = sorted(sell_signals_list, key=lambda x: x[1], reverse=True)[:7]
            signals_message = "🩸 *TOP 7 SHORT* 🩸\n\n"
            signals_message += "\n".join([f"{symbol}" for symbol, _ in sell_signals_list])
            await update.message.reply_text(signals_message, parse_mode='Markdown')
        else:
            await update.message.reply_text('There is currently no short signal available.')
    except Exception as e:
        logger.error(f"Error handling /sell_signals command: {e}")
        await update.message.reply_text(f"An error occurred: {e}")
    finally:
        loading_task.cancel()

async def long_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Scan the market for long signals and return the top 7 based on signal strength.
    """
    if not is_user_allowed(update):
        await update.message.reply_text(
            'You dont have access permission for the AI trader\n\n'
            'To register on the server: @paraloperceo\n\n'
            'To join the Global Community: @paraloper'
        )
        return

    message: Message = await update.message.reply_text('I am starting')
    loading_task = asyncio.create_task(loading_bar(message))

    try:
        symbols = await get_all_symbols()
        if not symbols:
            await message.edit_text('Failed to retrieve coin symbols from the Binance API.')
            return

        long_signals_list = []
        tasks = [get_technical_indicators(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        for symbol, indicators in zip(symbols, results):
            if indicators:
                signals = generate_signals(indicators)
                if signals['buy_signal']:
                    long_signals_list.append((symbol, signals['buy_signal_strength']))

        await loading_task

        if long_signals_list:
            long_signals_list = sorted(long_signals_list, key=lambda x: x[1], reverse=True)[:7]
            signals_message = "🚀 *TOP 7 LONG* 🚀\n\n"
            signals_message += "\n".join([f"{symbol}" for symbol, _ in long_signals_list])
            await update.message.reply_text(signals_message, parse_mode='Markdown')
        else:
            await update.message.reply_text('There is currently no long signal available.')
    except Exception as e:
        logger.error(f"Error handling /long_signals command: {e}")
        await update.message.reply_text(f"An error occurred: {e}")
    finally:
        loading_task.cancel()

async def get_all_symbols() -> list:
    """
    Retrieve all trading symbols with USDT as the quote asset.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.binance.com/api/v3/exchangeInfo') as response:
                response.raise_for_status()
                data = await response.json()
                return [
                    item['symbol']
                    for item in data['symbols']
                    if item['status'] == 'TRADING' and item['quoteAsset'] == 'USDT'
                ]
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching all symbols from Binance API: {e}")
        return []

def main() -> None:
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("coin", coin))
    application.add_handler(CommandHandler("long", long_signals))
    application.add_handler(CommandHandler("short", sell_signals))

    application.run_polling()

if __name__ == "__main__":
    main()
