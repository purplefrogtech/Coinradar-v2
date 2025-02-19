import pandas as pd
import aiohttp
import ta
from datetime import datetime
from config import BINANCE_API_URL, logger

async def get_crypto_data(symbol, interval='1h', limit=100):
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

def calculate_pivot_points(data):
    last = data.iloc[-2]
    pivot = (last['high'] + last['low'] + last['close']) / 3
    return {
        'pivot': pivot,
        'support1': (2 * pivot) - last['high'],
        'resistance1': (2 * pivot) - last['low'],
        'support2': pivot - (last['high'] - last['low']),
        'resistance2': pivot + (last['high'] - last['low'])
    }

def calculate_macd(data):
    macd = ta.trend.MACD(close=data['close'], window_slow=12, window_fast=6, window_sign=3)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    return data

def calculate_atr(data, period=7):
    atr = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=period)
    return atr.average_true_range().iloc[-1]

def calculate_entry_price(data):
    return data['open'].iloc[-1]

async def get_technical_indicators(symbol, timeframe='1h'):
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
