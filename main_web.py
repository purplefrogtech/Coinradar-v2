import asyncio
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List

# Import your core functions. In this example, we assume they are in the same file.
# Otherwise, you could do: from core import get_technical_indicators, generate_signals, format_price
import pandas as pd
import ta
import aiohttp
from datetime import datetime, timedelta

# ----------------------------
# --- Core Bot Logic Start ---
# ----------------------------

# (The following code is taken from your existing bot with adjustments for short-term analysis
#  and includes the format_price helper. Replace or refactor as needed.)

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def format_price(price: float) -> str:
    """
    Dynamically format the price based on its magnitude.
    """
    if price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    elif price >= 0.0001:
        return f"{price:.8f}"
    else:
        return f"{price:.10f}"

async def get_crypto_data(symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    """
    Fetch historical candlestick data from Binance using a 1h timeframe for short-term analysis.
    """
    try:
        url = f"{BINANCE_API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logging.error(f"Error fetching data from Binance API: {response.status}, {await response.text()}")
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
        logging.error(f"Error fetching data from Binance API: {e}")
        return pd.DataFrame()

def calculate_pivot_points(data: pd.DataFrame) -> dict:
    """
    Calculate pivot points based on the previous candle.
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
    Compute MACD, MACD signal, and MACD difference using short-term parameters.
    """
    macd_indicator = ta.trend.MACD(close=data['close'], window_slow=12, window_fast=6, window_sign=3)
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()
    data['macd_diff'] = macd_indicator.macd_diff()
    return data

def calculate_atr(data: pd.DataFrame, period: int = 7) -> float:
    """
    Calculate the ATR with a shorter period.
    """
    atr_indicator = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=period)
    return atr_indicator.average_true_range().iloc[-1]

def calculate_entry_price(data: pd.DataFrame) -> float:
    """
    Calculate the entry price using the open price of the latest candle.
    """
    return data['open'].iloc[-1]

async def get_technical_indicators(symbol: str) -> dict:
    """
    Fetch market data and compute technical indicators for short-term analysis.
    """
    try:
        data = await get_crypto_data(symbol, interval='1h', limit=100)
        if data.empty:
            raise ValueError("No data received from Binance API.")
        
        pivot_points = calculate_pivot_points(data)
        data = calculate_macd(data)
        atr = calculate_atr(data, period=7)
        current_price = data['close'].iloc[-1]
        entry_price = calculate_entry_price(data)

        # Additional trend filters: 20-period EMA and OBV
        data['ema_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()

        return {
            'pivot_points': pivot_points,
            'current_price': current_price,
            'atr': atr,
            'entry_price': entry_price,
            'data': data
        }
    except Exception as e:
        logging.error(f"Error in technical indicators calculation for {symbol}: {e}")
        return {}

def generate_signals(indicators: dict) -> dict:
    """
    Generate buy or sell signals and calculate TP/SL for short-term analysis.
    """
    current_price = indicators.get('current_price')
    pivot_points = indicators.get('pivot_points')
    atr = indicators.get('atr')
    data = indicators.get('data')

    # Trend confirmation using EMA and OBV
    ema_condition_long = current_price > data['ema_20'].iloc[-1]
    ema_condition_short = current_price < data['ema_20'].iloc[-1]
    obv_condition_long = data['obv'].iloc[-1] > data['obv'].iloc[-2] if len(data) > 1 else False
    obv_condition_short = data['obv'].iloc[-1] < data['obv'].iloc[-2] if len(data) > 1 else False

    buy_signal_strength = (data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]) if (current_price > pivot_points['pivot'] and ema_condition_long and obv_condition_long) else None
    sell_signal_strength = (data['macd_signal'].iloc[-1] - data['macd'].iloc[-1]) if (current_price < pivot_points['pivot'] and ema_condition_short and obv_condition_short) else None

    # Dynamic TP/SL with short-term multipliers
    tp_long = current_price + (2 * atr)
    sl_long = current_price - (1 * atr)
    tp_short = current_price - (2 * atr)
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

# ----------------------------
# --- Core Bot Logic End ---
# ----------------------------

# ----------------------------
# --- FastAPI Application  ---
# ----------------------------

app = FastAPI()

# Define a request model for commands
class CommandRequest(BaseModel):
    command: str
    args: List[str] = []

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """
    Serves a simple chat-style HTML interface.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>CoinRadar Chat Interface</title>
      <style>
        body { font-family: Arial, sans-serif; background: #f0f0f0; }
        #chat { width: 80%; margin: 20px auto; background: #fff; padding: 20px; border-radius: 8px; }
        .message { margin: 10px 0; }
        .user { color: blue; }
        .bot { color: green; }
      </style>
    </head>
    <body>
      <div id="chat">
        <div id="messages"></div>
        <input type="text" id="input" placeholder="Enter your command..." style="width:80%;">
        <button onclick="sendCommand()">Send</button>
      </div>
      <script>
        async function sendCommand() {
          const inputField = document.getElementById('input');
          const commandText = inputField.value;
          inputField.value = '';
          appendMessage('You: ' + commandText, 'user');
          
          // Split command and arguments by whitespace.
          let parts = commandText.trim().split(' ');
          let command = parts[0];
          let args = parts.slice(1);
          
          const response = await fetch('/command', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ command: command, args: args })
          });
          const data = await response.json();
          appendMessage('Bot: ' + data.response, 'bot');
        }
        function appendMessage(message, sender) {
          const messagesDiv = document.getElementById('messages');
          const messageDiv = document.createElement('div');
          messageDiv.className = 'message ' + sender;
          messageDiv.innerText = message;
          messagesDiv.appendChild(messageDiv);
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/command")
async def process_command(cmd: CommandRequest):
    """
    Process a command sent from the chat interface.
    Example commands: /coin BTCUSDT, /long, /short, etc.
    """
    command = cmd.command.strip()
    args = cmd.args

    if command.lower() == "/coin":
        if len(args) == 0:
            return JSONResponse(content={"response": "Please specify a coin symbol, e.g. /coin BTCUSDT"})
        symbol = args[0].upper()
        indicators = await get_technical_indicators(symbol)
        if not indicators:
            return JSONResponse(content={"response": f"⚠️ {symbol} - check the symbol or try again later."})
        signals = generate_signals(indicators)
        if signals['buy_signal']:
            response_text = (
                f"💎 *{symbol}*\n\n"
                f"🚀 *Direction*: LONG\n"
                f"💰 *Entry*: {format_price(indicators['entry_price'])}\n"
                f"🎯 *Take Profit*: {format_price(signals['tp_long'])}\n"
                f"🛡️ *Stop Loss*: {format_price(signals['sl_long'])}\n"
            )
        elif signals['sell_signal']:
            response_text = (
                f"💎 *{symbol}*\n\n"
                f"🩸 *Direction*: SHORT\n"
                f"💰 *Entry*: {format_price(indicators['entry_price'])}\n"
                f"🎯 *Take Profit*: {format_price(signals['tp_short'])}\n"
                f"🛡️ *Stop Loss*: {format_price(signals['sl_short'])}\n"
            )
        else:
            response_text = "🤚🏼 *No Clear Signal at the moment*"
        return JSONResponse(content={"response": response_text})
    elif command.lower() == "/long":
        # You can implement /long similarly by scanning multiple symbols.
        return JSONResponse(content={"response": "Long signals command is under development."})
    elif command.lower() == "/short":
        return JSONResponse(content={"response": "Short signals command is under development."})
    else:
        return JSONResponse(content={"response": "Unknown command. Please try again."})
