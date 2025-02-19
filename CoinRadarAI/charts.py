import io
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import ta
from data_fetch import get_crypto_data
from urllib.parse import quote

async def generate_chart(symbol, timeframe="1h"):
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

async def generate_adv_chart(symbol, timeframe="1h"):
    data = await get_crypto_data(symbol, interval=timeframe, limit=100)
    if data.empty:
        return None
    try:
        ichimoku = ta.trend.IchimokuIndicator(high=data['high'], low=data['low'], window1=9, window2=26, window3=52)
        data['ichimoku_a'] = ichimoku.ichimoku_a()
        data['ichimoku_b'] = ichimoku.ichimoku_b()
    except Exception as e:
        print(f"Ichimoku error: {e}")
    try:
        stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14, smooth_window=3)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
    except Exception as e:
        print(f"Stochastic error: {e}")
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

async def generate_trend_chart_custom(interval):
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
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf
