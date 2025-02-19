import asyncio
from data_fetch import get_technical_indicators, get_crypto_data
from signals import generate_signals
from charts import generate_chart, generate_trend_chart_custom
from notifications import map_term_to_interval
from utils import format_price
from config import TARGET_COINS, t, logger

async def coin_analysis_by_term(symbol, term, lang, username):
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
    risk = __import__("config").user_risk_settings.get(username)
    if risk is not None and direction is not None:
        rd = (indicators['entry_price'] - signals['sl_long']) if direction == t("long", lang) else (signals['sl_short'] - indicators['entry_price'])
        if rd > 0:
            lev = round((indicators['entry_price'] * (risk / 100)) / rd, 1)
            msg += (f"\n*{t('risk_management', lang)}*\n"
                    f"{t('your_risk', lang)} {risk}%\n"
                    f"Recommended Leverage: {lev}x\n"
                    f"Allocate {risk}% of your capital as margin.")
    chart = await generate_chart(symbol, tf)
    return msg, chart

async def long_signals_by_term(term, lang):
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

async def short_signals_by_term(term, lang):
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

async def trend_analysis_by_term(term, lang):
    tf = map_term_to_interval(term)
    fng = None
    import aiohttp
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

    # BTCUSDT verisini daha geniş bir limit ile çekiyoruz (100 mum verisi)
    try:
        btc_data = await get_crypto_data("BTCUSDT", interval=tf, limit=100)
        if btc_data.empty:
            raise ValueError("Empty BTC data")
    except Exception as e:
        logger.error(f"BTC data fetch error: {e}")
        return "Trend analizi için BTC verileri alınamadı.", None

    closes = btc_data["close"].tolist()

    # Yardımcı fonksiyonlar: EMA ve RSI hesaplamaları
    def calculate_ema(prices, period):
        if len(prices) < period:
            return sum(prices) / len(prices)
        k = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = price * k + ema * (1 - k)
        return ema

    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1:
            return 50  # Nötr değer
        gains = []
        losses = []
        for i in range(1, period+1):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Göstergelerin hesaplanması
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    ema100 = calculate_ema(closes, 100) if len(closes) >= 100 else None
    rsi = calculate_rsi(closes, 14)
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    macd = ema12 - ema26

    # Sinyalleri birleştiriyoruz
    bullish_conditions = 0
    bearish_conditions = 0
    total_conditions = 0

    # Koşul 1: EMA20 > EMA50 ise bullish sinyal
    total_conditions += 1
    if ema20 > ema50:
        bullish_conditions += 1
    else:
        bearish_conditions += 1

    # Koşul 2: EMA50 > EMA100 (varsa) bullish onay
    if ema100 is not None:
        total_conditions += 1
        if ema50 > ema100:
            bullish_conditions += 1
        else:
            bearish_conditions += 1

    # Koşul 3: RSI > 50 bullish, aksi halde bearish
    total_conditions += 1
    if rsi > 50:
        bullish_conditions += 1
    else:
        bearish_conditions += 1

    # Koşul 4: MACD pozitif ise bullish, negatif ise bearish
    total_conditions += 1
    if macd > 0:
        bullish_conditions += 1
    else:
        bearish_conditions += 1

    # Koşul 5: FNG değeri – FNG > 60 bullish, FNG < 40 bearish
    if fng is not None:
        total_conditions += 1
        if fng > 60:
            bullish_conditions += 1
        elif fng < 40:
            bearish_conditions += 1

    # Koşulların %60'ı üzerinde sinyal varsa eğilim belirle
    threshold = total_conditions * 0.6
    if bullish_conditions >= threshold and bullish_conditions > bearish_conditions:
        trend = "TREND Long 🚀"
    elif bearish_conditions >= threshold and bearish_conditions > bullish_conditions:
        trend = "TREND Short 🩸"
    else:
        trend = "TREND Nötr 🤚🏼"

    chart = await generate_trend_chart_custom(tf)
    return trend, chart
