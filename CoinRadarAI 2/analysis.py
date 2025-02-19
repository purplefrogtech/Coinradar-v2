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
