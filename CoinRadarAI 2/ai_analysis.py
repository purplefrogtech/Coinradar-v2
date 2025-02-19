import gemini_api
from config import GEMINI_API_KEY, t, logger

gemini_api.api_key = GEMINI_API_KEY

def gemini_completion(prompt, max_tokens=100, temperature=0.7):
    # Simüle edilmiş API çağrısı; gerçek API çağrınızı buraya ekleyin.
    return {"choices": [{"text": "CoinRadar AI"}]}

async def interpret_chart(symbol, timeframe, indicators) -> str:
    try:
        data = indicators.get("data")
        if data is None or data.empty:
            return "Yeterli veri bulunamadı."
        prompt = (
            f"Lütfen {symbol} için {timeframe} zaman diliminde oluşturulan teknik göstergeleri analiz et.\n"
            f"- Güncel Fiyat: {indicators.get('current_price'):.4f}\n"
            f"- EMA20: {data['ema_20'].iloc[-1]:.4f}\n"
            f"- MACD: {data['macd'].iloc[-1]:.4f} (Signal: {data['macd_signal'].iloc[-1]:.4f})\n"
            f"- RSI: {data['rsi'].iloc[-1]:.2f}\n"
            f"Lütfen kısa ve öz bir piyasa yorumu yap."
        )
        response = gemini_completion(prompt, max_tokens=100, temperature=0.7)
        analysis = response["choices"][0]["text"].strip()
        return analysis
    except Exception as e:
        logger.error(f"Gemini yorumlama hatası: {e}")
        return "Yapay zeka yorumlaması yapılamadı."
