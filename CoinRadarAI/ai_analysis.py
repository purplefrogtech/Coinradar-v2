import aiohttp
from config import GEMINI_API_KEY, t, logger
import google.generativeai as genai

async def gemini_completion(prompt, max_tokens=200, temperature=0.7):
    """
    Gerçek Gemini API çağrısı yapar.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }

    # API anahtarını doğru şekilde ayarla
    genai.configure(api_key=GEMINI_API_KEY)

    try:
        # API'ye istek yap
        async with aiohttp.ClientSession() as session:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    text = await response.text()
                    logger.error(f"Gemini API error: {response.status}: {text}")
                    raise Exception(f"Gemini API call failed: {response.status}: {text}")
    except Exception as e:
        logger.error(f"Gemini API çağrısı başarısız oldu: {e}", exc_info=True)
        raise

async def interpret_chart(symbol, timeframe, indicators) -> str:
    """
    Belirlenen sembol ve zaman dilimine ait teknik göstergeleri kullanarak,
    Gemini API'den kapsamlı bir piyasa yorumu alır.
    """
    try:
        data = indicators.get("data")
        if data is None or data.empty:
            return "Yeterli veri bulunamadı."

        # Hesaplanmış pivot noktalarını al
        pivot_points = indicators.get("pivot_points", {})
        pivot = pivot_points.get("pivot", 0)
        support1 = pivot_points.get("support1", 0)
        resistance1 = pivot_points.get("resistance1", 0)

        # Detaylı prompt oluşturuluyor
        prompt = (
            f"Lütfen {symbol} için {timeframe} zaman diliminde aşağıdaki teknik göstergeleri analiz et:\n"
            f"- Güncel Fiyat: {indicators.get('current_price'):.4f}\n"
            f"- EMA20: {data['ema_20'].iloc[-1]:.4f}\n"
            f"- MACD: {data['macd'].iloc[-1]:.4f} (Sinyal: {data['macd_signal'].iloc[-1]:.4f})\n"
            f"- RSI: {data['rsi'].iloc[-1]:.2f}\n"
            f"- Pivot: {pivot:.4f} | Destek 1: {support1:.4f} | Direnç 1: {resistance1:.4f}\n"
            f"Lütfen bu göstergelere dayanarak piyasanın mevcut durumunu, trendin devam edip etmeyeceğini, "
            f"olası dönüş noktalarını ve riskleri de içeren kapsamlı bir analiz yap. "
            f"Yatırımcıya uygun strateji önerilerini de ekle."
        )

        response = await gemini_completion(prompt, max_tokens=200, temperature=0.7)

        # Yanıtı işleme
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            analysis = candidate.get("output", {}).get("content", "").strip()
            if analysis:
                return analysis
        return "Gemini API'den uygun yanıt alınamadı."
    except Exception as e:
        logger.error(f"Gemini yorumlama hatası: {e}", exc_info=True)
        return "Yapay zeka yorumlaması yapılamadı."
