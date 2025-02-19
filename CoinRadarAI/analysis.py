import asyncio
from config import GEMINI_API_KEY, t, logger
import google.generativeai as palm  # Değişiklik: alias 'palm' olarak kullanılıyor

async def gemini_completion(prompt, max_tokens=200, temperature=0.7):
    """
    Gemini API çağrısı yapar.
    """
    palm.configure(api_key=GEMINI_API_KEY)
    try:
        # google.generativeai.generate_text senkron çalıştığından, async hale getirmek için asyncio.to_thread kullanıyoruz
        response = await asyncio.to_thread(
            palm.generate_text,
            model="gemini-2.0-flash",
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.95,
            top_k=40,
            response_mime_type="text/plain"
        )
        return response
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
