import gemini_api
from config import GEMINI_API_KEY, t, logger

gemini_api.api_key = GEMINI_API_KEY

def gemini_completion(prompt, max_tokens=100, temperature=0.7):
    # Simüle edilmiş API çağrısı; gerçek API çağrınızı buraya ekleyin.
    return {"choices": [{"text": "CoinRadar AI"}]}

async def interpret_chart(symbol, timeframe, indicators) -> str:
    # Gelişmiş analizde grafik yorumlaması artık yapılmayacak.
    # Bu nedenle, Gemini API çağrısı kaldırıldı.
    return ""
