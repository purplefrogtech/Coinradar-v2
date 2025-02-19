import os
import logging
from datetime import timedelta

# API Anahtarları ve Tokenlar
GEMINI_API_KEY = "AIzaSyC_CyXwRjw0MTxZDagtZ78vR_-KbPqclGc"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
TOKEN = '6900310524:AAFrnifQMXdEcAvwf_qZnT0zcBxbvnJmMXU'
NEWS_API_KEY = '58878f135c9c4d609f856aa552d2d12d'

# Loglama ayarları
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global değişkenler
ALLOWED_USERS = [
    'paraloperceo', 'LaunchControll', 'ensalgz', 'gorkemk6',
    'WOULTHERR', 'MacqTrulz', 'janexander', 'mmmmonur', 'Ern5716',
    'Lord1334', 'thebatyroff', 'M_Senol24'
]
ALLOWED_CHAT_IDS = [5124738136, 5633085280, 1332756927, 5140980618]
TARGET_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT'
]

daily_notification_data = {'date': None, 'count': 0}
user_last_active = {}
user_favorites = {}
user_risk_settings = {}
INACTIVITY_THRESHOLD = timedelta(minutes=10)

# Kullanıcı dil tercihleri (user_id -> "en" veya "tr")
user_language = {}

# Çeviri sözlüğü
translations = {
    "choose_language": {"en": "Please choose your language:", "tr": "Lütfen dilinizi seçiniz:"},
    "language_set_en": {"en": "Language set to English.", "tr": "Dil İngilizce olarak ayarlandı."},
    "language_set_tr": {"en": "Language set to Turkish.", "tr": "Dil Türkçe olarak ayarlandı."},
    "welcome_message": {
        "en": (
            "Welcome to CoinRadar AI! 🎉\n\n"
            "Use /coin <symbol> to get a signal (e.g. /coin BTCUSDT).\n"
            "Or scan the market with:\n"
            "/long - long signals\n"
            "/short - short signals\n"
            "/trend - market trend analysis\n\n"
            "Other commands:\n"
            "/chart <symbol> [timeframe] - Technical chart (e.g. /chart BTCUSDT 4h)\n"
            "/setfavorites <coin1,coin2,...> - Set your favorite coins\n"
            "/getfavorites - Show your favorite coins\n"
            "/setrisk <risk_percentage> - Set your risk percentage\n"
            "/getrisk - Show your risk setting\n"
            "/realtime <symbol> - Real-time price updates\n"
            "/adv_analysis <symbol> - Advanced technical analysis with AI commentary\n"
            "/news - Latest market news\n"
            "/lang <en/tr> - Change language"
        ),
        "tr": (
            "CoinRadar AI'ya hoşgeldiniz! 🎉\n\n"
            "/coin <sembol> komutuyla sinyal alın (örn. /coin BTCUSDT).\n"
            "Veya piyasayı tarayın:\n"
            "/long - uzun sinyaller\n"
            "/short - kısa sinyaller\n"
            "/trend - piyasa trend analizi\n\n"
            "Diğer komutlar:\n"
            "/chart <sembol> [zaman dilimi] - Teknik grafik (örn. /chart BTCUSDT 4h)\n"
            "/setfavorites <coin1,coin2,...> - Favori coinlerinizi ayarlayın\n"
            "/getfavorites - Favorilerinizi gösterin\n"
            "/setrisk <risk_yüzdesi> - Risk ayarınızı belirleyin\n"
            "/getrisk - Risk ayarınızı gösterin\n"
            "/realtime <sembol> - Gerçek zamanlı fiyat güncellemeleri\n"
            "/adv_analysis <sembol> - Gelişmiş teknik analiz (AI yorumlu)\n"
            "/news - En güncel piyasa haberleri\n"
            "/lang <en/tr> - Dil değiştirin"
        )
    },
    "news_header": {"en": "Latest Market News:", "tr": "En Güncel Piyasa Haberleri:"},
    "no_permission": {"en": "You don't have access permission for the AI trader.", "tr": "AI trader'a erişim izniniz yok."},
    "join_community": {"en": "Join Global Community: @coinradarsinyal", "tr": "Global Community'e katılın: @coinradarsinyal"},
    "trend_analysis_wait": {"en": "Performing trend analysis, please wait...", "tr": "Trend analizi yapılıyor, lütfen bekleyiniz..."},
    "chart_wait": {"en": "Generating chart, please wait...", "tr": "Grafik oluşturuluyor, lütfen bekleyiniz..."},
    "chart_usage": {"en": "Usage: /chart <symbol> [timeframe]. Example: /chart BTCUSDT 4h", "tr": "Kullanım: /chart <sembol> [zaman dilimi]. Örnek: /chart BTCUSDT 4h"},
    "chart_error": {"en": "Chart could not be generated. Please check coin symbol and timeframe.", "tr": "Grafik oluşturulamadı. Lütfen coin sembolünü ve zaman dilimini kontrol ediniz."},
    "setfavorites_usage": {"en": "Usage: /setfavorites BTCUSDT,ETHUSDT,...", "tr": "Kullanım: /setfavorites BTCUSDT,ETHUSDT,..."},
    "favorites_set": {"en": "Favorites set: ", "tr": "Favoriler ayarlandı: "},
    "no_favorites": {"en": "No favorites set.", "tr": "Favori ayarlanmamış."},
    "your_favorites": {"en": "Your favorites: ", "tr": "Favorileriniz: "},
    "setrisk_usage": {"en": "Usage: /setrisk <risk_percentage>", "tr": "Kullanım: /setrisk <risk_yüzdesi>"},
    "risk_set": {"en": "Risk setting set to ", "tr": "Risk ayarı "},
    "invalid_risk": {"en": "Invalid risk value.", "tr": "Geçersiz risk değeri."},
    "no_risk": {"en": "No risk setting found.", "tr": "Risk ayarı bulunamadı."},
    "your_risk": {"en": "Your risk setting: ", "tr": "Risk ayarınız: "},
    "specify_coin": {"en": "Please specify a coin, e.g. /coin BTCUSDT", "tr": "Lütfen bir coin belirtin, örn. /coin BTCUSDT"},
    "no_signal": {"en": "No Clear Signal at the moment", "tr": "Şu anda net bir sinyal yok"},
    "risk_management": {"en": "Risk Management:", "tr": "Risk Yönetimi:"},
    "realtime_usage": {"en": "Usage: /realtime <symbol>", "tr": "Kullanım: /realtime <sembol>"},
    "connecting_realtime": {"en": "Connecting to real-time data for ", "tr": "Gerçek zamanlı veriye bağlanılıyor: "},
    "error_realtime": {"en": "Error connecting to real-time data.", "tr": "Gerçek zamanlı veriye bağlanırken hata oluştu."},
    "adv_analysis_wait": {"en": "Performing advanced analysis for ", "tr": "Gelişmiş analiz yapılıyor: "},
    "fetching_news": {"en": "Fetching latest market news...", "tr": "En güncel piyasa haberleri getiriliyor..."},
    "no_news": {"en": "Could not fetch news at this time.", "tr": "Şu anda haberler getirilemedi."},
    "trade_signal": {"en": "Trade Signal", "tr": "Ticaret Sinyali"},
    "coin": {"en": "Coin", "tr": "Coin"},
    "direction": {"en": "Direction", "tr": "Yön"},
    "entry": {"en": "Entry", "tr": "Giriş"},
    "take_profit": {"en": "Take Profit", "tr": "Kar Al"},
    "stop_loss": {"en": "Stop Loss", "tr": "Zarar Durdur"},
    "long": {"en": "LONG", "tr": "UZUN"},
    "short": {"en": "SHORT", "tr": "KISA"}
    # Diğer çeviri anahtarlarını ekleyin...
}

def t(key, lang):
    return translations.get(key, {}).get(lang, "")
