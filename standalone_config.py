"""
ملف تكوين للتشغيل المستقل (بدون Replit)
"""
import os
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env إذا كان موجوداً
load_dotenv()

class StandaloneConfig:
    """إعدادات للتشغيل المستقل على أي سيرفر"""
    
    # Telegram Bot
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    AUTHORIZED_USERS = [int(x.strip()) for x in os.getenv('AUTHORIZED_USERS', '').split(',') if x.strip().isdigit()]
    
    # OpenAI (اختياري - إذا لم يكن متوفراً سيعمل البوت بدون تحليل ذكي)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    USE_AI_ANALYSIS = bool(OPENAI_API_KEY)
    
    # MetaTrader 5 (اختياري)
    MT5_LOGIN = os.getenv('MT5_LOGIN', '')
    MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
    MT5_SERVER = os.getenv('MT5_SERVER', '')
    USE_MT5 = bool(MT5_LOGIN and MT5_PASSWORD and MT5_SERVER)
    
    # Risk Management
    MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '10'))
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '1000'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '70'))
    
    # Storage
    DATA_DIRECTORY = os.getenv('DATA_DIRECTORY', 'bot_data')
    USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true'
    
    # Trading Symbols
    TRADING_SYMBOLS = {
        'XAUUSD': 'GC=F',      # الذهب
        'EURUSD': 'EURUSD=X',  # اليورو دولار
        'GBPUSD': 'GBPUSD=X',  # الجنيه دولار
        'USDJPY': 'USDJPY=X',  # دولار ين
        'BTCUSD': 'BTC-USD'    # البيتكوين
    }
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_DIRECTORY = os.getenv('LOG_DIRECTORY', 'logs')
    
    @classmethod
    def validate_config(cls) -> bool:
        """التحقق من صحة الإعدادات الأساسية"""
        if not cls.TELEGRAM_BOT_TOKEN:
            print("❌ خطأ: TELEGRAM_BOT_TOKEN مطلوب")
            return False
        
        if not cls.AUTHORIZED_USERS:
            print("⚠️ تحذير: لا توجد مستخدمين مصرح لهم")
            return False
        
        if not cls.USE_AI_ANALYSIS:
            print("⚠️ تحذير: لن يعمل التحليل الذكي بدون OPENAI_API_KEY")
        
        return True
    
    @classmethod
    def show_config(cls):
        """عرض الإعدادات الحالية"""
        print("🔧 إعدادات البوت:")
        print(f"  📱 Telegram Bot: {'✅ متصل' if cls.TELEGRAM_BOT_TOKEN else '❌ غير متصل'}")
        print(f"  🤖 تحليل ذكي: {'✅ مفعل' if cls.USE_AI_ANALYSIS else '❌ معطل'}")
        print(f"  📊 MetaTrader 5: {'✅ مفعل' if cls.USE_MT5 else '❌ معطل (وضع محاكاة)'}")
        print(f"  👥 مستخدمين مصرح لهم: {len(cls.AUTHORIZED_USERS)}")
        print(f"  💾 تخزين البيانات: {'قاعدة بيانات' if cls.USE_DATABASE else 'ملفات JSON'}")
        print(f"  📁 مجلد البيانات: {cls.DATA_DIRECTORY}")

# استخدام الإعدادات المستقلة
Config = StandaloneConfig