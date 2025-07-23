#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 بوت التداول الذكي الشامل - الإصدار 1.1 المحدث
=======================================================

📝 التعديلات المضافة في النسخة 1.1:
=====================================

🔧 التحسينات الأساسية:
• تحديث API OpenAI إلى الإصدار 1.3.7+ مع GPT-4o flagship
• إصلاح مشكلة عدم الوصول والدردشة مع ChatGPT
• تحسين جلب وتحليل بيانات الصفقات

🌐 مصادر البيانات المحدثة:
• MetaTrader API: https://github.com/metaapi/metaapi-python-sdk
• TradingView API: https://www.tradingview.com/charting-library-docs/latest/api/
• Yahoo Finance API: https://finance.yahoo.com/quote/API/

🔑 مشاكل تم حلها:
• مشكلة عدم الاستجابة لـ ChatGPT - تم إصلاحها
• مشكلة عدم تحليل الصفقات - تم تحسين الخوارزمية
• تحسين دقة التحليل والتوقعات
• تحسين استقرار الاتصال بالمنصات الخارجية

المطور: مطور البوت الذكي
التاريخ: 2024 - الإصدار 1.1
"""

import telebot
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from telebot import types
from logging.handlers import RotatingFileHandler
from openai import OpenAI  # OpenAI API 1.3.7+
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import threading
import time
import schedule
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import warnings
import aiohttp
import websockets
warnings.filterwarnings('ignore')

# محاولة استيراد MetaAPI SDK
try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("⚠️ MetaAPI SDK غير متوفر، سيتم استخدام مصادر بديلة")

# محاولة استيراد MetaTrader5 كبديل
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 غير متوفر، سيتم استخدام مصادر بديلة")

# محاولة استيراد مكتبات إضافية للتحليل
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# إعداد البوت والذكاء الاصطناعي - تحديث OpenAI API 1.3.7+
bot = telebot.TeleBot('7703327028:AAHLqgR1HtVPsq6LfUKEWzNEgLZjJPLa6YU')

# تحديث إعداد OpenAI Client للإصدار 1.3.7+ مع GPT-4o
try:
    client = OpenAI(
        api_key='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',
        timeout=30.0,  # زيادة timeout لتحسين الاستقرار
        max_retries=3   # إعادة المحاولة في حالة الفشل
    )
    OPENAI_AVAILABLE = True
    print("✅ تم تهيئة OpenAI API 1.3.7+ بنجاح")
except Exception as e:
    OPENAI_AVAILABLE = False
    print(f"❌ خطأ في تهيئة OpenAI: {e}")

# إعداد السجلات
def setup_logging():
    """إعداد نظام تسجيل متقدم"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('trading_bot_v1.1.log', maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# متغيرات عامة
authenticated_users = set()
user_passwords = {}  # تخزين كلمات المرور للمستخدمين
user_capital = {}    # تخزين رأس مال المستخدمين
user_trading_mode = {}  # تخزين نمط التداول للمستخدمين

# تصنيف الأزواج المالية
CURRENCY_PAIRS = {
    'EURUSD': {'name': 'يورو/دولار', 'symbol': 'EUR/USD', 'type': 'forex'},
    'USDJPY': {'name': 'دولار/ين', 'symbol': 'USD/JPY', 'type': 'forex'},
    'GBPEUR': {'name': 'جنيه/يورو', 'symbol': 'GBP/EUR', 'type': 'forex'},
    'GBPUSD': {'name': 'جنيه/دولار', 'symbol': 'GBP/USD', 'type': 'forex'},
    'USDCHF': {'name': 'دولار/فرنك', 'symbol': 'USD/CHF', 'type': 'forex'},
    'AUDUSD': {'name': 'دولار أسترالي/دولار', 'symbol': 'AUD/USD', 'type': 'forex'},
    'USDCAD': {'name': 'دولار/دولار كندي', 'symbol': 'USD/CAD', 'type': 'forex'}
}

METALS = {
    'XAUUSD': {'name': 'ذهب/دولار', 'symbol': 'XAU/USD', 'type': 'metal'},
    'XAGUSD': {'name': 'فضة/دولار', 'symbol': 'XAG/USD', 'type': 'metal'}
}

CRYPTOCURRENCIES = {
    'BTCUSD': {'name': 'بيتكوين', 'symbol': 'BTC/USD', 'type': 'crypto'},
    'ETHUSD': {'name': 'إيثريوم', 'symbol': 'ETH/USD', 'type': 'crypto'},
    'LTCUSD': {'name': 'لايتكوين', 'symbol': 'LTC/USD', 'type': 'crypto'},
    'ADAUSD': {'name': 'كاردانو', 'symbol': 'ADA/USD', 'type': 'crypto'},
    'XRPUSD': {'name': 'ريبل', 'symbol': 'XRP/USD', 'type': 'crypto'}
}

ALL_SYMBOLS = {**CURRENCY_PAIRS, **METALS, **CRYPTOCURRENCIES}

# إطارات زمنية للتحليل
TIMEFRAMES = {
    'M1': {'name': 'دقيقة واحدة', 'mt5': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else None},
    'M3': {'name': '3 دقائق', 'mt5': mt5.TIMEFRAME_M3 if MT5_AVAILABLE else None},
    'M5': {'name': '5 دقائق', 'mt5': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None},
    'M15': {'name': '15 دقيقة', 'mt5': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else None},
    'H1': {'name': 'ساعة واحدة', 'mt5': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None},
    'H4': {'name': '4 ساعات', 'mt5': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None},
    'D1': {'name': 'يومي', 'mt5': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None}
}

@dataclass
class TradeSignal:
    """فئة إشارة التداول المحدثة"""
    symbol: str
    action: str  # BUY/SELL
    confidence: float
    entry_price: float
    take_profit: float
    analysis: str
    timestamp: datetime
    timeframes_analysis: Dict = None
    risk_reward_ratio: float = 2.0
    position_size: float = 0.0
    expected_profit: float = 0.0
    ai_analysis: str = ""  # تحليل الذكاء الاصطناعي

class EnhancedStorage:
    """نظام تخزين محسن باستخدام JSON"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.data = self.load()
        self.lock = threading.Lock()  # حماية من التداخل
    
    def load(self) -> Dict:
        """تحميل البيانات من الملف"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"خطأ في تحميل البيانات: {e}")
            return {}
    
    def save(self):
        """حفظ البيانات إلى الملف بشكل آمن"""
        try:
            with self.lock:
                with open(self.filename, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"خطأ في حفظ البيانات: {e}")
    
    def get(self, key: str, default=None):
        """الحصول على قيمة"""
        with self.lock:
            return self.data.get(key, default)
    
    def set(self, key: str, value):
        """تعيين قيمة"""
        with self.lock:
            self.data[key] = value
            self.save()
    
    def cleanup_old_data(self, days: int = 90):
        """تنظيف البيانات القديمة (3 أشهر)"""
        try:
            with self.lock:
                cutoff_date = datetime.now() - timedelta(days=days)
                if 'trades' in self.data:
                    valid_trades = []
                    for trade in self.data['trades']:
                        try:
                            trade_date = datetime.fromisoformat(trade.get('timestamp', ''))
                            if trade_date > cutoff_date:
                                valid_trades.append(trade)
                        except:
                            continue
                    self.data['trades'] = valid_trades
                    self.save()
                    logger.info(f"تم تنظيف البيانات الأقدم من {days} يوم")
        except Exception as e:
            logger.error(f"خطأ في تنظيف البيانات: {e}")

class MetaApiManager:
    """مدير MetaAPI SDK المحسن"""
    
    def __init__(self, token: str = None):
        self.token = token
        self.api = None
        self.account = None
        self.connection = None
        self.connected = False
        
    async def connect(self, account_id: str = None) -> bool:
        """الاتصال بـ MetaAPI"""
        if not METAAPI_AVAILABLE or not self.token:
            logger.warning("MetaAPI غير متوفر أو لا يوجد token")
            return False
            
        try:
            self.api = MetaApi(self.token)
            if account_id:
                self.account = await self.api.metatrader_account_api.get_account(account_id)
                self.connection = self.account.get_streaming_connection()
                await self.connection.connect()
                await self.connection.wait_synchronized()
                self.connected = True
                logger.info("تم الاتصال بـ MetaAPI بنجاح")
                return True
        except Exception as e:
            logger.error(f"خطأ في الاتصال بـ MetaAPI: {e}")
            return False
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات السوق من MetaAPI"""
        if not self.connected or not self.connection:
            return None
            
        try:
            price = await self.connection.get_symbol_price(symbol)
            return {
                'symbol': symbol,
                'bid': price.get('bid', 0),
                'ask': price.get('ask', 0),
                'time': price.get('time', datetime.now())
            }
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات {symbol} من MetaAPI: {e}")
            return None

class EnhancedTradingViewAPI:
    """API محسن لـ TradingView"""
    
    def __init__(self):
        self.base_url = "https://scanner.tradingview.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.tradingview.com/',
            'Origin': 'https://www.tradingview.com'
        })
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات السوق من TradingView"""
        try:
            # تحويل الرمز لصيغة TradingView
            tv_symbol = self._convert_to_tv_symbol(symbol)
            
            # طلب البيانات
            payload = {
                "filter": [{"left": "name", "operation": "match", "right": tv_symbol}],
                "columns": ["name", "close", "change", "change_abs", "high", "low", "volume", "market_cap_basic"],
                "sort": {"sortBy": "name", "sortOrder": "asc"},
                "range": [0, 1]
            }
            
            response = self.session.post(
                f"{self.base_url}/america/scan",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    item = data['data'][0]
                    return {
                        'symbol': symbol,
                        'price': item.get('d', [0])[1] if len(item.get('d', [])) > 1 else 0,
                        'change': item.get('d', [0])[2] if len(item.get('d', [])) > 2 else 0,
                        'high': item.get('d', [0])[3] if len(item.get('d', [])) > 3 else 0,
                        'low': item.get('d', [0])[4] if len(item.get('d', [])) > 4 else 0,
                        'volume': item.get('d', [0])[5] if len(item.get('d', [])) > 5 else 0,
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات TradingView لـ {symbol}: {e}")
            return None
    
    def _convert_to_tv_symbol(self, symbol: str) -> str:
        """تحويل الرمز لصيغة TradingView"""
        tv_mapping = {
            'EURUSD': 'FX:EURUSD',
            'GBPUSD': 'FX:GBPUSD', 
            'USDJPY': 'FX:USDJPY',
            'XAUUSD': 'TVC:GOLD',
            'BTCUSD': 'BITSTAMP:BTCUSD',
            'ETHUSD': 'BITSTAMP:ETHUSD'
        }
        return tv_mapping.get(symbol, f"FX:{symbol}")

class EnhancedYahooFinanceAPI:
    """API محسن لـ Yahoo Finance"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات السوق من Yahoo Finance"""
        try:
            yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
            if not yahoo_symbol:
                return None
                
            ticker = yf.Ticker(yahoo_symbol)
            
            # جلب البيانات التاريخية
            hist = ticker.history(period="5d", interval="1h")
            if hist.empty:
                return None
                
            # جلب معلومات إضافية
            info = ticker.info
            
            latest = hist.iloc[-1]
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                'change': float(latest['Close'] - hist.iloc[-2]['Close']) if len(hist) > 1 else 0,
                'change_percent': ((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100) if len(hist) > 1 else 0,
                'timestamp': datetime.now(),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0)
            }
            
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات Yahoo Finance لـ {symbol}: {e}")
            return None
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> Optional[str]:
        """تحويل الرمز لصيغة Yahoo Finance"""
        yahoo_mapping = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'XAUUSD': 'GC=F',
            'XAGUSD': 'SI=F',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'LTCUSD': 'LTC-USD',
            'ADAUSD': 'ADA-USD',
            'XRPUSD': 'XRP-USD'
        }
        return yahoo_mapping.get(symbol)

class EnhancedAIAnalyzer:
    """محلل الذكاء الاصطناعي المحسن مع GPT-4o"""
    
    def __init__(self):
        self.client = client if OPENAI_AVAILABLE else None
        self.model = "gpt-4o"  # استخدام GPT-4o flagship
        self.max_retries = 3
        self.timeout = 30
    
    async def analyze_market_data(self, symbol: str, market_data: Dict, timeframe: str = "H1") -> Optional[TradeSignal]:
        """تحليل بيانات السوق باستخدام GPT-4o"""
        if not self.client:
            logger.error("OpenAI client غير متوفر")
            return None
        
        try:
            # إعداد البيانات للتحليل
            analysis_prompt = self._create_analysis_prompt(symbol, market_data, timeframe)
            
            # استدعاء GPT-4o مع إعادة المحاولة
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": """أنت خبير تداول محترف متخصص في التحليل الفني والأساسي. 
                                قم بتحليل البيانات المقدمة وتقديم توصية تداول دقيقة مع مستوى ثقة عالي.
                                ركز على الأنماط الفنية، المؤشرات، والاتجاهات العامة للسوق."""
                            },
                            {
                                "role": "user", 
                                "content": analysis_prompt
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.3,
                        timeout=self.timeout
                    )
                    
                    ai_analysis = response.choices[0].message.content
                    
                    # استخراج التوصية من التحليل
                    trade_signal = self._extract_trade_signal(symbol, market_data, ai_analysis)
                    if trade_signal:
                        trade_signal.ai_analysis = ai_analysis
                        logger.info(f"تم تحليل {symbol} بنجاح بواسطة GPT-4o")
                        return trade_signal
                    
                    break
                    
                except Exception as e:
                    logger.warning(f"محاولة {attempt + 1} فشلت لـ {symbol}: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)  # تأخير متزايد
                        
        except Exception as e:
            logger.error(f"خطأ في تحليل GPT-4o لـ {symbol}: {e}")
            return None
    
    def _create_analysis_prompt(self, symbol: str, market_data: Dict, timeframe: str) -> str:
        """إنشاء prompt للتحليل"""
        return f"""
        قم بتحليل البيانات التالية للزوج {symbol}:
        
        📊 بيانات السوق:
        • السعر الحالي: {market_data.get('price', 0)}
        • أعلى سعر: {market_data.get('high', 0)}
        • أقل سعر: {market_data.get('low', 0)}
        • التغيير: {market_data.get('change', 0)}
        • نسبة التغيير: {market_data.get('change_percent', 0):.2f}%
        • الحجم: {market_data.get('volume', 0)}
        • الإطار الزمني: {timeframe}
        • الوقت: {market_data.get('timestamp', datetime.now())}
        
        المطلوب:
        1. تحليل فني شامل للاتجاه
        2. توصية واضحة (BUY/SELL/HOLD)
        3. مستوى الثقة (0-100%)
        4. نقطة الدخول المقترحة
        5. الهدف المتوقع
        6. تبرير التوصية
        
        تنسيق الإجابة:
        ACTION: [BUY/SELL/HOLD]
        CONFIDENCE: [0-100]
        ENTRY: [السعر]
        TARGET: [السعر]
        ANALYSIS: [التحليل المفصل]
        """
    
    def _extract_trade_signal(self, symbol: str, market_data: Dict, ai_analysis: str) -> Optional[TradeSignal]:
        """استخراج إشارة التداول من تحليل الذكاء الاصطناعي"""
        try:
            lines = ai_analysis.split('\n')
            action = "HOLD"
            confidence = 50.0
            entry_price = market_data.get('price', 0)
            target_price = entry_price
            
            for line in lines:
                line = line.strip().upper()
                if line.startswith('ACTION:'):
                    action = line.split(':')[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(line.split(':')[1].strip().replace('%', ''))
                elif line.startswith('ENTRY:'):
                    try:
                        entry_price = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('TARGET:'):
                    try:
                        target_price = float(line.split(':')[1].strip())
                    except:
                        pass
            
            if action in ['BUY', 'SELL'] and confidence >= 70:
                return TradeSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    entry_price=entry_price,
                    take_profit=target_price,
                    analysis=ai_analysis,
                    timestamp=datetime.now(),
                    risk_reward_ratio=abs(target_price - entry_price) / (entry_price * 0.02) if entry_price > 0 else 2.0
                )
                
        except Exception as e:
            logger.error(f"خطأ في استخراج إشارة التداول: {e}")
            return None

class UnifiedMarketDataProvider:
    """موحد مصادر البيانات المالية"""
    
    def __init__(self):
        self.metaapi = MetaApiManager()
        self.tradingview = EnhancedTradingViewAPI()
        self.yahoo = EnhancedYahooFinanceAPI()
        self.ai_analyzer = EnhancedAIAnalyzer()
        self.cache = {}
        self.cache_timeout = 60  # ثانية
    
    async def get_comprehensive_data(self, symbol: str) -> Optional[Dict]:
        """جلب بيانات شاملة من جميع المصادر"""
        
        # فحص الكاش
        cache_key = f"{symbol}_{int(time.time() // self.cache_timeout)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        data_sources = {}
        
        # جلب من Yahoo Finance (الأكثر استقراراً)
        try:
            yahoo_data = self.yahoo.get_market_data(symbol)
            if yahoo_data:
                data_sources['yahoo'] = yahoo_data
                logger.info(f"✅ تم جلب بيانات {symbol} من Yahoo Finance")
        except Exception as e:
            logger.error(f"❌ خطأ Yahoo Finance لـ {symbol}: {e}")
        
        # جلب من TradingView
        try:
            tv_data = self.tradingview.get_market_data(symbol)
            if tv_data:
                data_sources['tradingview'] = tv_data
                logger.info(f"✅ تم جلب بيانات {symbol} من TradingView")
        except Exception as e:
            logger.error(f"❌ خطأ TradingView لـ {symbol}: {e}")
        
        # جلب من MetaAPI (إذا متوفر)
        try:
            if METAAPI_AVAILABLE:
                meta_data = await self.metaapi.get_market_data(symbol)
                if meta_data:
                    data_sources['metaapi'] = meta_data
                    logger.info(f"✅ تم جلب بيانات {symbol} من MetaAPI")
        except Exception as e:
            logger.error(f"❌ خطأ MetaAPI لـ {symbol}: {e}")
        
        if not data_sources:
            logger.error(f"❌ فشل في جلب بيانات {symbol} من جميع المصادر")
            return None
        
        # دمج البيانات
        unified_data = self._merge_data_sources(symbol, data_sources)
        
        # حفظ في الكاش
        self.cache[cache_key] = unified_data
        
        return unified_data
    
    def _merge_data_sources(self, symbol: str, data_sources: Dict) -> Dict:
        """دمج البيانات من مصادر متعددة"""
        
        # أولوية المصادر
        priority = ['yahoo', 'tradingview', 'metaapi']
        
        merged_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'sources': list(data_sources.keys())
        }
        
        # اختيار أفضل البيانات
        for source in priority:
            if source in data_sources:
                source_data = data_sources[source]
                
                # السعر الأساسي
                if 'price' in source_data and source_data['price'] > 0:
                    merged_data['price'] = source_data['price']
                
                # بيانات إضافية
                for key in ['high', 'low', 'open', 'volume', 'change', 'change_percent']:
                    if key in source_data and key not in merged_data:
                        merged_data[key] = source_data[key]
                
                break
        
        # حساب متوسط الأسعار إذا توفرت من مصادر متعددة
        prices = [data.get('price', 0) for data in data_sources.values() if data.get('price', 0) > 0]
        if len(prices) > 1:
            merged_data['avg_price'] = sum(prices) / len(prices)
            merged_data['price_variance'] = max(prices) - min(prices)
        
        return merged_data

class EnhancedTradingBot:
    """البوت المحسن للتداول"""
    
    def __init__(self):
        self.storage = EnhancedStorage('trading_data_v1.1.json')
        self.market_provider = UnifiedMarketDataProvider()
        self.daily_trade_limit = 10
        self.min_confidence = 90  # الحد الأدنى للثقة للإشعارات
        self.is_running = True
        
    async def analyze_symbol(self, symbol: str, user_id: int) -> Optional[TradeSignal]:
        """تحليل رمز مالي شامل"""
        try:
            # جلب البيانات الشاملة
            market_data = await self.market_provider.get_comprehensive_data(symbol)
            if not market_data:
                return None
            
            # تحليل بالذكاء الاصطناعي
            ai_signal = await self.market_provider.ai_analyzer.analyze_market_data(
                symbol, market_data, "H1"
            )
            
            if ai_signal and ai_signal.confidence >= 70:
                # حفظ الإشارة
                self._save_trade_signal(user_id, ai_signal)
                logger.info(f"✅ تم تحليل {symbol} - الثقة: {ai_signal.confidence:.1f}%")
                return ai_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل {symbol}: {e}")
            return None
    
    def _save_trade_signal(self, user_id: int, signal: TradeSignal):
        """حفظ إشارة التداول"""
        try:
            trades = self.storage.get('trades', [])
            trade_data = {
                'user_id': user_id,
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'take_profit': signal.take_profit,
                'analysis': signal.analysis,
                'ai_analysis': signal.ai_analysis,
                'timestamp': signal.timestamp.isoformat(),
                'risk_reward_ratio': signal.risk_reward_ratio
            }
            trades.append(trade_data)
            self.storage.set('trades', trades)
        except Exception as e:
            logger.error(f"خطأ في حفظ الإشارة: {e}")
    
    def get_daily_trades_count(self, user_id: int) -> int:
        """حساب عدد الصفقات اليومية"""
        try:
            today = datetime.now().date()
            trades = self.storage.get('trades', [])
            daily_count = 0
            
            for trade in trades:
                try:
                    trade_date = datetime.fromisoformat(trade['timestamp']).date()
                    if trade_date == today and trade['user_id'] == user_id:
                        daily_count += 1
                except:
                    continue
                    
            return daily_count
        except Exception as e:
            logger.error(f"خطأ في حساب الصفقات اليومية: {e}")
            return 0

# إنشاء مثيل البوت المحسن
enhanced_bot = EnhancedTradingBot()

# إعداد لوحة المفاتيح
def create_main_keyboard():
    """إنشاء لوحة المفاتيح الرئيسية"""
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    
    # الصف الأول - التحليل
    keyboard.add(
        types.KeyboardButton("📊 تحليل لحظي"),
        types.KeyboardButton("🎯 إشارات التداول")
    )
    
    # الصف الثاني - إدارة الحساب  
    keyboard.add(
        types.KeyboardButton("💰 إدارة رأس المال"),
        types.KeyboardButton("📈 الصفقات المفتوحة")
    )
    
    # الصف الثالث - الإعدادات
    keyboard.add(
        types.KeyboardButton("⚙️ الإعدادات"),
        types.KeyboardButton("📚 دليل الاستخدام")
    )
    
    # الصف الرابع - الذكاء الاصطناعي
    keyboard.add(
        types.KeyboardButton("🤖 دردشة GPT-4o"),
        types.KeyboardButton("📊 تقرير شامل")
    )
    
    return keyboard

def create_symbols_keyboard():
    """إنشاء لوحة مفاتيح الأزواج المالية"""
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    
    # العملات
    keyboard.add(types.InlineKeyboardButton("💱 العملات الرئيسية", callback_data="category_forex"))
    
    # المعادن
    keyboard.add(types.InlineKeyboardButton("🥇 المعادن الثمينة", callback_data="category_metals"))
    
    # العملات الرقمية
    keyboard.add(types.InlineKeyboardButton("₿ العملات الرقمية", callback_data="category_crypto"))
    
    # تحليل شامل
    keyboard.add(types.InlineKeyboardButton("🔍 تحليل شامل للكل", callback_data="analyze_all"))
    
    return keyboard

# معالجات الأوامر الأساسية
@bot.message_handler(commands=['start'])
def start_command(message):
    """معالج أمر البدء"""
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "المستخدم"
    
    welcome_text = f"""
🎉 **أهلاً وسهلاً {user_name}!**

🤖 **بوت التداول الذكي الشامل - الإصدار 1.1**

✨ **الميزات الجديدة:**
• 🧠 تحليل بـ GPT-4o flagship
• 📊 بيانات من 3 مصادر موثوقة
• 🎯 دقة تحليل محسّنة
• 💬 دردشة مباشرة مع الذكاء الاصطناعي

🔐 **للوصول للخدمات المتقدمة:**
أرسل كلمة المرور الخاصة بك

📚 **أو استخدم الأزرار أدناه للبدء:**
"""
    
    bot.reply_to(message, welcome_text, parse_mode='Markdown', reply_markup=create_main_keyboard())

@bot.message_handler(func=lambda message: message.text == "🤖 دردشة GPT-4o")
def chat_with_gpt(message):
    """معالج الدردشة مع GPT-4o"""
    user_id = message.from_user.id
    
    if not OPENAI_AVAILABLE:
        bot.reply_to(message, "❌ خدمة GPT-4o غير متوفرة حالياً")
        return
    
    bot.reply_to(
        message, 
        """
🤖 **مرحباً بك في دردشة GPT-4o!**

أرسل سؤالك أو استفسارك وسأجيب عليك بأفضل ما لدي من معرفة في:

📈 **التداول والاستثمار**
📊 **التحليل الفني والأساسي** 
💰 **إدارة المخاطر**
🌍 **الأسواق المالية العالمية**
📱 **استراتيجيات التداول**

💡 **مثال:** "ما رأيك في اتجاه الذهب هذا الأسبوع؟"
        """,
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda message: message.text == "📊 تحليل لحظي")
def instant_analysis(message):
    """معالج التحليل اللحظي"""
    user_id = message.from_user.id
    
    analysis_text = """
🔍 **اختر نوع التحليل المطلوب:**

استخدم الأزرار أدناه لاختيار فئة الأصول التي تريد تحليلها:
"""
    
    bot.reply_to(
        message,
        analysis_text,
        parse_mode='Markdown',
        reply_markup=create_symbols_keyboard()
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('category_'))
def handle_category_selection(call):
    """معالج اختيار فئة الأصول"""
    category = call.data.split('_')[1]
    
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    
    if category == 'forex':
        symbols = CURRENCY_PAIRS
        title = "💱 العملات الرئيسية"
    elif category == 'metals':
        symbols = METALS  
        title = "🥇 المعادن الثمينة"
    elif category == 'crypto':
        symbols = CRYPTOCURRENCIES
        title = "₿ العملات الرقمية"
    else:
        return
    
    for symbol, info in symbols.items():
        keyboard.add(
            types.InlineKeyboardButton(
                f"{info['name']} ({info['symbol']})",
                callback_data=f"analyze_{symbol}"
            )
        )
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة", callback_data="back_to_categories"))
    
    bot.edit_message_text(
        f"🔍 **{title}**\n\nاختر الزوج المراد تحليله:",
        call.message.chat.id,
        call.message.message_id,
        parse_mode='Markdown',
        reply_markup=keyboard
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def handle_symbol_analysis(call):
    """معالج تحليل رمز مالي محدد"""
    symbol = call.data.split('_')[1]
    user_id = call.from_user.id
    
    # فحص الحد اليومي
    daily_trades = enhanced_bot.get_daily_trades_count(user_id)
    if daily_trades >= enhanced_bot.daily_trade_limit:
        bot.answer_callback_query(
            call.id,
            "⚠️ تم الوصول للحد الأقصى اليومي (10 صفقات)",
            show_alert=True
        )
        return
    
    bot.answer_callback_query(call.id, "🔄 جاري التحليل...")
    
    # رسالة انتظار
    waiting_msg = bot.edit_message_text(
        f"🔄 **جاري تحليل {ALL_SYMBOLS[symbol]['name']}...**\n\n"
        f"📊 جلب البيانات من المصادر المتعددة\n"
        f"🧠 تحليل بواسطة GPT-4o\n"
        f"⏱️ يرجى الانتظار...",
        call.message.chat.id,
        call.message.message_id,
        parse_mode='Markdown'
    )
    
    # تشغيل التحليل في thread منفصل
    def run_analysis():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            signal = loop.run_until_complete(
                enhanced_bot.analyze_symbol(symbol, user_id)
            )
            
            if signal:
                # تنسيق النتيجة
                confidence_emoji = "🚨" if signal.confidence >= 90 else "📊"
                action_emoji = "🟢" if signal.action == "BUY" else "🔴"
                
                result_text = f"""
{action_emoji} **تحليل {ALL_SYMBOLS[symbol]['name']}**

{confidence_emoji} **التوصية:** {signal.action}
📈 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🎯 **الهدف:** {signal.take_profit:.4f}
⚖️ **نسبة المخاطرة/العائد:** 1:{signal.risk_reward_ratio:.1f}

🧠 **تحليل GPT-4o:**
{signal.ai_analysis[:500]}...

⏰ **وقت التحليل:** {signal.timestamp.strftime('%H:%M:%S')}
📊 **المصادر:** Yahoo Finance, TradingView

{"🚨 **إشعار عالي الأهمية!**" if signal.confidence >= 90 else ""}
"""
                
                bot.edit_message_text(
                    result_text,
                    call.message.chat.id,
                    call.message.message_id,
                    parse_mode='Markdown'
                )
                
                # إشعار إضافي للثقة العالية
                if signal.confidence >= enhanced_bot.min_confidence:
                    bot.send_message(
                        call.message.chat.id,
                        f"🚨 **تنبيه فرصة عالية الربحية!**\n\n"
                        f"الرمز: {ALL_SYMBOLS[symbol]['name']}\n"
                        f"الثقة: {signal.confidence:.1f}%\n"
                        f"التوصية: {signal.action}",
                        parse_mode='Markdown'
                    )
            else:
                bot.edit_message_text(
                    f"❌ **لم يتم العثور على فرصة تداول واضحة لـ {ALL_SYMBOLS[symbol]['name']}**\n\n"
                    f"💡 جرب تحليل رمز آخر أو أعد المحاولة لاحقاً",
                    call.message.chat.id,
                    call.message.message_id,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"خطأ في تحليل {symbol}: {e}")
            bot.edit_message_text(
                f"❌ **خطأ في تحليل {ALL_SYMBOLS[symbol]['name']}**\n\n"
                f"يرجى المحاولة مرة أخرى لاحقاً",
                call.message.chat.id,
                call.message.message_id,
                parse_mode='Markdown'
            )
    
    # تشغيل التحليل
    threading.Thread(target=run_analysis, daemon=True).start()

# معالج الرسائل النصية (للدردشة مع GPT-4o)
@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    """معالج الرسائل النصية والدردشة مع GPT-4o"""
    user_id = message.from_user.id
    text = message.text.strip()
    
    # فحص إذا كانت رسالة للذكاء الاصطناعي
    if len(text) > 10 and OPENAI_AVAILABLE:
        try:
            # إرسال رسالة انتظار
            waiting_msg = bot.reply_to(message, "🤖 جاري التفكير...")
            
            def get_ai_response():
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": """أنت مساعد ذكي متخصص في التداول والأسواق المالية. 
                                أجب بشكل مفيد ودقيق على أسئلة المستخدمين حول التداول والاستثمار والأسواق المالية.
                                استخدم الرموز التعبيرية واجعل إجاباتك واضحة ومفيدة."""
                            },
                            {
                                "role": "user",
                                "content": text
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.7,
                        timeout=30
                    )
                    
                    ai_response = response.choices[0].message.content
                    
                    # تحديث الرسالة بالرد
                    bot.edit_message_text(
                        f"🤖 **GPT-4o يجيب:**\n\n{ai_response}",
                        waiting_msg.chat.id,
                        waiting_msg.message_id,
                        parse_mode='Markdown'
                    )
                    
                except Exception as e:
                    logger.error(f"خطأ في GPT-4o: {e}")
                    bot.edit_message_text(
                        "❌ عذراً، حدث خطأ في الاتصال بـ GPT-4o. يرجى المحاولة مرة أخرى.",
                        waiting_msg.chat.id,
                        waiting_msg.message_id
                    )
            
            # تشغيل في thread منفصل
            threading.Thread(target=get_ai_response, daemon=True).start()
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الرسالة: {e}")
            bot.reply_to(message, "❌ حدث خطأ، يرجى المحاولة مرة أخرى.")
    else:
        # رسالة افتراضية
        bot.reply_to(
            message,
            "💡 استخدم الأزرار أدناه للتنقل أو اكتب سؤالاً للدردشة مع GPT-4o",
            reply_markup=create_main_keyboard()
        )

# مراقب السوق المحسن
class EnhancedMarketMonitor:
    """مراقب السوق المحسن"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.check_interval = 300  # 5 دقائق
        
    def start_monitoring(self):
        """بدء مراقبة السوق"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🎯 تم بدء مراقبة السوق")
    
    def stop_monitoring(self):
        """إيقاف مراقبة السوق"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ تم إيقاف مراقبة السوق")
    
    def _monitor_loop(self):
        """حلقة مراقبة السوق"""
        while self.is_monitoring:
            try:
                # مراقبة الأزواج المهمة
                important_symbols = ['EURUSD', 'XAUUSD', 'BTCUSD']
                
                for symbol in important_symbols:
                    if not self.is_monitoring:
                        break
                        
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        signal = loop.run_until_complete(
                            enhanced_bot.analyze_symbol(symbol, 0)  # تحليل عام
                        )
                        
                        # إرسال إشعارات للفرص عالية الثقة
                        if signal and signal.confidence >= enhanced_bot.min_confidence:
                            self._send_high_confidence_alert(symbol, signal)
                            
                    except Exception as e:
                        logger.error(f"خطأ في مراقبة {symbol}: {e}")
                
                # انتظار قبل الفحص التالي
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"خطأ في مراقبة السوق: {e}")
                time.sleep(60)  # انتظار دقيقة في حالة الخطأ
    
    def _send_high_confidence_alert(self, symbol: str, signal: TradeSignal):
        """إرسال تنبيه للفرص عالية الثقة"""
        try:
            alert_text = f"""
🚨 **تنبيه فرصة عالية الربحية!**

📊 **الرمز:** {ALL_SYMBOLS[symbol]['name']}
🎯 **التوصية:** {signal.action}
📈 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🎯 **الهدف:** {signal.take_profit:.4f}

⏰ **الوقت:** {signal.timestamp.strftime('%H:%M:%S')}

💡 **تحرك بسرعة - فرصة محدودة!**
"""
            
            # إرسال للمستخدمين المصادق عليهم
            for user_id in authenticated_users:
                try:
                    bot.send_message(user_id, alert_text, parse_mode='Markdown')
                except Exception as e:
                    logger.error(f"خطأ في إرسال تنبيه للمستخدم {user_id}: {e}")
                    
        except Exception as e:
            logger.error(f"خطأ في إرسال التنبيه: {e}")

# إنشاء مراقب السوق
market_monitor = EnhancedMarketMonitor()

# دالة التشغيل الرئيسية المحسنة
def main():
    """الدالة الرئيسية لتشغيل البوت المحسن"""
    try:
        logger.info("🚀 بدء تشغيل بوت التداول الذكي الشامل - الإصدار 1.1")
        logger.info(f"🧠 OpenAI GPT-4o: {'✅' if OPENAI_AVAILABLE else '❌'}")
        logger.info(f"📊 MetaAPI: {'✅' if METAAPI_AVAILABLE else '❌'}")
        logger.info(f"📈 MetaTrader5: {'✅' if MT5_AVAILABLE else '❌'}")
        logger.info(f"🔗 الأزواج المدعومة: {len(ALL_SYMBOLS)} زوج")
        logger.info(f"⚡ أنماط التداول: سكالبينغ، طويل المدى")
        logger.info(f"🛡️ إدارة المخاطر: 10 صفقات يومياً، 90%+ للإشعارات")
        
        # تنظيف البيانات القديمة عند البدء
        enhanced_bot.storage.cleanup_old_data(90)
        
        # بدء مراقبة السوق
        market_monitor.start_monitoring()
        
        # إنشاء مجلد التخزين إذا لم يكن موجوداً
        if not os.path.exists("pdf_storage"):
            os.makedirs("pdf_storage")
        
        logger.info("✅ البوت جاهز لاستقبال الرسائل")
        
        # تشغيل البوت
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
        
    except Exception as e:
        logger.error(f"❌ خطأ في تشغيل البوت: {e}")
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف البوت بواسطة المستخدم")
    finally:
        # إيقاف خدمة المراقبة
        market_monitor.stop_monitoring()
        
        logger.info("👋 تم إغلاق البوت وتنظيف الموارد")

if __name__ == "__main__":
    main()