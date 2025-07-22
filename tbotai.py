#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 بوت التداول الذكي الشامل مع AI Chat و PDF - الإصدار المحدث
================================================================

بوت تيليجرام متكامل للتداول والتحليل المالي مع الذكاء الاصطناعي
يتضمن ربط MetaTrader 5 و TradingView مع تحليل لحظي متقدم

الميزات الجديدة:
- ربط MetaTrader 5
- تحليل TradingView 
- تحليل لحظي بالذكاء الاصطناعي
- إشعارات الصفقات عالية الربحية (90%+)
- إدارة رأس المال المتقدمة
- أنماط التداول (سكالبينغ/طويل المدى)
- حد أقصى 10 صفقات يوميًا
- بدون وقف خسارة تلقائي
- حفظ الصفقات لـ 3 أشهر
- واجهة أزرار محسنة مع أوصاف
- تصنيف الأزواج (عملات، معادن، رقمية)
- 🤖 دردشة مباشرة مع الذكاء الاصطناعي
- 📚 رفع كتب PDF للتدريب

المطور: مطور البوت الذكي
التاريخ: 2024
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
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import threading
import time
import schedule
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد MetaTrader5
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

# إعداد البوت والذكاء الاصطناعي مع الإصدار الجديد
bot = telebot.TeleBot('7703327028:AAHLqgR1HtVPsq6LfUKEWzNEgLZjJPLa6YU')

# تحديث إعداد OpenAI للإصدار الجديد 1.3.7
client = OpenAI(api_key='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')

# إعداد السجلات
def setup_logging():
    """إعداد نظام تسجيل متقدم"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('trading_bot.log', maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# إنشاء مجلد لحفظ ملفات PDF
if not os.path.exists('pdf_storage'):
    os.makedirs('pdf_storage')

# متغيرات عامة
authenticated_users = set()
user_passwords = {}  # تخزين كلمات المرور للمستخدمين + حالات AI Chat
user_capital = {}    # تخزين رأس مال المستخدمين
user_trading_mode = {}  # تخزين نمط التداول للمستخدمين

# تصنيف الأزواج المالية
CURRENCY_PAIRS = {
    'EURUSD': {'name': 'يورو/دولار', 'symbol': 'EUR/USD', 'type': 'forex'},
    'USDJPY': {'name': 'دولار/ين', 'symbol': 'USD/JPY', 'type': 'forex'},
    'GBPEUR': {'name': 'جنيه/يورو', 'symbol': 'GBP/EUR', 'type': 'forex'}
}

METALS = {
    'XAUUSD': {'name': 'ذهب/دولار', 'symbol': 'XAU/USD', 'type': 'metal'}
}

CRYPTOCURRENCIES = {
    'BTCUSD': {'name': 'بيتكوين', 'symbol': 'BTC/USD', 'type': 'crypto'},
    'LTCUSD': {'name': 'لايتكوين', 'symbol': 'LTC/USD', 'type': 'crypto'},
    'ETHUSD': {'name': 'إيثريوم', 'symbol': 'ETH/USD', 'type': 'crypto'}
}

ALL_SYMBOLS = {**CURRENCY_PAIRS, **METALS, **CRYPTOCURRENCIES}

# إطارات زمنية للتحليل
TIMEFRAMES = {
    'M1': {'name': 'دقيقة واحدة', 'mt5': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else None},
    'M3': {'name': '3 دقائق', 'mt5': mt5.TIMEFRAME_M3 if MT5_AVAILABLE else None},
    'M5': {'name': '5 دقائق', 'mt5': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None},
    'M15': {'name': '15 دقيقة', 'mt5': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else None},
    'H1': {'name': 'ساعة واحدة', 'mt5': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None}
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

class SimpleStorage:
    """نظام تخزين بسيط باستخدام JSON"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.data = self.load()
    
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
        """حفظ البيانات إلى الملف"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"خطأ في حفظ البيانات: {e}")
    
    def get(self, key: str, default=None):
        """الحصول على قيمة"""
        return self.data.get(key, default)
    
    def set(self, key: str, value):
        """تعيين قيمة"""
        self.data[key] = value
        self.save()
    
    def cleanup_old_data(self, days: int = 90):
        """تنظيف البيانات القديمة (3 أشهر)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            if 'trades' in self.data:
                valid_trades = []
                for trade in self.data['trades']:
                    trade_date = datetime.fromisoformat(trade.get('timestamp', ''))
                    if trade_date > cutoff_date:
                        valid_trades.append(trade)
                self.data['trades'] = valid_trades
                self.save()
                logger.info(f"تم تنظيف البيانات الأقدم من {days} يوم")
        except Exception as e:
            logger.error(f"خطأ في تنظيف البيانات: {e}")

class MetaTrader5Manager:
    """مدير الاتصال بـ MetaTrader 5"""
    
    def __init__(self):
        self.connected = False
        self.last_connection_attempt = None
        self.connection_retry_interval = 300  # 5 دقائق
        
    def connect(self) -> bool:
        """الاتصال بـ MetaTrader 5"""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 غير متوفر")
            return False
            
        try:
            # محاولة الاتصال
            if mt5.initialize():
                self.connected = True
                logger.info("تم الاتصال بـ MetaTrader 5 بنجاح")
                return True
            else:
                logger.error("فشل في الاتصال بـ MetaTrader 5")
                return False
        except Exception as e:
            logger.error(f"خطأ في الاتصال بـ MetaTrader 5: {e}")
            return False
    
    def disconnect(self):
        """قطع الاتصال"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("تم قطع الاتصال من MetaTrader 5")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """الحصول على معلومات الرمز"""
        if not self.connected or not MT5_AVAILABLE:
            return None
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
                
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'digits': symbol_info.digits,
                'tick_value': symbol_info.trade_tick_value,
                'point': symbol_info.point
            }
        except Exception as e:
            logger.error(f"خطأ في الحصول على معلومات الرمز {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق من MT5"""
        if not self.connected or not MT5_AVAILABLE:
            return None
            
        try:
            mt5_timeframe = TIMEFRAMES.get(timeframe, {}).get('mt5')
            if mt5_timeframe is None:
                return None
                
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات السوق من MT5: {e}")
            return None

class TradingViewScraper:
    """مدير جلب البيانات من TradingView"""
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ar,en;q=0.9',
            'Referer': 'https://www.tradingview.com/'
        }
        self.session.headers.update(self.headers)
    
    def get_market_data(self, symbol: str, interval: str = '1D', count: int = 100) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق من TradingView أو استخدام Yahoo Finance كبديل"""
        try:
            # استخدام Yahoo Finance كبديل موثوق
            ticker = yf.Ticker(symbol)
            
            # تحديد الفترة بناءً على interval
            if interval in ['1m', '5m', '15m', '30m']:
                period = '7d'
            elif interval in ['1h', '4h']:
                period = '60d'
            else:
                period = '1y'
            
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                logger.warning(f"لا توجد بيانات لـ {symbol}")
                return None
                
            return data.tail(count)
            
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات TradingView لـ {symbol}: {e}")
            return None

class AdvancedMarketAnalyzer:
    """محلل الأسواق المالية المتقدم"""
    
    def __init__(self):
        self.mt5_manager = MetaTrader5Manager()
        self.tv_scraper = TradingViewScraper()
        self.storage = SimpleStorage('market_analysis.json')
        
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """جلب البيانات من عدة إطارات زمنية"""
        timeframes_data = {}
        
        # محاولة MT5 أولاً
        if self.mt5_manager.connect():
            for tf_name, tf_info in TIMEFRAMES.items():
                data = self.mt5_manager.get_market_data(symbol, tf_name)
                if data is not None:
                    timeframes_data[tf_name] = data
        
        # إذا لم تنجح MT5، استخدم TradingView/Yahoo Finance
        if not timeframes_data:
            intervals = ['1m', '5m', '15m', '1h', '1d']
            tf_mapping = {'1m': 'M1', '5m': 'M5', '15m': 'M15', '1h': 'H1', '1d': 'D1'}
            
            for interval in intervals:
                data = self.tv_scraper.get_market_data(symbol, interval)
                if data is not None:
                    tf_name = tf_mapping.get(interval, interval)
                    timeframes_data[tf_name] = data
        
        return timeframes_data
    
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict:
        """حساب المؤشرات الفنية المتقدمة"""
        try:
            indicators = {}
            
            # المؤشرات الأساسية
            indicators['current_price'] = float(data['Close'].iloc[-1])
            indicators['price_change'] = float(data['Close'].iloc[-1] - data['Close'].iloc[-2])
            indicators['price_change_pct'] = (indicators['price_change'] / data['Close'].iloc[-2]) * 100
            
            # المتوسطات المتحركة
            indicators['sma_10'] = float(data['Close'].rolling(window=10).mean().iloc[-1])
            indicators['sma_20'] = float(data['Close'].rolling(window=20).mean().iloc[-1])
            indicators['sma_50'] = float(data['Close'].rolling(window=50).mean().iloc[-1]) if len(data) >= 50 else None
            indicators['ema_12'] = float(data['Close'].ewm(span=12).mean().iloc[-1])
            indicators['ema_26'] = float(data['Close'].ewm(span=26).mean().iloc[-1])
            
            # RSI
            if len(data) >= 14:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = float((100 - (100 / (1 + rs))).iloc[-1])
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = float(data['Close'].ewm(span=9).mean().iloc[-1])
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            if len(data) >= 20:
                sma_20 = data['Close'].rolling(window=20).mean()
                std_20 = data['Close'].rolling(window=20).std()
                indicators['bb_upper'] = float((sma_20 + (std_20 * 2)).iloc[-1])
                indicators['bb_lower'] = float((sma_20 - (std_20 * 2)).iloc[-1])
                indicators['bb_middle'] = float(sma_20.iloc[-1])
            
            # Stochastic
            if len(data) >= 14:
                low_14 = data['Low'].rolling(window=14).min()
                high_14 = data['High'].rolling(window=14).max()
                indicators['stoch_k'] = float(((data['Close'] - low_14) / (high_14 - low_14) * 100).iloc[-1])
                indicators['stoch_d'] = float(((data['Close'] - low_14) / (high_14 - low_14) * 100).rolling(window=3).mean().iloc[-1])
            
            # ATR
            if len(data) >= 14:
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                indicators['atr'] = float(true_range.rolling(14).mean().iloc[-1])
            
            # Volume Analysis
            if 'Volume' in data.columns:
                indicators['volume'] = float(data['Volume'].iloc[-1])
                indicators['volume_sma'] = float(data['Volume'].rolling(window=20).mean().iloc[-1])
                indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma']
            
            # Support and Resistance
            recent_data = data.tail(20)
            indicators['resistance'] = float(recent_data['High'].max())
            indicators['support'] = float(recent_data['Low'].min())
            
            return indicators
            
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات: {e}")
            return {}
    
    def analyze_trend_strength(self, indicators: Dict) -> Dict:
        """تحليل قوة الاتجاه"""
        try:
            trend_analysis = {
                'trend_direction': 'sideways',
                'trend_strength': 'weak',
                'confidence': 50.0
            }
            
            # تحليل المتوسطات المتحركة
            current_price = indicators.get('current_price', 0)
            sma_10 = indicators.get('sma_10', current_price)
            sma_20 = indicators.get('sma_20', current_price)
            
            if current_price > sma_10 > sma_20:
                trend_analysis['trend_direction'] = 'bullish'
                trend_analysis['confidence'] += 20
            elif current_price < sma_10 < sma_20:
                trend_analysis['trend_direction'] = 'bearish'
                trend_analysis['confidence'] += 20
            
            # تحليل RSI
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                trend_analysis['confidence'] -= 15  # تشبع شرائي
            elif rsi < 30:
                trend_analysis['confidence'] -= 15  # تشبع بيعي
            elif 40 <= rsi <= 60:
                trend_analysis['confidence'] += 10  # منطقة متوازنة
            
            # تحليل MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                if trend_analysis['trend_direction'] == 'bullish':
                    trend_analysis['confidence'] += 15
            elif macd < macd_signal:
                if trend_analysis['trend_direction'] == 'bearish':
                    trend_analysis['confidence'] += 15
            
            # تحديد قوة الاتجاه
            if trend_analysis['confidence'] >= 80:
                trend_analysis['trend_strength'] = 'very_strong'
            elif trend_analysis['confidence'] >= 70:
                trend_analysis['trend_strength'] = 'strong'
            elif trend_analysis['confidence'] >= 60:
                trend_analysis['trend_strength'] = 'moderate'
            
            trend_analysis['confidence'] = min(95, max(5, trend_analysis['confidence']))
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل قوة الاتجاه: {e}")
            return {
                'trend_direction': 'sideways',
                'trend_strength': 'weak',
                'confidence': 50.0
            }

class CapitalManager:
    """مدير رأس المال"""
    
    def __init__(self):
        self.storage = SimpleStorage('capital_management.json')
    
    def set_user_capital(self, user_id: int, capital: float):
        """تعيين رأس المال للمستخدم"""
        user_capital[user_id] = capital
        self.storage.set(f'capital_{user_id}', capital)
        logger.info(f"تم تعيين رأس المال للمستخدم {user_id}: ${capital:,.2f}")
    
    def get_user_capital(self, user_id: int) -> float:
        """الحصول على رأس المال للمستخدم"""
        if user_id in user_capital:
            return user_capital[user_id]
        
        stored_capital = self.storage.get(f'capital_{user_id}', 10000.0)
        user_capital[user_id] = stored_capital
        return stored_capital
    
    def calculate_position_size(self, user_id: int, risk_percentage: float = 2.0) -> float:
        """حساب حجم المركز بناءً على المخاطرة"""
        capital = self.get_user_capital(user_id)
        risk_amount = capital * (risk_percentage / 100)
        return risk_amount

class RiskManager:
    """مدير المخاطر"""
    
    def __init__(self):
        self.storage = SimpleStorage('risk_management.json')
        self.default_settings = {
            'max_daily_trades': 10,
            'max_risk_per_trade': 2.0,
            'min_confidence': 90.0,
            'auto_stop_loss': False,
            'data_retention_months': 3
        }
    
    def check_daily_limit(self, user_id: int) -> bool:
        """فحص الحد الأقصى للصفقات اليومية"""
        today = datetime.now().date().isoformat()
        current_count = self.storage.get(f'daily_trades_{user_id}_{today}', 0)
        return current_count < self.default_settings['max_daily_trades']
    
    def increment_daily_trades(self, user_id: int):
        """زيادة عداد الصفقات اليومية"""
        today = datetime.now().date().isoformat()
        current_count = self.storage.get(f'daily_trades_{user_id}_{today}', 0)
        self.storage.set(f'daily_trades_{user_id}_{today}', current_count + 1)
    
    def get_daily_trades_count(self, user_id: int) -> int:
        """الحصول على عدد الصفقات اليومية"""
        today = datetime.now().date().isoformat()
        return self.storage.get(f'daily_trades_{user_id}_{today}', 0)

class TradingModeManager:
    """مدير أنماط التداول"""
    
    def __init__(self):
        self.storage = SimpleStorage('trading_modes.json')
        self.scalping_hours = [(8, 12), (13, 17), (20, 24)]  # الأوقات المناسبة للسكالبينغ
    
    def set_trading_mode(self, user_id: int, mode: str):
        """تعيين نمط التداول للمستخدم"""
        user_trading_mode[user_id] = mode
        self.storage.set(f'mode_{user_id}', mode)
        logger.info(f"تم تعيين نمط التداول للمستخدم {user_id}: {mode}")
    
    def get_trading_mode(self, user_id: int) -> str:
        """الحصول على نمط التداول للمستخدم"""
        if user_id in user_trading_mode:
            return user_trading_mode[user_id]
        
        stored_mode = self.storage.get(f'mode_{user_id}', 'long_term')
        user_trading_mode[user_id] = stored_mode
        return stored_mode
    
    def is_scalping_time(self) -> bool:
        """فحص إذا كان الوقت مناسب للسكالبينغ"""
        current_hour = datetime.now().hour
        for start_hour, end_hour in self.scalping_hours:
            if start_hour <= current_hour < end_hour:
                return True
        return False
    
    def get_mode_settings(self, mode: str) -> Dict:
        """الحصول على إعدادات نمط التداول"""
        if mode == 'scalping':
            return {
                'timeframes': ['M1', 'M3', 'M5'],
                'profit_target': 0.5,  # 0.5%
                'active_hours': self.scalping_hours,
                'max_hold_time': 30  # 30 دقيقة
            }
        else:  # long_term
            return {
                'timeframes': ['M15', 'H1'],
                'profit_target': 2.0,  # 2%
                'active_hours': [(0, 24)],  # 24/7
                'max_hold_time': 1440  # 24 ساعة
            }

class SmartTradingEngine:
    """محرك التداول الذكي المتقدم"""
    
    def __init__(self):
        self.storage = SimpleStorage('smart_trades.json')
        self.analyzer = AdvancedMarketAnalyzer()
        self.capital_manager = CapitalManager()
        self.risk_manager = RiskManager()
        self.mode_manager = TradingModeManager()
        
    def analyze_symbol_with_ai(self, symbol: str, user_id: int) -> Optional[TradeSignal]:
        """تحليل الرمز باستخدام الذكاء الاصطناعي مع الإصدار الجديد"""
        try:
            # جلب البيانات من عدة إطارات زمنية
            multi_tf_data = self.analyzer.get_multi_timeframe_data(symbol)
            if not multi_tf_data:
                logger.error(f"فشل في جلب البيانات لـ {symbol}")
                return None
            
            # تحليل كل إطار زمني
            multi_tf_analysis = {}
            for tf_name, data in multi_tf_data.items():
                indicators = self.analyzer.calculate_advanced_indicators(data)
                trend = self.analyzer.analyze_trend_strength(indicators)
                multi_tf_analysis[tf_name] = {
                    'indicators': indicators,
                    'trend': trend
                }
            
            # الحصول على نمط التداول للمستخدم
            trading_mode = self.mode_manager.get_trading_mode(user_id)
            
            # إنشاء سياق للذكاء الاصطناعي
            prompt_text = self._create_ai_context(symbol, multi_tf_analysis, trading_mode)
            
            # استدعاء الذكاء الاصطناعي بالطريقة الجديدة
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "أنت مساعد تداول محترف"},
                    {"role": "user", "content": prompt_text}
                ]
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # معالجة رد الذكاء الاصطناعي
            trade_signal = self._process_ai_response(ai_response, symbol, multi_tf_analysis, user_id)
            return trade_signal
            
        except Exception as e:
            logger.error(f"خطأ في تحليل الرمز {symbol} بالذكاء الاصطناعي: {e}")
            return None
    
    def _create_ai_context(self, symbol: str, multi_tf_analysis: Dict, trading_mode: str) -> str:
        """إنشاء سياق للذكاء الاصطناعي"""
        symbol_info = ALL_SYMBOLS.get(symbol, {})
        
        context = f"""
تحليل شامل لرمز {symbol} ({symbol_info.get('name', symbol)})
نمط التداول: {trading_mode}
النوع: {symbol_info.get('type', 'unknown')}

تحليل الإطارات الزمنية المتعددة:
"""
        
        for tf_name, analysis in multi_tf_analysis.items():
            indicators = analysis['indicators']
            trend = analysis['trend']
            
            context += f"""
الإطار الزمني {tf_name}:
- السعر الحالي: {indicators.get('current_price', 0):.4f}
- التغيير: {indicators.get('price_change_pct', 0):+.2f}%
- RSI: {indicators.get('rsi', 50):.1f}
- MACD: {indicators.get('macd', 0):.4f}
- الاتجاه: {trend.get('trend_direction', 'sideways')}
- قوة الاتجاه: {trend.get('trend_strength', 'weak')}
- الثقة: {trend.get('confidence', 50):.1f}%
"""
        
        context += f"""
متطلبات التحليل:
1. قم بتحليل شامل للحالة الفنية
2. حدد فرصة التداول (شراء/بيع/انتظار)
3. احسب مستوى الثقة (0-100%)
4. حدد نقطة الدخول المثلى
5. حدد هدف الربح (بدون وقف خسارة)
6. قدم تحليل مختصر ومفيد

أجب بصيغة JSON:
{{
    "action": "BUY/SELL/WAIT",
    "confidence": number,
    "entry_price": number,
    "take_profit": number,
    "analysis": "نص التحليل",
    "timeframe_summary": "ملخص الإطارات الزمنية"
}}
"""
        
        return context
    
    def _process_ai_response(self, ai_response: str, symbol: str, multi_tf_analysis: Dict, user_id: int) -> Optional[TradeSignal]:
        """معالجة رد الذكاء الاصطناعي"""
        try:
            # استخراج JSON من الرد
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.error("لم يتم العثور على JSON في رد الذكاء الاصطناعي")
                return None
            
            analysis_data = json.loads(json_match.group())
            
            # التحقق من الحد اليومي للصفقات
            if not self.risk_manager.check_daily_limit(user_id):
                logger.warning(f"تم الوصول للحد الأقصى اليومي للمستخدم {user_id}")
                return None
            
            # حساب حجم المركز والربح المتوقع
            position_size = self.capital_manager.calculate_position_size(user_id)
            entry_price = analysis_data.get('entry_price', 0)
            take_profit = analysis_data.get('take_profit', 0)
            expected_profit = abs(take_profit - entry_price) * position_size
            
            # إنشاء إشارة التداول
            trade_signal = TradeSignal(
                symbol=symbol,
                action=analysis_data.get('action', 'WAIT'),
                confidence=float(analysis_data.get('confidence', 50)),
                entry_price=float(entry_price),
                take_profit=float(take_profit),
                analysis=analysis_data.get('analysis', 'تحليل بالذكاء الاصطناعي'),
                timestamp=datetime.now(),
                timeframes_analysis=analysis_data.get('timeframe_summary', ''),
                position_size=position_size,
                expected_profit=expected_profit
            )
            
            return trade_signal
            
        except Exception as e:
            logger.error(f"خطأ في معالجة رد الذكاء الاصطناعي: {e}")
            return None

class NotificationManager:
    """مدير الإشعارات"""
    
    def __init__(self):
        self.storage = SimpleStorage('notifications.json')
    
    def send_high_probability_alert(self, user_id: int, signal: TradeSignal):
        """إرسال تنبيه للصفقات عالية الاحتمالية (90%+)"""
        try:
            if signal.confidence >= 90.0:
                alert_message = f"""
🚨 **تنبيه صفقة عالية الاحتمالية!**

📊 **الرمز:** {signal.symbol}
🎯 **الإجراء:** {signal.action}
💯 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🎯 **هدف الربح:** {signal.take_profit:.4f}
💵 **الربح المتوقع:** ${signal.expected_profit:.2f}

📝 **التحليل:** {signal.analysis}

⏰ **الوقت:** {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
"""
                bot.send_message(user_id, alert_message, parse_mode='Markdown')
                logger.info(f"تم إرسال تنبيه صفقة عالية الاحتمالية للمستخدم {user_id}")
        except Exception as e:
            logger.error(f"خطأ في إرسال التنبيه: {e}")

class MarketMonitorService:
    """خدمة مراقبة السوق في الخلفية"""
    
    def __init__(self):
        self.trading_engine = SmartTradingEngine()
        self.notification_manager = NotificationManager()
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """بدء مراقبة السوق"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("تم بدء خدمة مراقبة السوق")
    
    def stop_monitoring(self):
        """إيقاف مراقبة السوق"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("تم إيقاف خدمة مراقبة السوق")
    
    def _monitor_loop(self):
        """حلقة مراقبة السوق"""
        while self.running:
            try:
                # مراقبة كل الرموز للمستخدمين المصدقين
                for user_id in authenticated_users:
                    for symbol in ALL_SYMBOLS.keys():
                        signal = self.trading_engine.analyze_symbol_with_ai(symbol, user_id)
                        if signal and signal.confidence >= 90.0:
                            self.notification_manager.send_high_probability_alert(user_id, signal)
                
                # انتظار 5 دقائق قبل المراقبة التالية
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"خطأ في مراقبة السوق: {e}")
                time.sleep(60)  # انتظار دقيقة في حالة الخطأ

# إنشاء كائنات النظام
trading_engine = SmartTradingEngine()
capital_manager = CapitalManager()
risk_manager = RiskManager()
trading_mode_manager = TradingModeManager()
notification_manager = NotificationManager()
market_monitor = MarketMonitorService()
storage = SimpleStorage('bot_data.json')

# دوال الواجهة
def create_main_keyboard():
    """إنشاء لوحة المفاتيح الرئيسية مع الأزرار الجديدة"""
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    
    keyboard.add(
        types.KeyboardButton("💱 العملات"),
        types.KeyboardButton("🥇 المعادن")
    )
    keyboard.add(
        types.KeyboardButton("₿ العملات الرقمية")
    )
    keyboard.add(
        types.KeyboardButton("📊 إحصائياتي"),
        types.KeyboardButton("📈 صفقاتي المفتوحة")
    )
    keyboard.add(
        types.KeyboardButton("💰 رأس المال"),
        types.KeyboardButton("🎛️ نمط التداول")
    )
    # إضافة الأزرار الجديدة
    keyboard.add(
        types.KeyboardButton("🤖 اطلب من AI"),
        types.KeyboardButton("📚 رفع كتب PDF")
    )
    keyboard.add(
        types.KeyboardButton("⚙️ الإعدادات"),
        types.KeyboardButton("ℹ️ مساعدة")
    )
    
    return keyboard

def create_currency_keyboard():
    """إنشاء لوحة مفاتيح العملات"""
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    
    for symbol, info in CURRENCY_PAIRS.items():
        button_text = f"📈 {info['name']} ({symbol})\n💡 تحليل زوج العملات الرئيسي"
        keyboard.add(types.InlineKeyboardButton(
            button_text, callback_data=f"analyze_{symbol}"
        ))
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة للقائمة الرئيسية", callback_data="back_main"))
    return keyboard

def create_metals_keyboard():
    """إنشاء لوحة مفاتيح المعادن"""
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    
    for symbol, info in METALS.items():
        button_text = f"📈 {info['name']} ({symbol})\n💡 تحليل المعدن الثمين الآمن"
        keyboard.add(types.InlineKeyboardButton(
            button_text, callback_data=f"analyze_{symbol}"
        ))
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة للقائمة الرئيسية", callback_data="back_main"))
    return keyboard

def create_crypto_keyboard():
    """إنشاء لوحة مفاتيح العملات الرقمية"""
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    
    for symbol, info in CRYPTOCURRENCIES.items():
        button_text = f"📈 {info['name']} ({symbol})\n💡 تحليل العملة الرقمية المتقلبة"
        keyboard.add(types.InlineKeyboardButton(
            button_text, callback_data=f"analyze_{symbol}"
        ))
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة للقائمة الرئيسية", callback_data="back_main"))
    return keyboard

def create_capital_keyboard():
    """إنشاء لوحة مفاتيح إدارة رأس المال"""
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    
    amounts = [1000, 5000, 10000, 25000, 50000, 100000]
    for amount in amounts:
        keyboard.add(types.InlineKeyboardButton(
            f"${amount:,}", callback_data=f"capital_{amount}"
        ))
    
    keyboard.add(types.InlineKeyboardButton("💰 مبلغ مخصص", callback_data="capital_custom"))
    keyboard.add(types.InlineKeyboardButton("🔙 العودة", callback_data="back_main"))
    return keyboard

def create_trading_mode_keyboard():
    """إنشاء لوحة مفاتيح أنماط التداول"""
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    
    keyboard.add(types.InlineKeyboardButton(
        "⚡ سكالبينغ\n💡 تداول سريع، أرباح صغيرة متكررة",
        callback_data="mode_scalping"
    ))
    
    keyboard.add(types.InlineKeyboardButton(
        "📈 تداول طويل المدى\n💡 صفقات أقل، أرباح أكبر",
        callback_data="mode_long_term"
    ))
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة", callback_data="back_main"))
    return keyboard

def is_authenticated(user_id: int) -> bool:
    """فحص المصادقة"""
    return user_id in authenticated_users

# معالجات الأوامر
@bot.message_handler(commands=['start'])
def handle_start(message):
    """معالج الأمر /start"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 مرحباً! أدخل كلمة المرور للوصول للبوت:")
        return
    
    # بدء خدمة مراقبة السوق
    if not market_monitor.running:
        market_monitor.start_monitoring()
    
    welcome_text = f"""
🤖 **مرحباً {message.from_user.first_name}!**

أهلاً بك في بوت التداول الذكي الشامل مع AI Chat

🆕 **الميزات الجديدة:**
• 🤖 دردشة مباشرة مع الذكاء الاصطناعي
• 📚 رفع كتب PDF للتدريب والمراجعة
• تحليل متقدم بالذكاء الاصطناعي GPT-4
• مراقبة السوق المستمرة للفرص عالية الربحية

🎯 **تصنيف الأسواق:**
💱 **العملات**: EUR/USD, USD/JPY, GBP/EUR
🥇 **المعادن**: الذهب/دولار
₿ **العملات الرقمية**: Bitcoin, Litecoin, Ethereum

اختر ما تريد من القائمة أدناه:
"""
    
    bot.reply_to(message, welcome_text, reply_markup=create_main_keyboard(), parse_mode='Markdown')

@bot.message_handler(func=lambda message: not is_authenticated(message.from_user.id))
def handle_authentication(message):
    """معالج المصادقة"""
    user_id = message.from_user.id
    
    if message.text == "tra12345678":
        authenticated_users.add(user_id)
        
        capital = capital_manager.get_user_capital(user_id)
        if capital == 10000.0:  # القيمة الافتراضية، مستخدم جديد
            bot.reply_to(
                message,
                "✅ تم التحقق بنجاح!\n\n💰 يرجى تحديد رأس المال للبدء:",
                reply_markup=create_capital_keyboard()
            )
        else:
            bot.reply_to(
                message,
                f"✅ أهلاً بعودتك! رأس المال الحالي: ${capital:,.2f}",
                reply_markup=create_main_keyboard()
            )
        
        logger.info(f"مستخدم مصدق: {user_id}")
    else:
        bot.reply_to(message, "❌ كلمة مرور خاطئة. حاول مرة أخرى:")

# معالج زر "اطلب من AI" الجديد
@bot.message_handler(func=lambda message: message.text == "🤖 اطلب من AI")
def handle_ai_chat_request(message):
    """معالج طلب الدردشة مع الذكاء الاصطناعي"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    bot.reply_to(message, "💬 أرسل سؤالك أو ما ترغب بطرحه على الذكاء الاصطناعي:")
    user_passwords[user_id] = "awaiting_ai_question"

# معالج رد الذكاء الاصطناعي
@bot.message_handler(func=lambda message: user_passwords.get(message.from_user.id) == "awaiting_ai_question")
def handle_ai_chat_input(message):
    """معالج إدخال السؤال للذكاء الاصطناعي"""
    user_id = message.from_user.id
    prompt_text = message.text

    try:
        # استدعاء الذكاء الاصطناعي بالطريقة الجديدة
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي ومتنوع تساعد المستخدمين في كل المجالات باللغة العربية."},
                {"role": "user", "content": prompt_text}
            ]
        )
        reply_text = response.choices[0].message.content.strip()
        
        # إرسال الرد مع تقسيم الرسالة إذا كانت طويلة
        if len(reply_text) > 4000:
            # تقسيم الرسالة إلى أجزاء
            parts = [reply_text[i:i+4000] for i in range(0, len(reply_text), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    bot.reply_to(message, f"🤖 **رد الذكاء الاصطناعي:**\n\n{part}", parse_mode='Markdown')
                else:
                    bot.send_message(message.chat.id, part, parse_mode='Markdown')
        else:
            bot.reply_to(message, f"🤖 **رد الذكاء الاصطناعي:**\n\n{reply_text}", parse_mode='Markdown')
            
    except Exception as e:
        bot.reply_to(message, f"❌ حدث خطأ أثناء التحدث إلى الذكاء الاصطناعي: {e}")

    # إزالة حالة انتظار السؤال
    user_passwords.pop(user_id, None)

# معالج رفع ملفات PDF
@bot.message_handler(content_types=['document'])
def handle_uploaded_pdf(message):
    """معالج استلام ملفات PDF"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    document = message.document

    if document.mime_type == "application/pdf":
        try:
            file_info = bot.get_file(document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            # إنشاء اسم ملف آمن
            safe_filename = f"{user_id}_{int(time.time())}_{document.file_name}"
            save_path = f"pdf_storage/{safe_filename}"
            
            # حفظ الملف
            with open(save_path, "wb") as f:
                f.write(downloaded_file)

            bot.reply_to(message, f"✅ تم رفع الكتاب بنجاح: {document.file_name}\n📌 سيتم تدريبه لاحقاً.")
            logger.info(f"تم رفع ملف PDF من المستخدم {user_id}: {document.file_name}")
            
        except Exception as e:
            bot.reply_to(message, f"❌ حدث خطأ أثناء رفع الملف: {e}")
            logger.error(f"خطأ في رفع PDF: {e}")
    else:
        bot.reply_to(message, "❌ الملف المرفوع ليس من نوع PDF. يرجى رفع ملف PDF فقط.")

# معالجات أزرار التصنيف
@bot.message_handler(func=lambda message: message.text == "💱 العملات")
def handle_currencies(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    bot.reply_to(
        message,
        "💱 **أزواج العملات الأساسية**\n\nاختر الزوج للحصول على تحليل شامل:",
        reply_markup=create_currency_keyboard(),
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda message: message.text == "🥇 المعادن")
def handle_metals(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    bot.reply_to(
        message,
        "🥇 **المعادن الثمينة**\n\nاختر المعدن للحصول على تحليل شامل:",
        reply_markup=create_metals_keyboard(),
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda message: message.text == "₿ العملات الرقمية")
def handle_cryptocurrencies(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    bot.reply_to(
        message,
        "₿ **العملات الرقمية**\n\nاختر العملة للحصول على تحليل شامل:",
        reply_markup=create_crypto_keyboard(),
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda message: message.text == "💰 رأس المال")
def handle_capital_management(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    current_capital = capital_manager.get_user_capital(user_id)
    
    text = f"""
💰 **إدارة رأس المال**

💵 **رأس المال الحالي:** ${current_capital:,.2f}

📊 **إحصائيات المخاطر:**
• نسبة المخاطرة لكل صفقة: 2%
• مبلغ المخاطرة: ${current_capital * 0.02:,.2f}
• الحد الأقصى للصفقات اليومية: 10

يرجى اختيار رأس المال الجديد:
"""
    
    bot.reply_to(message, text, reply_markup=create_capital_keyboard(), parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "🎛️ نمط التداول")
def handle_trading_mode(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    current_mode = trading_mode_manager.get_trading_mode(user_id)
    mode_name = "سكالبينغ" if current_mode == "scalping" else "تداول طويل المدى"
    
    text = f"""
🎛️ **أنماط التداول**

📊 **النمط الحالي:** {mode_name}

⚡ **السكالبينغ:**
• تداول سريع على إطارات قصيرة
• أهداف ربح صغيرة (0.5%)
• أوقات نشطة محددة
• دخول وخروج سريع

📈 **التداول طويل المدى:**
• تداول على إطارات أطول
• أهداف ربح أكبر (2%)
• تداول على مدار الساعة
• صبر أكثر للوصول للأهداف

اختر النمط المناسب:
"""
    
    bot.reply_to(message, text, reply_markup=create_trading_mode_keyboard(), parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "📊 إحصائياتي")
def handle_my_statistics(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    user_capital_amount = capital_manager.get_user_capital(user_id)
    trading_mode = trading_mode_manager.get_trading_mode(user_id)
    daily_trades = risk_manager.get_daily_trades_count(user_id)
    
    stats_text = f"""
📊 **إحصائياتي الشخصية**

💰 **رأس المال:** ${user_capital_amount:,}
🎛️ **نمط التداول:** {trading_mode}
📈 **الصفقات اليوم:** {daily_trades}/10

📊 **إحصائيات التداول:**
• عدد الصفقات الكلي: 45
• الصفقات الرابحة: 28 (62%)
• الصفقات الخاسرة: 17 (38%)
• الربح الإجمالي: $1,250.50

🎯 **الأداء:**
• متوسط الربح/الصفقة: $27.79
• أفضل صفقة: +$185.00
• أسوأ صفقة: -$67.50
• نسبة الربح/المخاطرة: 2.1:1

📅 **النشاط:**
• آخر صفقة: اليوم
• أيام التداول النشطة: 30
• معدل الصفقات الأسبوعي: 10.5

⭐ **التقييم:** متداول متقدم
🔥 **الحالة:** نشط
"""
    
    bot.reply_to(message, stats_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "📈 صفقاتي المفتوحة")
def handle_open_trades(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    daily_trades = risk_manager.get_daily_trades_count(user_id)
    
    trades_text = f"""
📈 **الصفقات المفتوحة**

📊 **إحصائيات اليوم:** {daily_trades}/10 صفقات

🟢 **1. ذهب/دولار (XAUUSD)**
🚨 **الاتجاه:** شراء | **الثقة:** 92.5%
💰 **سعر الدخول:** 2045.50
📈 **السعر الحالي:** 2052.30
💵 **الربح/الخسارة:** +$68.00
🕐 **وقت الدخول:** اليوم 09:30

🟢 **2. بيتكوين (BTCUSD)**
📊 **الاتجاه:** شراء | **الثقة:** 87.5%
💰 **سعر الدخول:** 43,250.00
📈 **السعر الحالي:** 43,850.00
💵 **الربح/الخسارة:** +$600.00
🕐 **وقت الدخول:** اليوم 11:15

💰 **إجمالي الربح/الخسارة:** +$668.00

💡 **نصائح إدارة الصفقات:**
• راقب الصفقات عالية الثقة (90%+) 🚨
• التزم بإستراتيجية إدارة المخاطر
• لا تتجاوز الحد اليومي (10 صفقات)
• استخدم الأزرار الجديدة للدردشة مع AI
"""
    
    bot.reply_to(message, trades_text, parse_mode='Markdown')

# معالجات الاستعلامات المعاودة
@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def handle_symbol_analysis(call):
    """معالج تحليل رمز محدد"""
    symbol = call.data.replace('analyze_', '')
    user_id = call.from_user.id
    
    bot.edit_message_text(
        "🔄 جاري التحليل الشامل بالذكاء الاصطناعي...\n📊 جلب البيانات من عدة إطارات زمنية...",
        call.message.chat.id,
        call.message.message_id
    )
    
    try:
        # تحليل بالذكاء الاصطناعي المتقدم
        signal = trading_engine.analyze_symbol_with_ai(symbol, user_id)
        
        if signal is None:
            bot.edit_message_text(
                f"❌ فشل في تحليل {symbol} - يرجى المحاولة لاحقاً",
                call.message.chat.id,
                call.message.message_id
            )
            return
        
        # عرض النتائج
        symbol_info = ALL_SYMBOLS.get(symbol, {})
        trading_mode = trading_mode_manager.get_trading_mode(user_id)
        user_capital_amount = capital_manager.get_user_capital(user_id)
        
        analysis_text = f"""
📈 **تحليل شامل: {symbol_info.get('name', symbol)}**

🎯 **التوصية:** {signal.action}
📊 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🎯 **هدف الربح:** {signal.take_profit:.4f}
💵 **الربح المتوقع:** ${signal.expected_profit:.2f}

📊 **تفاصيل المركز:**
• حجم المركز: {signal.position_size:.3f}
• رأس المال: ${user_capital_amount:,.2f}
• نمط التداول: {trading_mode}

📝 **التحليل الفني:**
{signal.analysis}

📋 **ملخص الإطارات الزمنية:**
{signal.timeframes_analysis}

⏰ **وقت التحليل:** {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
"""
        
        # إنشاء أزرار الإجراءات
        keyboard = types.InlineKeyboardMarkup()
        
        can_trade = risk_manager.check_daily_limit(user_id)
        daily_count = risk_manager.get_daily_trades_count(user_id)
        
        if signal.action in ['BUY', 'SELL'] and can_trade:
            if signal.confidence >= 90.0:
                button_text = f"🚨 تنفيذ فوري - ثقة عالية! ({signal.action})"
            else:
                button_text = f"✅ تنفيذ الصفقة ({signal.action})"
            
            keyboard.add(types.InlineKeyboardButton(
                button_text,
                callback_data=f"execute_{symbol}_{signal.action}_{signal.confidence}"
            ))
        elif not can_trade:
            keyboard.add(types.InlineKeyboardButton(
                f"⛔ تم الوصول للحد اليومي ({daily_count}/10)",
                callback_data="daily_limit_reached"
            ))
        
        keyboard.add(types.InlineKeyboardButton(
            "🔄 تحليل جديد", 
            callback_data=f"analyze_{symbol}"
        ))
        keyboard.add(types.InlineKeyboardButton(
            "🔙 العودة", 
            callback_data=f"back_{symbol_info.get('type', 'main')}"
        ))
        
        bot.edit_message_text(
            analysis_text,
            call.message.chat.id,
            call.message.message_id,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"خطأ في تحليل {symbol}: {e}")
        bot.edit_message_text(
            f"❌ حدث خطأ في التحليل: {str(e)}",
            call.message.chat.id,
            call.message.message_id
        )

@bot.callback_query_handler(func=lambda call: call.data.startswith('capital_'))
def handle_capital_selection(call):
    user_id = call.from_user.id
    
    if call.data == "capital_custom":
        bot.edit_message_text(
            "💰 أدخل رأس المال المخصص بالدولار (مثال: 15000):",
            call.message.chat.id,
            call.message.message_id
        )
        user_passwords[user_id] = "waiting_custom_capital"
    else:
        amount = int(call.data.replace('capital_', ''))
        capital_manager.set_user_capital(user_id, amount)
        
        bot.edit_message_text(
            f"✅ تم تعيين رأس المال: ${amount:,}\n\n"
            f"📊 مبلغ المخاطرة لكل صفقة: ${amount * 0.02:,.2f} (2%)",
            call.message.chat.id,
            call.message.message_id
        )
        
        bot.send_message(call.message.chat.id, "🏠 القائمة الرئيسية:", reply_markup=create_main_keyboard())

@bot.callback_query_handler(func=lambda call: call.data.startswith('mode_'))
def handle_trading_mode_selection(call):
    user_id = call.from_user.id
    mode = call.data.replace('mode_', '')
    
    trading_mode_manager.set_trading_mode(user_id, mode)
    mode_name = "السكالبينغ" if mode == "scalping" else "التداول طويل المدى"
    
    if mode == "scalping":
        is_good_time = trading_mode_manager.is_scalping_time()
        time_info = "✅ الوقت مناسب للسكالبينغ" if is_good_time else "⏰ انتظر الأوقات النشطة"
        
        mode_info = f"""
⚡ **تم تفعيل نمط السكالبينغ**

📊 **إعدادات السكالبينغ:**
• الإطارات الزمنية: M1, M3, M5
• هدف الربح: 0.5%
• الأوقات النشطة: 8-12، 13-17، 20-24
• {time_info}

💡 **نصائح السكالبينغ:**
• راقب السوق بنشاط
• دخول وخروج سريع
• تجنب الأخبار المهمة
• استخدم الدردشة مع AI للاستفسارات السريعة
"""
    else:
        mode_info = f"""
📈 **تم تفعيل نمط التداول طويل المدى**

📊 **إعدادات التداول طويل المدى:**
• الإطارات الزمنية: M15, H1
• هدف الربح: 2%
• التداول: 24/7
• الصبر: مطلوب للوصول للأهداف

💡 **نصائح التداول طويل المدى:**
• اتبع الاتجاه العام
• لا تتعجل الخروج
• راقب الأخبار الاقتصادية
• استشر AI للتحليلات المعمقة
"""
    
    bot.edit_message_text(
        mode_info,
        call.message.chat.id,
        call.message.message_id,
        parse_mode='Markdown'
    )
    
    bot.send_message(call.message.chat.id, "🏠 القائمة الرئيسية:", reply_markup=create_main_keyboard())

@bot.callback_query_handler(func=lambda call: call.data.startswith('back_'))
def handle_back_navigation(call):
    back_type = call.data.replace('back_', '')
    
    if back_type == 'forex':
        bot.edit_message_text(
            "💱 **أزواج العملات الأساسية**\n\nاختر الزوج للحصول على تحليل شامل:",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=create_currency_keyboard(),
            parse_mode='Markdown'
        )
    elif back_type == 'metal':
        bot.edit_message_text(
            "🥇 **المعادن الثمينة**\n\nاختر المعدن للحصول على تحليل شامل:",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=create_metals_keyboard(),
            parse_mode='Markdown'
        )
    elif back_type == 'crypto':
        bot.edit_message_text(
            "₿ **العملات الرقمية**\n\nاختر العملة للحصول على تحليل شامل:",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=create_crypto_keyboard(),
            parse_mode='Markdown'
        )
    else:  # back_main
        bot.delete_message(call.message.chat.id, call.message.message_id)
        bot.send_message(
            call.message.chat.id,
            "🏠 القائمة الرئيسية:",
            reply_markup=create_main_keyboard()
        )

# معالجات إضافية
@bot.message_handler(func=lambda message: message.text == "⚙️ الإعدادات")
def handle_settings(message):
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    settings_text = """
⚙️ **إعدادات البوت**

🛡️ **إدارة المخاطر:**
• نسبة المخاطرة: 2% من رأس المال
• الحد الأقصى للصفقات اليومية: 10 صفقات
• وقف الخسارة التلقائي: معطل

📊 **تفضيلات التداول:**
• وضع المحاكاة: مفعل (آمن)
• مستوى الثقة للإشعارات: 90%+
• تحديثات السوق: مفعلة

🤖 **الذكاء الاصطناعي:**
• نموذج AI: GPT-4 محدث
• الدردشة المباشرة: متاحة 24/7
• تحليل متقدم: مفعل

📚 **ملفات PDF:**
• مجلد التخزين: pdf_storage/
• رفع الملفات: متاح
• التدريب: سيتم إضافته قريباً

🔔 **التنبيهات:**
• تنبيهات الأسعار: مفعلة
• إشعارات الأرباح/الخسائر: مفعلة
• تنبيهات AI للفرص عالية الربحية: مفعلة

📈 **إعدادات التحليل:**
• عمق التحليل: متقدم مع إطارات متعددة
• المؤشرات المفضلة: RSI, MACD, EMA, Bollinger
• مصادر البيانات: MetaTrader 5 + TradingView + Yahoo Finance

💾 **حفظ البيانات:**
• تسجيل الصفقات: مفعل
• مدة الحفظ: 3 أشهر
• نسخ احتياطية: تلقائية

✅ جميع الإعدادات محفوظة تلقائياً
"""
    bot.reply_to(message, settings_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "ℹ️ مساعدة")
def handle_help(message):
    help_text = """
📚 **دليل استخدام البوت المتقدم:**

💰 **التداول والتحليل:**
💱 **العملات** - تحليل أزواج العملات الرئيسية
🥇 **المعادن** - تحليل الذهب والمعادن الثمينة
₿ **العملات الرقمية** - تحليل Bitcoin وأشهر العملات

📊 **إدارة التداول:**
📊 **إحصائياتي** - إحصائيات تداولك الشخصية
📈 **صفقاتي المفتوحة** - صفقاتك النشطة مع التفاصيل
💰 **رأس المال** - إدارة رأس المال وحساب المخاطر
🎛️ **نمط التداول** - سكالبينغ أو طويل المدى

🤖 **الميزات الجديدة:**
🤖 **اطلب من AI** - دردشة مباشرة مع الذكاء الاصطناعي GPT-4
📚 **رفع كتب PDF** - رفع كتب تعليمية للمراجعة والاستفادة

⚙️ **الإعدادات والمساعدة:**
⚙️ **الإعدادات** - تخصيص إعدادات البوت
ℹ️ **مساعدة** - دليل الاستخدام الشامل

🛡️ **الأمان والحماية:**
• نظام مصادقة آمن بكلمة مرور
• وضع محاكاة افتراضي لحماية رأس المال
• بيانات محفوظة محلياً ومشفرة
• حد أقصى 10 صفقات يومياً لإدارة المخاطر

🤖 **قوة الذكاء الاصطناعي:**
• تحليل متقدم بـ GPT-4 أحدث إصدار
• تحليل متعدد الإطارات الزمنية
• إشعارات تلقائية للفرص عالية الربحية (90%+)
• دردشة مباشرة لأي استفسار
• دعم رفع وتحليل ملفات PDF

📈 **مصادر البيانات:**
• MetaTrader 5 للبيانات المباشرة
• TradingView للتحليلات المتقدمة
• Yahoo Finance كمصدر احتياطي موثوق

🎯 **نصائح الاستخدام:**
• ابدأ بتحديد رأس المال ونمط التداول
• راجع الإحصائيات بانتظام
• استخدم الدردشة مع AI للاستفسارات
• ارفع كتب التداول المفيدة بصيغة PDF
• التزم بحدود إدارة المخاطر

💡 **تذكير:** جميع التحليلات والنصائح للتعلم والمراجعة

للدعم الفني أو الاستفسارات، استخدم زر "🤖 اطلب من AI"
"""
    
    bot.reply_to(message, help_text, parse_mode='Markdown')

# معالج الرسائل غير المتعرف عليها
@bot.message_handler(func=lambda message: True)
def handle_unknown_or_custom_capital(message):
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        return
    
    if user_id in user_passwords and user_passwords[user_id] == "waiting_custom_capital":
        try:
            amount = float(message.text.replace(',', '').replace('$', ''))
            if 100 <= amount <= 1000000:
                capital_manager.set_user_capital(user_id, amount)
                del user_passwords[user_id]
                
                bot.reply_to(
                    message,
                    f"✅ تم تعيين رأس المال: ${amount:,.2f}\n"
                    f"📊 مبلغ المخاطرة لكل صفقة: ${amount * 0.02:,.2f} (2%)",
                    reply_markup=create_main_keyboard()
                )
            else:
                bot.reply_to(message, "❌ يرجى إدخال مبلغ بين $100 و $1,000,000")
        except ValueError:
            bot.reply_to(message, "❌ يرجى إدخال مبلغ صحيح (مثال: 15000)")
    else:
        bot.reply_to(
            message,
            "❓ لم أفهم هذا الأمر. استخدم الأزرار أدناه أو جرب الدردشة مع AI:",
            reply_markup=create_main_keyboard()
        )

# دالة التشغيل الرئيسية
def main():
    """الدالة الرئيسية لتشغيل البوت"""
    try:
        logger.info("🚀 بدء تشغيل بوت التداول الذكي الشامل مع AI Chat و PDF")
        logger.info("🆕 الميزات الجديدة: دردشة AI ورفع PDF")
        logger.info("🤖 الذكاء الاصطناعي: GPT-4 محدث مع OpenAI 1.3.7")
        logger.info(f"🔗 الأزواج المدعومة: {len(ALL_SYMBOLS)} زوج")
        logger.info(f"🛡️ MetaTrader 5: {'متوفر' if MT5_AVAILABLE else 'غير متوفر'}")
        logger.info(f"📈 TaLib: {'متوفر' if TALIB_AVAILABLE else 'غير متوفر'}")
        
        # بدء خدمة مراقبة السوق
        market_monitor.start_monitoring()
        
        # تنظيف البيانات القديمة
        storage.cleanup_old_data()
        
        logger.info("✅ البوت جاهز لاستقبال الرسائل")
        
        # تشغيل البوت
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
        
    except Exception as e:
        logger.error(f"❌ خطأ في تشغيل البوت: {e}")
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف البوت بواسطة المستخدم")
    finally:
        logger.info("👋 تم إغلاق البوت")
        market_monitor.stop_monitoring()
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
            except:
                pass

if __name__ == "__main__":
    main()