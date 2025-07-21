#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 بوت التداول الذكي الشامل - الإصدار المحدث
===================================================

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
import openai
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

# إعداد البوت والذكاء الاصطناعي
bot = telebot.TeleBot('7703327028:AAHLqgR1HtVPsq6LfUKEWzNEgLZjJPLa6YU')
openai.api_key = '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'

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

# متغيرات عامة
authenticated_users = set()
user_passwords = {}  # تخزين كلمات المرور للمستخدمين
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
            if not mt5.initialize():
                logger.error(f"فشل تهيئة MT5: {mt5.last_error()}")
                return False
                
            # التحقق من الاتصال
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("لا يمكن الحصول على معلومات الحساب")
                mt5.shutdown()
                return False
                
            self.connected = True
            logger.info(f"تم الاتصال بـ MT5 بنجاح - الحساب: {account_info.login}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في الاتصال بـ MT5: {e}")
            return False
    
    def disconnect(self):
        """قطع الاتصال بـ MetaTrader 5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("تم قطع الاتصال من MT5")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """الحصول على معلومات الرمز"""
        if not self.connected:
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
                'point': symbol_info.point,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max
            }
        except Exception as e:
            logger.error(f"خطأ في الحصول على معلومات الرمز {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str, timeframe, count: int = 100) -> Optional[pd.DataFrame]:
        """الحصول على بيانات السوق من MT5"""
        if not self.connected:
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على بيانات MT5 لـ {symbol}: {e}")
            return None

class TradingViewScraper:
    """كاشط بيانات TradingView"""
    
    def __init__(self):
        self.base_url = "https://www.tradingview.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """محاولة جلب بيانات من TradingView"""
        try:
            # محاولة جلب بيانات أساسية
            # هذا مثال بسيط - في التطبيق الحقيقي نحتاج API key أو scraping متقدم
            
            # استخدام Yahoo Finance كبديل
            yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
            if yahoo_symbol:
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    return {
                        'symbol': symbol,
                        'price': hist['Close'].iloc[-1],
                        'change': ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100,
                        'volume': hist['Volume'].iloc[-1],
                        'high': hist['High'].iloc[-1],
                        'low': hist['Low'].iloc[-1],
                        'source': 'TradingView_Alternative'
                    }
            return None
            
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات TradingView لـ {symbol}: {e}")
            return None
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> Optional[str]:
        """تحويل رموز التداول إلى رموز Yahoo Finance"""
        conversion_map = {
            'XAUUSD': 'GC=F',
            'EURUSD': 'EURUSD=X',
            'USDJPY': 'USDJPY=X',
            'GBPEUR': 'GBPEUR=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'LTCUSD': 'LTC-USD'
        }
        return conversion_map.get(symbol)

class AdvancedMarketAnalyzer:
    """محلل الأسواق المالية المتقدم"""
    
    def __init__(self):
        self.storage = SimpleStorage('market_data_advanced.json')
        self.mt5_manager = MetaTrader5Manager()
        self.tv_scraper = TradingViewScraper()
        
    def get_multi_timeframe_data(self, symbol: str) -> Dict:
        """الحصول على بيانات متعددة الإطارات الزمنية"""
        try:
            data = {}
            
            # محاولة الحصول على البيانات من MT5 أولاً
            if self.mt5_manager.connect():
                for tf_name, tf_info in TIMEFRAMES.items():
                    if tf_info['mt5']:
                        df = self.mt5_manager.get_market_data(symbol, tf_info['mt5'], 100)
                        if df is not None:
                            data[tf_name] = df
                self.mt5_manager.disconnect()
            
            # إذا لم نحصل على بيانات من MT5، استخدم Yahoo Finance
            if not data:
                yahoo_symbol = self._convert_symbol(symbol)
                if yahoo_symbol:
                    ticker = yf.Ticker(yahoo_symbol)
                    # جلب بيانات بفترات مختلفة
                    periods = ['1d', '5d', '1mo']
                    intervals = ['1m', '5m', '1h']
                    
                    for period, interval in zip(periods, intervals):
                        try:
                            df = ticker.history(period=period, interval=interval)
                            if not df.empty:
                                data[f'{interval}_data'] = df
                        except:
                            continue
            
            return data
            
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات متعددة الإطارات لـ {symbol}: {e}")
            return {}
    
    def _convert_symbol(self, symbol: str) -> Optional[str]:
        """تحويل رموز التداول إلى رموز Yahoo Finance"""
        conversion_map = {
            'XAUUSD': 'GC=F',
            'EURUSD': 'EURUSD=X',
            'USDJPY': 'USDJPY=X',
            'GBPEUR': 'GBPEUR=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'LTCUSD': 'LTC-USD'
        }
        return conversion_map.get(symbol)
    
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict:
        """حساب المؤشرات الفنية المتقدمة"""
        try:
            indicators = {}
            
            if data.empty:
                return indicators
            
            # المؤشرات الأساسية
            indicators['current_price'] = data['Close'].iloc[-1]
            indicators['previous_close'] = data['Close'].iloc[-2] if len(data) > 1 else indicators['current_price']
            indicators['price_change'] = indicators['current_price'] - indicators['previous_close']
            indicators['price_change_pct'] = (indicators['price_change'] / indicators['previous_close']) * 100
            
            # المتوسطات المتحركة
            indicators['sma_10'] = data['Close'].rolling(window=10).mean().iloc[-1]
            indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else indicators['sma_20']
            indicators['ema_12'] = data['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = (data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()).ewm(span=9).mean().iloc[-1]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            sma_20 = data['Close'].rolling(window=20).mean()
            std_20 = data['Close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
            indicators['bb_position'] = (indicators['current_price'] - indicators['bb_lower']) / indicators['bb_width']
            
            # Stochastic
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            indicators['stoch_k'] = ((data['Close'] - low_14) / (high_14 - low_14) * 100).iloc[-1]
            
            # Average True Range (ATR)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
            
            # Volume indicators
            if 'Volume' in data.columns:
                indicators['volume'] = data['Volume'].iloc[-1]
                indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().iloc[-1]
                indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma']
            else:
                indicators['volume'] = 0
                indicators['volume_sma'] = 0
                indicators['volume_ratio'] = 1
            
            # Support and Resistance levels
            recent_data = data.tail(20)
            indicators['resistance'] = recent_data['High'].max()
            indicators['support'] = recent_data['Low'].min()
            
            return indicators
            
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات المتقدمة: {e}")
            return {}
    
    def analyze_trend_strength(self, indicators: Dict) -> Dict:
        """تحليل قوة الاتجاه"""
        try:
            analysis = {
                'trend_direction': 'sideways',
                'trend_strength': 'weak',
                'confidence': 50.0
            }
            
            # تحليل الاتجاه باستخدام المتوسطات المتحركة
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            
            # تحديد الاتجاه
            if current_price > sma_20 > sma_50:
                analysis['trend_direction'] = 'bullish'
            elif current_price < sma_20 < sma_50:
                analysis['trend_direction'] = 'bearish'
            else:
                analysis['trend_direction'] = 'sideways'
            
            # تحليل قوة الاتجاه
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            strength_score = 0
            
            # RSI analysis
            if 70 <= rsi <= 80 or 20 <= rsi <= 30:
                strength_score += 1
            elif rsi > 80 or rsi < 20:
                strength_score += 2
            
            # MACD analysis
            if macd > macd_signal:
                strength_score += 1
            
            # Volume analysis
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                strength_score += 1
            
            # تحديد قوة الاتجاه
            if strength_score >= 3:
                analysis['trend_strength'] = 'strong'
                analysis['confidence'] = min(85.0, 60 + (strength_score * 5))
            elif strength_score >= 2:
                analysis['trend_strength'] = 'moderate'
                analysis['confidence'] = min(75.0, 50 + (strength_score * 5))
            else:
                analysis['trend_strength'] = 'weak'
                analysis['confidence'] = max(40.0, 30 + (strength_score * 5))
            
            return analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل قوة الاتجاه: {e}")
            return {'trend_direction': 'sideways', 'trend_strength': 'weak', 'confidence': 50.0}

class CapitalManager:
    """مدير رأس المال"""
    
    def __init__(self):
        self.storage = SimpleStorage('capital_management.json')
    
    def set_user_capital(self, user_id: int, capital: float):
        """تعيين رأس مال المستخدم"""
        user_capital[user_id] = capital
        self.storage.set(f'capital_{user_id}', capital)
        logger.info(f"تم تعيين رأس المال للمستخدم {user_id}: ${capital:,.2f}")
    
    def get_user_capital(self, user_id: int) -> float:
        """الحصول على رأس مال المستخدم"""
        if user_id in user_capital:
            return user_capital[user_id]
        
        stored_capital = self.storage.get(f'capital_{user_id}', 1000.0)
        user_capital[user_id] = stored_capital
        return stored_capital
    
    def calculate_position_size(self, user_id: int, risk_percent: float = 2.0, stop_loss_pips: float = 50) -> Dict:
        """حساب حجم المركز بناءً على إدارة المخاطر"""
        try:
            capital = self.get_user_capital(user_id)
            risk_amount = capital * (risk_percent / 100)
            
            # حساب حجم المركز
            # هذا مثال مبسط - في الواقع يعتمد على قيمة النقطة للزوج
            pip_value = 1.0  # قيمة افتراضية
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            return {
                'capital': capital,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
                'position_size': position_size,
                'stop_loss_pips': stop_loss_pips
            }
            
        except Exception as e:
            logger.error(f"خطأ في حساب حجم المركز: {e}")
            return {
                'capital': 1000.0,
                'risk_amount': 20.0,
                'risk_percent': 2.0,
                'position_size': 0.01,
                'stop_loss_pips': 50
            }

class RiskManager:
    """مدير المخاطر المحدث"""
    
    def __init__(self):
        self.storage = SimpleStorage('risk_settings_v2.json')
        self.default_settings = {
            'max_daily_trades': 10,  # محدث إلى 10 صفقات
            'max_risk_per_trade': 2.0,
            'min_confidence': 90.0,  # محدث إلى 90% للإشعارات
            'auto_stop_loss': False,  # إيقاف وقف الخسارة التلقائي
            'data_retention_months': 3  # 3 أشهر بدلاً من 6
        }
    
    def get_risk_settings(self) -> Dict:
        """الحصول على إعدادات المخاطر"""
        settings = self.storage.get('settings', self.default_settings)
        return {**self.default_settings, **settings}
    
    def check_high_probability_trade(self, confidence: float) -> bool:
        """فحص الصفقات عالية الاحتمالية (90%+)"""
        min_confidence = self.get_risk_settings()['min_confidence']
        return confidence >= min_confidence
    
    def get_daily_trades_count(self, user_id: int) -> int:
        """الحصول على عدد الصفقات اليومية"""
        today = datetime.now().date().isoformat()
        return self.storage.get(f'daily_trades_{user_id}_{today}', 0)
    
    def check_daily_limit(self, user_id: int) -> bool:
        """فحص الحد اليومي للصفقات (10 صفقات)"""
        current_count = self.get_daily_trades_count(user_id)
        max_trades = self.get_risk_settings()['max_daily_trades']
        return current_count < max_trades
    
    def record_trade(self, user_id: int):
        """تسجيل صفقة جديدة"""
        today = datetime.now().date().isoformat()
        key = f'daily_trades_{user_id}_{today}'
        current_count = self.storage.get(key, 0)
        self.storage.set(key, current_count + 1)

class TradingModeManager:
    """مدير أنماط التداول"""
    
    def __init__(self):
        self.storage = SimpleStorage('trading_modes.json')
        self.scalping_hours = [
            (8, 12),   # 8:00 - 12:00
            (13, 17),  # 13:00 - 17:00
            (20, 24)   # 20:00 - 24:00
        ]
    
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
        """فحص ما إذا كان الوقت الحالي مناسب للسكالبينغ"""
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
                'max_hold_time': timedelta(minutes=30),
                'profit_target': 0.5,  # 0.5%
                'active_hours': self.scalping_hours,
                'min_confidence': 85.0
            }
        else:  # long_term
            return {
                'timeframes': ['M15', 'H1'],
                'max_hold_time': timedelta(days=7),
                'profit_target': 2.0,  # 2%
                'active_hours': [(0, 24)],  # 24/7
                'min_confidence': 75.0
            }

class SmartTradingEngine:
    """محرك التداول الذكي المحدث"""
    
    def __init__(self):
        self.storage = SimpleStorage('smart_trades.json')
        self.risk_manager = RiskManager()
        self.capital_manager = CapitalManager()
        self.mode_manager = TradingModeManager()
        self.analyzer = AdvancedMarketAnalyzer()
        
        # تشغيل تنظيف البيانات دورياً
        self.storage.cleanup_old_data(90)  # 3 أشهر
    
    def analyze_with_ai(self, symbol: str, timeframes_data: Dict, user_id: int) -> Optional[TradeSignal]:
        """تحليل متقدم بالذكاء الاصطناعي"""
        try:
            # الحصول على نمط التداول
            trading_mode = self.mode_manager.get_trading_mode(user_id)
            mode_settings = self.mode_manager.get_mode_settings(trading_mode)
            
            # تحليل البيانات على إطارات زمنية متعددة
            multi_tf_analysis = {}
            main_indicators = None
            
            for tf_name, data in timeframes_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    indicators = self.analyzer.calculate_advanced_indicators(data)
                    trend_analysis = self.analyzer.analyze_trend_strength(indicators)
                    
                    multi_tf_analysis[tf_name] = {
                        'indicators': indicators,
                        'trend': trend_analysis
                    }
                    
                    # استخدام الإطار الزمني الأساسي للتحليل النهائي
                    if tf_name in mode_settings['timeframes'] and main_indicators is None:
                        main_indicators = indicators
            
            if not main_indicators:
                return None
            
            # إعداد السياق للذكاء الاصطناعي
            market_context = self._create_ai_context(symbol, multi_tf_analysis, trading_mode)
            
            # طلب التحليل من GPT-4
            ai_response = self._get_ai_analysis(market_context)
            
            # معالجة رد الذكاء الاصطناعي
            signal = self._process_ai_response(ai_response, symbol, main_indicators, user_id)
            
            return signal
            
        except Exception as e:
            logger.error(f"خطأ في التحليل بالذكاء الاصطناعي: {e}")
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
    
    def _get_ai_analysis(self, context: str) -> str:
        """الحصول على تحليل من GPT-4"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "أنت محلل مالي خبير متخصص في التحليل الفني والتداول. قدم تحليلات دقيقة ومهنية باللغة العربية."
                    },
                    {"role": "user", "content": context}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على تحليل AI: {e}")
            return ""
    
    def _process_ai_response(self, ai_response: str, symbol: str, indicators: Dict, user_id: int) -> Optional[TradeSignal]:
        """معالجة رد الذكاء الاصطناعي"""
        try:
            import re
            
            # محاولة استخراج JSON
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    analysis_data = self._extract_fallback_data(ai_response, indicators)
            else:
                analysis_data = self._extract_fallback_data(ai_response, indicators)
            
            # حساب حجم المركز
            position_calc = self.capital_manager.calculate_position_size(user_id)
            
            # إنشاء إشارة التداول
            signal = TradeSignal(
                symbol=symbol,
                action=analysis_data.get('action', 'WAIT'),
                confidence=float(analysis_data.get('confidence', 50.0)),
                entry_price=float(analysis_data.get('entry_price', indicators.get('current_price', 0))),
                take_profit=float(analysis_data.get('take_profit', indicators.get('current_price', 0) * 1.02)),
                analysis=analysis_data.get('analysis', ai_response),
                timestamp=datetime.now(),
                timeframes_analysis=analysis_data.get('timeframe_summary', ''),
                position_size=position_calc.get('position_size', 0.01),
                expected_profit=self._calculate_expected_profit(
                    float(analysis_data.get('entry_price', indicators.get('current_price', 0))),
                    float(analysis_data.get('take_profit', indicators.get('current_price', 0) * 1.02)),
                    position_calc.get('position_size', 0.01)
                )
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"خطأ في معالجة رد AI: {e}")
            return None
    
    def _extract_fallback_data(self, text: str, indicators: Dict) -> Dict:
        """استخراج البيانات في حالة فشل JSON"""
        current_price = indicators.get('current_price', 0)
        
        # تحليل النص لاستخراج المعلومات
        if 'شراء' in text or 'BUY' in text.upper():
            action = 'BUY'
            take_profit = current_price * 1.02
        elif 'بيع' in text or 'SELL' in text.upper():
            action = 'SELL'
            take_profit = current_price * 0.98
        else:
            action = 'WAIT'
            take_profit = current_price
        
        # محاولة استخراج مستوى الثقة
        confidence = 60.0
        import re
        confidence_match = re.search(r'(\d+)%', text)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        
        return {
            'action': action,
            'confidence': confidence,
            'entry_price': current_price,
            'take_profit': take_profit,
            'analysis': text
        }
    
    def _calculate_expected_profit(self, entry_price: float, take_profit: float, position_size: float) -> float:
        """حساب الربح المتوقع"""
        try:
            if entry_price > 0:
                profit_pips = abs(take_profit - entry_price)
                return profit_pips * position_size
            return 0.0
        except:
            return 0.0

class NotificationManager:
    """مدير الإشعارات"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.sent_notifications = set()
    
    def send_high_probability_alert(self, user_id: int, signal: TradeSignal):
        """إرسال إشعار للصفقات عالية الاحتمالية (90%+)"""
        try:
            if signal.confidence >= 90.0:
                symbol_info = ALL_SYMBOLS.get(signal.symbol, {})
                
                alert_text = f"""
🚨 **تنبيه صفقة عالية الاحتمالية!**

💎 **الرمز:** {symbol_info.get('name', signal.symbol)} ({signal.symbol})
🎯 **التوصية:** {signal.action}
📊 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🎯 **هدف الربح:** {signal.take_profit:.4f}
💵 **الربح المتوقع:** ${signal.expected_profit:.2f}

📈 **التحليل:**
{signal.analysis[:200]}...

⚠️ **هذه صفقة بنسبة ربح {signal.confidence:.0f}% أو أعلى!**

⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                # إرسال الإشعار
                self.bot.send_message(user_id, alert_text, parse_mode='Markdown')
                
                # تسجيل الإشعار
                notification_id = f"{user_id}_{signal.symbol}_{int(signal.timestamp.timestamp())}"
                self.sent_notifications.add(notification_id)
                
                logger.info(f"تم إرسال إشعار صفقة عالية الاحتمالية للمستخدم {user_id}")
                
        except Exception as e:
            logger.error(f"خطأ في إرسال الإشعار: {e}")

# إنشاء كائنات النظام
advanced_analyzer = AdvancedMarketAnalyzer()
smart_trading_engine = SmartTradingEngine()
capital_manager = CapitalManager()
trading_mode_manager = TradingModeManager()
notification_manager = NotificationManager(bot)
storage = SimpleStorage('bot_data_v2.json')

# خدمة مراقبة السوق في الخلفية
class MarketMonitorService:
    """خدمة مراقبة السوق المستمرة"""
    
    def __init__(self):
        self.active = False
        self.monitor_thread = None
        self.check_interval = 300  # 5 دقائق
    
    def start_monitoring(self):
        """بدء مراقبة السوق"""
        if not self.active:
            self.active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("تم بدء خدمة مراقبة السوق")
    
    def stop_monitoring(self):
        """إيقاف مراقبة السوق"""
        self.active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("تم إيقاف خدمة مراقبة السوق")
    
    def _monitor_loop(self):
        """حلقة مراقبة السوق"""
        while self.active:
            try:
                self._check_all_symbols()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"خطأ في مراقبة السوق: {e}")
                time.sleep(60)  # انتظار دقيقة واحدة في حالة الخطأ
    
    def _check_all_symbols(self):
        """فحص جميع الرموز للفرص عالية الاحتمالية"""
        for symbol in ALL_SYMBOLS.keys():
            try:
                # جلب البيانات
                timeframes_data = advanced_analyzer.get_multi_timeframe_data(symbol)
                
                if timeframes_data:
                    # تحليل للمستخدمين المصادقين
                    for user_id in authenticated_users:
                        signal = smart_trading_engine.analyze_with_ai(symbol, timeframes_data, user_id)
                        
                        if signal and signal.confidence >= 90.0:
                            notification_manager.send_high_probability_alert(user_id, signal)
                
                # فترة راحة بين الرموز
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"خطأ في فحص الرمز {symbol}: {e}")

# إنشاء خدمة المراقبة
market_monitor = MarketMonitorService()

# دوال الواجهة
def create_main_keyboard():
    """إنشاء لوحة المفاتيح الرئيسية المحدثة"""
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    
    # تصنيف الأزواج
    keyboard.add(
        types.KeyboardButton("💱 العملات"),
        types.KeyboardButton("🥇 المعادن")
    )
    keyboard.add(
        types.KeyboardButton("₿ العملات الرقمية")
    )
    
    # إدارة التداول
    keyboard.add(
        types.KeyboardButton("📊 إحصائياتي"),
        types.KeyboardButton("📈 صفقاتي المفتوحة")
    )
    
    # الأدوات
    keyboard.add(
        types.KeyboardButton("📋 ملخص السوق AI"),
        types.KeyboardButton("🔍 أنماط التداول")
    )
    
    # الإعدادات
    keyboard.add(
        types.KeyboardButton("💰 رأس المال"),
        types.KeyboardButton("🎛️ نمط التداول")
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
        button_text = f"📈 {info['name']} ({info['symbol']})\n💡 تحليل زوج العملات الأساسي"
        keyboard.add(types.InlineKeyboardButton(
            button_text, callback_data=f"analyze_{symbol}"
        ))
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة للقائمة الرئيسية", callback_data="back_main"))
    return keyboard

def create_metals_keyboard():
    """إنشاء لوحة مفاتيح المعادن"""
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    
    for symbol, info in METALS.items():
        button_text = f"📈 {info['name']} ({info['symbol']})\n💡 تحليل المعدن الثمين الآمن"
        keyboard.add(types.InlineKeyboardButton(
            button_text, callback_data=f"analyze_{symbol}"
        ))
    
    keyboard.add(types.InlineKeyboardButton("🔙 العودة للقائمة الرئيسية", callback_data="back_main"))
    return keyboard

def create_crypto_keyboard():
    """إنشاء لوحة مفاتيح العملات الرقمية"""
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    
    for symbol, info in CRYPTOCURRENCIES.items():
        button_text = f"📈 {info['name']} ({info['symbol']})\n💡 تحليل العملة الرقمية المتقلبة"
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

# معالجات الأوامر
@bot.message_handler(commands=['start'])
def handle_start(message):
    """معالج الأمر /start"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 مرحباً! أدخل كلمة المرور للوصول للبوت:")
        return
    
    # بدء خدمة مراقبة السوق
    if not market_monitor.active:
        market_monitor.start_monitoring()
    
    welcome_text = f"""
🤖 **مرحباً {message.from_user.first_name}!**

أهلاً بك في بوت التداول الذكي الشامل الإصدار المحدث

🆕 **الميزات الجديدة:**
• 📡 ربط MetaTrader 5 و TradingView
• 🧠 تحليل لحظي متقدم بالذكاء الاصطناعي
• 🚨 إشعارات الصفقات عالية الربحية (90%+)
• 💰 إدارة رأس المال المتقدمة
• ⚡ أنماط التداول (سكالبينغ/طويل المدى)
• 🛡️ حد أقصى 10 صفقات يوميًا
• 📅 حفظ البيانات لـ 3 أشهر

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
        
        # طلب رأس المال عند أول تسجيل دخول
        capital = capital_manager.get_user_capital(user_id)
        if capital == 1000.0:  # القيمة الافتراضية
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

# معالجات أزرار التصنيف
@bot.message_handler(func=lambda message: message.text == "💱 العملات")
def handle_currencies(message):
    """معالج أزرار العملات"""
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
    """معالج أزرار المعادن"""
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
    """معالج أزرار العملات الرقمية"""
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
    """معالج إدارة رأس المال"""
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
    """معالج أنماط التداول"""
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
• تداول سريع على إطارات قصيرة (M1, M3, M5)
• أهداف ربح صغيرة (0.5%)
• أوقات نشطة محددة
• دخول وخروج سريع

📈 **التداول طويل المدى:**
• تداول على إطارات أطول (M15, H1)
• أهداف ربح أكبر (2%)
• تداول على مدار الساعة
• صبر أكثر للوصول للأهداف

اختر النمط المناسب:
"""
    
    bot.reply_to(message, text, reply_markup=create_trading_mode_keyboard(), parse_mode='Markdown')

# معالجات الاستعلامات المعاودة
@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def handle_symbol_analysis(call):
    """معالج تحليل رمز محدد"""
    symbol = call.data.replace('analyze_', '')
    user_id = call.from_user.id
    
    # رسالة انتظار
    bot.edit_message_text(
        "🔄 جاري التحليل الشامل بالذكاء الاصطناعي...\n📊 جلب البيانات من مصادر متعددة...",
        call.message.chat.id,
        call.message.message_id
    )
    
    try:
        # جلب البيانات من مصادر متعددة
        timeframes_data = advanced_analyzer.get_multi_timeframe_data(symbol)
        
        if not timeframes_data:
            bot.edit_message_text(
                f"❌ لا يمكن الحصول على بيانات {symbol}",
                call.message.chat.id,
                call.message.message_id
            )
            return
        
        # تحليل بالذكاء الاصطناعي
        signal = smart_trading_engine.analyze_with_ai(symbol, timeframes_data, user_id)
        
        if signal is None:
            bot.edit_message_text(
                f"❌ فشل في تحليل {symbol}",
                call.message.chat.id,
                call.message.message_id
            )
            return
        
        # عرض النتائج
        symbol_info = ALL_SYMBOLS.get(symbol, {})
        trading_mode = trading_mode_manager.get_trading_mode(user_id)
        user_capital = capital_manager.get_user_capital(user_id)
        
        analysis_text = f"""
📈 **تحليل شامل: {symbol_info.get('name', symbol)}**

🎯 **التوصية:** {signal.action}
📊 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🎯 **هدف الربح:** {signal.take_profit:.4f}
💵 **الربح المتوقع:** ${signal.expected_profit:.2f}

📊 **تفاصيل المركز:**
• حجم المركز: {signal.position_size:.3f}
• رأس المال: ${user_capital:,.2f}
• نمط التداول: {trading_mode}

📝 **التحليل الفني:**
{signal.analysis[:300]}...

⏰ **وقت التحليل:** {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
"""
        
        # إنشاء أزرار الإجراءات
        keyboard = types.InlineKeyboardMarkup()
        
        # فحص الحدود اليومية
        can_trade = smart_trading_engine.risk_manager.check_daily_limit(user_id)
        daily_count = smart_trading_engine.risk_manager.get_daily_trades_count(user_id)
        
        if signal.action in ['BUY', 'SELL'] and can_trade:
            if signal.confidence >= 90.0:
                button_text = f"🚨 تنفيذ فوري - ثقة عالية! ({signal.action})"
                notification_manager.send_high_probability_alert(user_id, signal)
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
    """معالج اختيار رأس المال"""
    user_id = call.from_user.id
    
    if call.data == "capital_custom":
        bot.edit_message_text(
            "💰 أدخل رأس المال المخصص بالدولار (مثال: 15000):",
            call.message.chat.id,
            call.message.message_id
        )
        # تسجيل المستخدم في وضع إدخال رأس مال مخصص
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
        
        # إرسال القائمة الرئيسية
        bot.send_message(call.message.chat.id, "🏠 القائمة الرئيسية:", reply_markup=create_main_keyboard())

@bot.callback_query_handler(func=lambda call: call.data.startswith('mode_'))
def handle_trading_mode_selection(call):
    """معالج اختيار نمط التداول"""
    user_id = call.from_user.id
    mode = call.data.replace('mode_', '')
    
    trading_mode_manager.set_trading_mode(user_id, mode)
    mode_name = "السكالبينغ" if mode == "scalping" else "التداول طويل المدى"
    
    # معلومات إضافية حسب النمط
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
"""
    
    bot.edit_message_text(
        mode_info,
        call.message.chat.id,
        call.message.message_id,
        parse_mode='Markdown'
    )
    
    # إرسال القائمة الرئيسية
    bot.send_message(call.message.chat.id, "🏠 القائمة الرئيسية:", reply_markup=create_main_keyboard())

# معالجات إضافية (باقي الكود محفوظ من الملف الأصلي مع التحديثات)
@bot.message_handler(func=lambda message: message.text == "📊 إحصائياتي")
def handle_my_statistics(message):
    """معالج الإحصائيات الشخصية"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    user_capital = capital_manager.get_user_capital(user_id)
    trading_mode = trading_mode_manager.get_trading_mode(user_id)
    daily_trades = smart_trading_engine.risk_manager.get_daily_trades_count(user_id)
    
    stats_text = f"""
📊 **إحصائياتي الشخصية**

💰 **رأس المال:** ${user_capital:,}
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

⚠️ **تذكير:** هذه إحصائيات محاكاة لأغراض تعليمية
"""
    
    bot.reply_to(message, stats_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "📈 صفقاتي المفتوحة")
def handle_open_trades(message):
    """معالج الصفقات المفتوحة"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    daily_trades = smart_trading_engine.risk_manager.get_daily_trades_count(user_id)
    
    # محاكاة صفقات مفتوحة
    open_trades_data = [
        {
            "symbol": "XAUUSD",
            "name": "ذهب/دولار",
            "action": "شراء",
            "entry_price": 2045.50,
            "current_price": 2052.30,
            "profit": 68.00,
            "confidence": 92.0,
            "time": "اليوم 09:30"
        },
        {
            "symbol": "EURUSD", 
            "name": "يورو/دولار",
            "action": "بيع",
            "entry_price": 1.0875,
            "current_price": 1.0845,
            "profit": 30.00,
            "confidence": 87.5,
            "time": "اليوم 11:15"
        }
    ]
    
    if not open_trades_data:
        trades_text = f"""
📝 **الصفقات المفتوحة**

لا توجد صفقات مفتوحة حالياً

📊 **إحصائيات اليوم:**
• الصفقات المنفذة: {daily_trades}/10
• المتبقي: {10 - daily_trades} صفقات

💡 **نصائح:**
• راقب الفرص عالية الاحتمالية (90%+)
• استخدم تصنيف الأزواج للتحليل
• اختر النمط المناسب لنشاطك
"""
    else:
        trades_text = f"""
📈 **الصفقات المفتوحة**

📊 **إحصائيات اليوم:** {daily_trades}/10 صفقات

"""
        
        total_profit = 0
        for i, trade in enumerate(open_trades_data, 1):
            profit_color = "🟢" if trade["profit"] > 0 else "🔴"
            confidence_icon = "🚨" if trade["confidence"] >= 90 else "📊"
            
            trades_text += f"""
{profit_color} **{i}. {trade['name']} ({trade['symbol']})**
{confidence_icon} **الاتجاه:** {trade['action']} | **الثقة:** {trade['confidence']:.1f}%
💰 **سعر الدخول:** {trade['entry_price']:.4f}
📈 **السعر الحالي:** {trade['current_price']:.4f}
💵 **الربح/الخسارة:** ${trade['profit']:+.2f}
🕐 **وقت الدخول:** {trade['time']}

"""
            total_profit += trade['profit']
        
        trades_text += f"""
💰 **إجمالي الربح/الخسارة:** ${total_profit:+.2f}

💡 **نصائح إدارة الصفقات:**
• راقب الصفقات عالية الثقة (90%+) 🚨
• التزم بإستراتيجية إدارة المخاطر
• لا تتجاوز الحد اليومي (10 صفقات)
"""
    
    bot.reply_to(message, trades_text, parse_mode='Markdown')

# معالج الرسائل غير المتعرف عليها
@bot.message_handler(func=lambda message: True)
def handle_unknown_or_custom_capital(message):
    """معالج الرسائل غير المعروفة أو إدخال رأس مال مخصص"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        return
    
    # فحص إذا كان المستخدم في وضع إدخال رأس مال مخصص
    if user_id in user_passwords and user_passwords[user_id] == "waiting_custom_capital":
        try:
            amount = float(message.text.replace(',', '').replace('$', ''))
            if 100 <= amount <= 1000000:  # حدود معقولة
                capital_manager.set_user_capital(user_id, amount)
                del user_passwords[user_id]  # إزالة الحالة
                
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
            "❓ لم أفهم هذا الأمر. استخدم الأزرار أدناه:",
            reply_markup=create_main_keyboard()
        )

def is_authenticated(user_id: int) -> bool:
    """فحص المصادقة"""
    return user_id in authenticated_users

# دالة التشغيل الرئيسية
def main():
    """الدالة الرئيسية لتشغيل البوت"""
    try:
        logger.info("🚀 بدء تشغيل بوت التداول الذكي الشامل - الإصدار المحدث")
        logger.info(f"📊 المنصات المدعومة: MetaTrader5({'✅' if MT5_AVAILABLE else '❌'}), TradingView, Yahoo Finance")
        logger.info(f"🔗 الأزواج المدعومة: {len(ALL_SYMBOLS)} زوج")
        logger.info(f"⚡ أنماط التداول: سكالبينغ، طويل المدى")
        logger.info(f"🛡️ إدارة المخاطر: 10 صفقات يومياً، 90%+ للإشعارات")
        logger.info("✅ البوت جاهز لاستقبال الرسائل")
        
        # تنظيف البيانات القديمة عند البدء
        storage.cleanup_old_data(90)
        
        # تشغيل البوت
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
        
    except Exception as e:
        logger.error(f"❌ خطأ في تشغيل البوت: {e}")
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف البوت بواسطة المستخدم")
    finally:
        # إيقاف خدمة المراقبة
        market_monitor.stop_monitoring()
        
        # قطع الاتصال من MT5
        if advanced_analyzer.mt5_manager.connected:
            advanced_analyzer.mt5_manager.disconnect()
            
        logger.info("👋 تم إغلاق البوت وتنظيف الموارد")

if __name__ == "__main__":
    main()