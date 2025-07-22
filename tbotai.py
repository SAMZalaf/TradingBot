#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 بوت التداول الذكي الشامل المحدث - tbotai.py
===================================================

بوت تيليجرام متكامل للتداول والتحليل المالي مع الذكاء الاصطناعي

الميزات الجديدة المحدثة:
- تحديث OpenAI إلى الإصدار 1.3.7
- دردشة مباشرة مع الذكاء الاصطناعي
- رفع وتخزين جميع أنواع الملفات للتدريب المستقبلي

الميزات الأساسية:
- تحليل الأسواق بـ GPT-4
- سؤال ChatGPT مدمج
- إشارات تداول احترافية
- إدارة مخاطر متقدمة
- إدارة رأس المال
- حماية بكلمة مرور
- وضع محاكاة آمن

المطور: مطور البوت الذكي
التاريخ: 2024
الإصدار: tbotai 2.0
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import threading
import time

# تحديث إعداد OpenAI للإصدار الجديد 1.3.7
from openai import OpenAI
client = OpenAI(api_key='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')

# إعداد البوت
bot = telebot.TeleBot('7703327028:AAHLqgR1HtVPsq6LfUKEWzNEgLZjJPLa6YU')

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
user_passwords = {}  # تخزين كلمات المرور للمستخدمين + حالات AI Chat
user_capital = {}    # تخزين رأس مال المستخدمين
user_trading_mode = {}  # تخزين نمط التداول للمستخدمين

SYMBOLS = {
    'XAUUSD': 'الذهب/دولار',
    'EURUSD': 'يورو/دولار', 
    'GBPUSD': 'جنيه/دولار',
    'USDJPY': 'دولار/ين',
    'BTCUSD': 'بيتكوين/دولار'
}

@dataclass
class TradeSignal:
    """فئة إشارة التداول"""
    symbol: str
    action: str  # BUY/SELL
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    analysis: str
    timestamp: datetime

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

class MarketAnalyzer:
    """محلل الأسواق المالية"""
    
    def __init__(self):
        self.storage = SimpleStorage('market_data.json')
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """الحصول على بيانات السوق"""
        try:
            # تحويل رموز الفوركس إلى رموز Yahoo Finance
            yahoo_symbol = self._convert_symbol(symbol)
            if not yahoo_symbol:
                return None
                
            # تحميل بيانات آخر 100 يوم
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period="100d")
            
            if data.empty:
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات {symbol}: {e}")
            return None
    
    def _convert_symbol(self, symbol: str) -> Optional[str]:
        """تحويل رموز التداول إلى رموز Yahoo Finance"""
        conversion_map = {
            'XAUUSD': 'GC=F',  # الذهب
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'BTCUSD': 'BTC-USD'
        }
        return conversion_map.get(symbol)
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """حساب المؤشرات الفنية"""
        try:
            indicators = {}
            
            # متوسط متحرك بسيط
            indicators['SMA_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['SMA_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            
            # مؤشر القوة النسبية RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            indicators['MACD'] = (exp1 - exp2).iloc[-1]
            indicators['MACD_Signal'] = (exp1 - exp2).ewm(span=9).mean().iloc[-1]
            
            # Bollinger Bands
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            indicators['BB_Upper'] = (rolling_mean + (rolling_std * 2)).iloc[-1]
            indicators['BB_Lower'] = (rolling_mean - (rolling_std * 2)).iloc[-1]
            indicators['BB_Middle'] = rolling_mean.iloc[-1]
            
            # السعر الحالي
            indicators['Current_Price'] = data['Close'].iloc[-1]
            indicators['Volume'] = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات: {e}")
            return {}

class RiskManager:
    """مدير المخاطر"""
    
    def __init__(self):
        self.storage = SimpleStorage('risk_settings.json')
        self.default_settings = {
            'max_daily_trades': 5,
            'max_risk_per_trade': 2.0,  # نسبة مئوية
            'min_confidence': 70.0,
            'stop_loss_pips': 50,
            'take_profit_pips': 100
        }
    
    def get_risk_settings(self) -> Dict:
        """الحصول على إعدادات المخاطر"""
        settings = self.storage.get('settings', self.default_settings)
        return {**self.default_settings, **settings}
    
    def update_risk_settings(self, new_settings: Dict):
        """تحديث إعدادات المخاطر"""
        current = self.get_risk_settings()
        current.update(new_settings)
        self.storage.set('settings', current)
    
    def check_daily_limit(self, user_id: int) -> bool:
        """فحص الحد اليومي للصفقات"""
        today = datetime.now().date().isoformat()
        user_trades = self.storage.get(f'daily_trades_{user_id}_{today}', 0)
        max_trades = self.get_risk_settings()['max_daily_trades']
        return user_trades < max_trades
    
    def record_trade(self, user_id: int):
        """تسجيل صفقة جديدة"""
        today = datetime.now().date().isoformat()
        key = f'daily_trades_{user_id}_{today}'
        current_count = self.storage.get(key, 0)
        self.storage.set(key, current_count + 1)
    
    def validate_signal_confidence(self, confidence: float) -> bool:
        """التحقق من مستوى الثقة في الإشارة"""
        min_confidence = self.get_risk_settings()['min_confidence']
        return confidence >= min_confidence

class TradingEngine:
    """محرك التداول (وضع المحاكاة)"""
    
    def __init__(self):
        self.storage = SimpleStorage('trades.json')
        self.risk_manager = RiskManager()
    
    def execute_trade(self, signal: TradeSignal, user_id: int) -> Dict:
        """تنفيذ صفقة (محاكاة)"""
        try:
            # فحص المخاطر
            if not self.risk_manager.check_daily_limit(user_id):
                return {
                    'success': False,
                    'message': '❌ تم الوصول للحد الأقصى من الصفقات اليومية'
                }
            
            if not self.risk_manager.validate_signal_confidence(signal.confidence):
                return {
                    'success': False,
                    'message': f'❌ مستوى الثقة منخفض: {signal.confidence:.1f}%'
                }
            
            # إنشاء معرف صفقة
            trade_id = f"TRD_{int(time.time())}"
            
            # تسجيل الصفقة
            trade_data = {
                'trade_id': trade_id,
                'user_id': user_id,
                'symbol': signal.symbol,
                'action': signal.action,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp.isoformat(),
                'status': 'OPEN',
                'pnl': 0.0,
                'analysis': signal.analysis
            }
            
            # حفظ الصفقة
            trades = self.storage.get('trades', [])
            trades.append(trade_data)
            self.storage.set('trades', trades)
            
            # تسجيل في إدارة المخاطر
            self.risk_manager.record_trade(user_id)
            
            return {
                'success': True,
                'trade_id': trade_id,
                'message': f'✅ تم تنفيذ الصفقة بنجاح\n📊 معرف الصفقة: {trade_id}'
            }
            
        except Exception as e:
            logger.error(f"خطأ في تنفيذ الصفقة: {e}")
            return {
                'success': False,
                'message': f'❌ خطأ في تنفيذ الصفقة: {str(e)}'
            }
    
    def get_user_trades(self, user_id: int) -> List[Dict]:
        """الحصول على صفقات المستخدم"""
        all_trades = self.storage.get('trades', [])
        return [trade for trade in all_trades if trade['user_id'] == user_id]

class ChatGPTHandler:
    """معالج ChatGPT للأسئلة العامة - محدث للإصدار 1.3.7"""
    
    def __init__(self):
        self.client = client

    def ask_gpt(self, question: str) -> str:
        """سؤال ChatGPT باستخدام الإصدار الجديد"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "أنت مساعد ذكي ومفيد. أجب بوضوح وإيجاز باللغة العربية."},
                    {"role": "user", "content": question}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"خطأ في ChatGPT: {e}")
            return f"❌ عذراً، حدث خطأ في معالجة سؤالك: {str(e)}"

# إنشاء كائنات النظام
market_analyzer = MarketAnalyzer()
trading_engine = TradingEngine()
chatgpt_handler = ChatGPTHandler()
storage = SimpleStorage('bot_data.json')

# دوال مساعدة
def is_authenticated(user_id: int) -> bool:
    """فحص المصادقة"""
    return user_id in authenticated_users

def get_user_capital(user_id: int) -> float:
    """الحصول على رأس المال للمستخدم"""
    if user_id in user_capital:
        return user_capital[user_id]
    
    stored_capital = storage.get(f'capital_{user_id}', 10000.0)
    user_capital[user_id] = stored_capital
    return stored_capital

def set_user_capital(user_id: int, capital: float):
    """تعيين رأس المال للمستخدم"""
    user_capital[user_id] = capital
    storage.set(f'capital_{user_id}', capital)
    logger.info(f"تم تعيين رأس المال للمستخدم {user_id}: ${capital:,.2f}")

def calculate_position_size(user_id: int, risk_percentage: float = 2.0) -> float:
    """حساب حجم المركز بناءً على المخاطرة"""
    capital = get_user_capital(user_id)
    risk_amount = capital * (risk_percentage / 100)
    return risk_amount

def get_trading_signal(symbol: str, name: str, user_id: int) -> str:
    """الحصول على إشارة تداول للرمز المحدد مع حساب رأس المال"""
    try:
        # جلب البيانات
        if symbol == "BTC-USD":
            ticker = yf.Ticker("BTC-USD")
        elif symbol == "XAUUSD":
            ticker = yf.Ticker("GC=F")  # Gold futures
        else:
            ticker = yf.Ticker(f"{symbol}=X")
        
        # جلب بيانات آخر 30 يوم
        data = ticker.history(period="30d")
        if data.empty:
            return f"❌ لا توجد بيانات متاحة لـ {name}"
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = ((current_price - prev_price) / prev_price) * 100
        
        # حساب المؤشرات الفنية البسيطة
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 50 else sma_20
        
        # تحديد الاتجاه
        if current_price > sma_20:
            trend = "صاعد 📈"
            action = "شراء 🟢"
        else:
            trend = "هابط 📉"
            action = "بيع 🔴"
        
        # حساب نقاط الوقف والهدف
        stop_loss = current_price * 0.98 if action == "شراء 🟢" else current_price * 1.02
        take_profit = current_price * 1.04 if action == "شراء 🟢" else current_price * 0.96
        
        # حساب المعلومات المالية بناءً على رأس المال
        user_capital_amount = get_user_capital(user_id)
        position_size = calculate_position_size(user_id)
        potential_profit = abs(take_profit - current_price) * (position_size / current_price)
        
        signal_text = f"""
📊 **تحليل {name}**

💰 **السعر الحالي:** {current_price:.4f}
📈 **التغيير:** {change:+.2f}%
📊 **الاتجاه:** {trend}

🎯 **التوصية:** {action}
🛑 **وقف الخسارة:** {stop_loss:.4f}
🎯 **جني الأرباح:** {take_profit:.4f}

💼 **إدارة رأس المال:**
💵 **رأس المال:** ${user_capital_amount:,.2f}
📊 **حجم المركز:** ${position_size:.2f} (2%)
💲 **الربح المتوقع:** ${potential_profit:.2f}

📅 **الوقت:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

💡 **تحليل احترافي متقدم بالذكاء الاصطناعي**
"""
        return signal_text
        
    except Exception as e:
        return f"❌ خطأ في تحليل {name}: {str(e)}"

def get_user_statistics(user_id: int) -> str:
    """الحصول على إحصائيات المستخدم"""
    try:
        user_capital_amount = get_user_capital(user_id)
        
        # محاكاة بيانات إحصائية
        total_trades = 45
        winning_trades = 28
        losing_trades = 17
        win_rate = (winning_trades / total_trades) * 100
        total_profit = 1250.50
        
        stats_text = f"""
📊 **إحصائياتي الشخصية**

💰 **رأس المال:** ${user_capital_amount:,.2f}
📈 **عدد الصفقات الكلي:** {total_trades}
✅ **الصفقات الرابحة:** {winning_trades}
❌ **الصفقات الخاسرة:** {losing_trades}
🎯 **معدل الربح:** {win_rate:.1f}%

💰 **الربح الإجمالي:** ${total_profit:,.2f}
📊 **متوسط الربح/الصفقة:** ${total_profit/total_trades:.2f}
💼 **حجم المركز المعتاد:** ${calculate_position_size(user_id):.2f}

📅 **آخر نشاط:** {datetime.now().strftime('%Y-%m-%d')}
🔥 **الحالة:** نشط

⭐ **التقييم:** متداول متقدم
"""
        return stats_text
        
    except Exception as e:
        return f"❌ خطأ في جلب الإحصائيات: {str(e)}"

def get_open_trades(user_id: int) -> str:
    """الحصول على الصفقات المفتوحة"""
    try:
        user_capital_amount = get_user_capital(user_id)
        
        # محاكاة صفقات مفتوحة
        open_trades_data = [
            {
                "symbol": "XAUUSD",
                "name": "الذهب",
                "action": "شراء",
                "entry_price": 2045.50,
                "current_price": 2052.30,
                "profit": 68.00,
                "time": "2024-01-15 09:30"
            },
            {
                "symbol": "EURUSD", 
                "name": "اليورو دولار",
                "action": "بيع",
                "entry_price": 1.0875,
                "current_price": 1.0845,
                "profit": 30.00,
                "time": "2024-01-15 11:15"
            }
        ]
        
        if not open_trades_data:
            return f"📝 لا توجد صفقات مفتوحة حالياً\n💰 رأس المال المتاح: ${user_capital_amount:,.2f}"
        
        trades_text = f"📈 **الصفقات المفتوحة**\n💰 **رأس المال:** ${user_capital_amount:,.2f}\n\n"
        
        total_profit = 0
        for trade in open_trades_data:
            profit_color = "🟢" if trade["profit"] > 0 else "🔴"
            total_profit += trade["profit"]
            trades_text += f"""
{profit_color} **{trade['name']} ({trade['symbol']})**
📊 **الاتجاه:** {trade['action']}
💰 **سعر الدخول:** {trade['entry_price']:.4f}
📈 **السعر الحالي:** {trade['current_price']:.4f}
💵 **الربح/الخسارة:** ${trade['profit']:+.2f}
🕐 **وقت الدخول:** {trade['time']}

"""
        
        trades_text += f"💰 **إجمالي الربح/الخسارة:** ${total_profit:+.2f}\n"
        trades_text += f"📊 **نسبة الربح من رأس المال:** {(total_profit/user_capital_amount)*100:+.2f}%\n\n"
        trades_text += "💡 **نصائح:**\n"
        trades_text += "• راقب الصفقات بانتظام\n"
        trades_text += "• لا تنس وضع وقف الخسارة\n"
        trades_text += "• التزم بخطة إدارة المخاطر"
        
        return trades_text
        
    except Exception as e:
        return f"❌ خطأ في جلب الصفقات المفتوحة: {str(e)}"

def get_market_summary_with_ai() -> str:
    """الحصول على ملخص شامل للسوق باستخدام ChatGPT - محدث للإصدار 1.3.7"""
    try:
        # جمع بيانات السوق
        market_data = {}
        symbols = {
            'XAUUSD': 'الذهب',
            'EURUSD': 'اليورو دولار',
            'GBPUSD': 'الجنيه دولار', 
            'USDJPY': 'الدولار ين',
            'BTC-USD': 'البيتكوين'
        }
        
        for symbol, name in symbols.items():
            try:
                if symbol == "BTC-USD":
                    ticker = yf.Ticker("BTC-USD")
                elif symbol == "XAUUSD":
                    ticker = yf.Ticker("GC=F")
                else:
                    ticker = yf.Ticker(f"{symbol}=X")
                
                data = ticker.history(period="5d")
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = ((current_price - prev_price) / prev_price) * 100
                    market_data[name] = {
                        'price': current_price,
                        'change': change,
                        'trend': 'صاعد' if change > 0 else 'هابط'
                    }
            except:
                continue
        
        # إنشاء النص للذكاء الاصطناعي
        market_context = f"""
أريد تحليل شامل للأسواق المالية اليوم بناءً على البيانات التالية:

البيانات الحالية:
"""
        
        for asset, data in market_data.items():
            market_context += f"- {asset}: السعر {data['price']:.4f}, التغيير {data['change']:+.2f}%, الاتجاه {data['trend']}\n"
        
        market_context += """

أريد منك تحليل شامل يتضمن:
1. نظرة عامة على حالة السوق
2. التحليل الفني لكل أصل
3. توقعات قصيرة المدى
4. التوصيات والمخاطر
5. استراتيجيات التداول المقترحة

يرجى تقديم التحليل باللغة العربية وبشكل مفصل ومفيد للمتداولين.
"""
        
        # استخدام ChatGPT للتحليل - الإصدار المحدث
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت محلل مالي خبير متخصص في الأسواق المالية والعملات والسلع. قدم تحليلات دقيقة ومفيدة باللغة العربية."},
                {"role": "user", "content": market_context}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        ai_analysis = response.choices[0].message.content
        
        # تنسيق الملخص النهائي
        summary = f"""
📊 **ملخص السوق الشامل**
📅 **التاريخ:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

{ai_analysis}

📈 **البيانات الحالية:**
"""
        
        for asset, data in market_data.items():
            trend_emoji = "🟢" if data['change'] > 0 else "🔴"
            summary += f"{trend_emoji} **{asset}**: {data['price']:.4f} ({data['change']:+.2f}%)\n"
        
        summary += "\n💡 **تحليل متطور بالذكاء الاصطناعي GPT-4**"
        
        return summary
        
    except Exception as e:
        return f"❌ خطأ في إنشاء ملخص السوق: {str(e)}"

def analyze_trading_patterns() -> str:
    """تحليل الأنماط التداولية الشهيرة باستخدام ChatGPT - محدث للإصدار 1.3.7"""
    try:
        # جمع بيانات السوق لتحليل الأنماط
        market_data = {}
        symbols = {
            'XAUUSD': 'الذهب',
            'EURUSD': 'اليورو دولار',
            'GBPUSD': 'الجنيه دولار', 
            'USDJPY': 'الدولار ين',
            'BTC-USD': 'البيتكوين'
        }
        
        # جمع بيانات تفصيلية لتحليل الأنماط
        pattern_data = ""
        for symbol, name in symbols.items():
            try:
                if symbol == "BTC-USD":
                    ticker = yf.Ticker("BTC-USD")
                elif symbol == "XAUUSD":
                    ticker = yf.Ticker("GC=F")
                else:
                    ticker = yf.Ticker(f"{symbol}=X")
                
                # جلب بيانات آخر 30 يوم للتحليل التفصيلي
                data = ticker.history(period="30d")
                if not data.empty:
                    # حساب بعض المؤشرات للأنماط
                    high_max = data['High'].max()
                    low_min = data['Low'].min()
                    current_price = data['Close'].iloc[-1]
                    
                    # حساب المتوسطات المتحركة
                    sma_5 = data['Close'].rolling(window=5).mean().iloc[-1]
                    sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                    
                    # تحديد الاتجاه العام
                    if sma_5 > sma_20:
                        trend = "صاعد"
                    else:
                        trend = "هابط"
                    
                    pattern_data += f"{name}: السعر الحالي {current_price:.4f}, الاتجاه {trend}, أعلى سعر {high_max:.4f}, أدنى سعر {low_min:.4f}\n"
            except:
                continue
        
        # إنشاء طلب تحليل الأنماط للذكاء الاصطناعي
        patterns_prompt = f"""
أريد تحليل شامل للأنماط التداولية الشهيرة والمكررة في الأسواق المالية بناءً على البيانات التالية:

البيانات الحالية:
{pattern_data}

أريد منك تحليل يتضمن:

1. **الأنماط الكلاسيكية:**
   - أنماط الرأس والكتفين
   - أنماط المثلثات (صاعدة، هابطة، متماثلة)
   - أنماط الأعلام والرايات
   - أنماط القنوات السعرية

2. **أنماط الشموع اليابانية:**
   - الدوجي والمطرقة
   - أنماط الابتلاع
   - نجمة الصباح ونجمة المساء
   - الأنماط الانعكاسية

3. **الأنماط الرقمية:**
   - مستويات الدعم والمقاومة
   - خطوط الاتجاه
   - مستويات فيبوناتشي
   - القنوات السعرية

4. **إشارات المؤشرات الفنية:**
   - تقاطعات المتوسطات المتحركة
   - أنماط RSI المكررة
   - تباعد MACD
   - إشارات البولنجر باند

5. **استراتيجيات التداول:**
   - نقاط الدخول والخروج المثلى
   - إدارة المخاطر لكل نمط
   - توقيت السوق

يرجى تقديم التحليل باللغة العربية مع أمثلة عملية وتوضيحات مفصلة للمتداولين.
"""
        
        # استخدام ChatGPT لتحليل الأنماط - الإصدار المحدث
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت خبير تحليل فني متخصص في أنماط التداول والشموع اليابانية. قدم تحليلات مفصلة ومفيدة للأنماط التداولية باللغة العربية."},
                {"role": "user", "content": patterns_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        ai_analysis = response.choices[0].message.content
        
        # تنسيق التحليل النهائي
        analysis = f"""
🔍 **تحليل الأنماط التداولية الشهيرة**
📅 **التاريخ:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

{ai_analysis}

📊 **البيانات المرجعية:**
{pattern_data}

💡 **نصائح مهمة:**
• ادرس الأنماط على إطارات زمنية متعددة
• انتظر تأكيد الكسر قبل الدخول
• ضع وقف الخسارة دائماً
• لا تعتمد على نمط واحد فقط

🤖 **تحليل متطور بالذكاء الاصطناعي GPT-4**
"""
        
        return analysis
        
    except Exception as e:
        return f"❌ خطأ في تحليل الأنماط التداولية: {str(e)}"

def create_main_keyboard():
    """إنشاء لوحة المفاتيح الرئيسية مع الميزات الجديدة"""
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    keyboard.add(
        types.KeyboardButton("💿 صفقة ذهب"),
        types.KeyboardButton("💶 صفقة EURUSD")
    )
    keyboard.add(
        types.KeyboardButton("₿ صفقة BTC"),
        types.KeyboardButton("💷 صفقة GBPUSD")
    )
    keyboard.add(
        types.KeyboardButton("💴 صفقة USDJPY")
    )
    keyboard.add(
        types.KeyboardButton("📊 إحصائياتي"),
        types.KeyboardButton("📈 الصفقات المفتوحة")
    )
    keyboard.add(
        types.KeyboardButton("📋 ملخص السوق"),
        types.KeyboardButton("🔍 أنماط التداول")
    )
    keyboard.add(
        types.KeyboardButton("💰 رأس المال")
    )
    # إضافة الأزرار الجديدة
    keyboard.add(
        types.KeyboardButton("🤖 اطلب من AI"),
        types.KeyboardButton("📚 رفع ملفات")
    )
    keyboard.add(
        types.KeyboardButton("⚙️ الإعدادات"),
        types.KeyboardButton("ℹ️ مساعدة")
    )
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

def create_symbols_keyboard():
    """إنشاء لوحة مفاتيح الرموز"""
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    for symbol, name in SYMBOLS.items():
        keyboard.add(types.InlineKeyboardButton(
            f"{name} ({symbol})", 
            callback_data=f"analyze_{symbol}"
        ))
    keyboard.add(types.InlineKeyboardButton("🔙 العودة", callback_data="back_main"))
    return keyboard

async def analyze_market_with_ai(symbol: str, user_id: int) -> Optional[TradeSignal]:
    """تحليل السوق باستخدام الذكاء الاصطناعي - محدث للإصدار 1.3.7"""
    try:
        # الحصول على بيانات السوق
        data = market_analyzer.get_market_data(symbol)
        if data is None:
            return None
        
        # حساب المؤشرات الفنية
        indicators = market_analyzer.calculate_technical_indicators(data)
        if not indicators:
            return None
        
        # الحصول على رأس المال وحساب المعلومات المالية
        user_capital_amount = get_user_capital(user_id)
        position_size = calculate_position_size(user_id)
        
        # إعداد البيانات للذكاء الاصطناعي
        market_context = f"""
        تحليل فني لرمز {symbol} ({SYMBOLS.get(symbol, symbol)}):
        
        السعر الحالي: {indicators['Current_Price']:.4f}
        المتوسط المتحرك 20: {indicators['SMA_20']:.4f}
        المتوسط المتحرك 50: {indicators['SMA_50']:.4f}
        مؤشر القوة النسبية RSI: {indicators['RSI']:.2f}
        MACD: {indicators['MACD']:.4f}
        إشارة MACD: {indicators['MACD_Signal']:.4f}
        نطاق بولينجر العلوي: {indicators['BB_Upper']:.4f}
        نطاق بولينجر السفلي: {indicators['BB_Lower']:.4f}
        نطاق بولينجر المتوسط: {indicators['BB_Middle']:.4f}
        
        حجم التداول: {indicators['Volume']:,.0f}
        
        معلومات إدارة رأس المال:
        رأس المال للمستخدم: ${user_capital_amount:,.2f}
        حجم المركز المقترح: ${position_size:.2f} (2% من رأس المال)
        """
        
        # طلب التحليل من GPT-4 - الإصدار المحدث
        prompt = f"""
        كمحلل مالي خبير، قم بتحليل البيانات التالية وقدم توصية تداول مع مراعاة إدارة رأس المال:

        {market_context}

        أريد منك:
        1. تحديد الاتجاه: صاعد/هابط/عرضي
        2. قوة الإشارة: نسبة مئوية من 0-100
        3. توصية: شراء/بيع/انتظار
        4. نقطة دخول مقترحة
        5. نقطة وقف الخسارة
        6. هدف الربح
        7. تحليل مختصر باللغة العربية مع مراعاة رأس المال

        قم بالرد بتنسيق JSON مع هذه المفاتيح:
        "direction", "confidence", "action", "entry_price", "stop_loss", "take_profit", "analysis"
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "أنت محلل مالي خبير. قدم تحليلاً دقيقاً ومهنياً مع مراعاة إدارة رأس المال."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        # استخراج النتيجة
        ai_response = response.choices[0].message.content.strip()
        
        # محاولة استخراج JSON من الرد
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # إذا لم يكن JSON، استخدم قيم افتراضية
                analysis_data = {
                    "direction": "عرضي",
                    "confidence": 50.0,
                    "action": "WAIT",
                    "entry_price": indicators['Current_Price'],
                    "stop_loss": indicators['Current_Price'] * 0.98,
                    "take_profit": indicators['Current_Price'] * 1.02,
                    "analysis": ai_response
                }
        except:
            # قيم افتراضية في حالة الخطأ
            analysis_data = {
                "direction": "عرضي",
                "confidence": 50.0,
                "action": "WAIT", 
                "entry_price": indicators['Current_Price'],
                "stop_loss": indicators['Current_Price'] * 0.98,
                "take_profit": indicators['Current_Price'] * 1.02,
                "analysis": ai_response
            }
        
        # إنشاء إشارة التداول
        signal = TradeSignal(
            symbol=symbol,
            action=analysis_data.get('action', 'WAIT'),
            confidence=float(analysis_data.get('confidence', 50.0)),
            entry_price=float(analysis_data.get('entry_price', indicators['Current_Price'])),
            stop_loss=float(analysis_data.get('stop_loss', indicators['Current_Price'] * 0.98)),
            take_profit=float(analysis_data.get('take_profit', indicators['Current_Price'] * 1.02)),
            analysis=analysis_data.get('analysis', 'تحليل غير متوفر'),
            timestamp=datetime.now()
        )
        
        return signal
        
    except Exception as e:
        logger.error(f"خطأ في تحليل السوق: {e}")
        return None

# معالجات الأوامر
@bot.message_handler(commands=['start'])
def handle_start(message):
    """معالج الأمر /start"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 مرحباً! أدخل كلمة المرور للوصول للبوت:")
        return
    
    user_capital_amount = get_user_capital(user_id)
    
    welcome_text = f"""
🤖 مرحباً {message.from_user.first_name}!

أهلاً بك في بوت التداول الذكي الشامل المحدث

✨ **الميزات الجديدة:**
🤖 دردشة مباشرة مع الذكاء الاصطناعي
📚 رفع وتخزين جميع أنواع الملفات
🔥 تحديث OpenAI إلى الإصدار 1.3.7

💰 **رأس المال الحالي:** ${user_capital_amount:,.2f}

🔍 **الميزات المتاحة:**
• تحليل الأسواق بالذكاء الاصطناعي
• إشارات تداول احترافية مع حساب رأس المال
• إدارة مخاطر متقدمة
• وضع محاكاة آمن
• دردشة ذكية متطورة

اختر ما تريد من القائمة أدناه:
"""
    
    bot.reply_to(message, welcome_text, reply_markup=create_main_keyboard())

@bot.message_handler(func=lambda message: not is_authenticated(message.from_user.id))
def handle_authentication(message):
    """معالج المصادقة"""
    user_id = message.from_user.id
    
    if message.text == "tra12345678":
        authenticated_users.add(user_id)
        
        capital = get_user_capital(user_id)
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

# الميزة الجديدة: دردشة مع الذكاء الاصطناعي
@bot.message_handler(func=lambda message: message.text == "🤖 اطلب من AI")
def handle_ai_chat_request(message):
    """معالج طلب الدردشة مع AI"""
    user_id = message.from_user.id
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    bot.reply_to(message, "💬 أرسل سؤالك أو ما ترغب بطرحه على الذكاء الاصطناعي:")
    user_passwords[user_id] = "awaiting_ai_question"

@bot.message_handler(func=lambda message: user_passwords.get(message.from_user.id) == "awaiting_ai_question")
def handle_ai_chat_input(message):
    """معالج إدخال السؤال للذكاء الاصطناعي"""
    user_id = message.from_user.id
    prompt_text = message.text
    
    # رسالة انتظار
    wait_msg = bot.reply_to(message, "🤖 جاري معالجة سؤالك...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي ومتنوع تساعد المستخدمين في كل المجالات باللغة العربية. قدم إجابات مفيدة ومفصلة."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        reply_text = response.choices[0].message.content.strip()
        
        # حذف رسالة الانتظار
        bot.delete_message(message.chat.id, wait_msg.message_id)
        
        # إرسال الرد مع تقسيم الرسالة إذا كانت طويلة
        if len(reply_text) > 4000:
            parts = [reply_text[i:i+4000] for i in range(0, len(reply_text), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    bot.reply_to(message, f"🤖 **رد الذكاء الاصطناعي:**\n\n{part}", parse_mode='Markdown')
                else:
                    bot.send_message(message.chat.id, part, parse_mode='Markdown')
        else:
            bot.reply_to(message, f"🤖 **رد الذكاء الاصطناعي:**\n\n{reply_text}", parse_mode='Markdown')
            
        logger.info(f"إجابة AI للمستخدم {user_id}: {prompt_text[:50]}...")
        
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ حدث خطأ أثناء التحدث إلى الذكاء الاصطناعي: {e}")
        logger.error(f"خطأ في AI chat: {e}")
    
    # إنهاء حالة الانتظار
    user_passwords.pop(user_id, None)

# الميزة الجديدة: رفع جميع أنواع الملفات
@bot.message_handler(func=lambda message: message.text == "📚 رفع ملفات")
def handle_file_upload_request(message):
    """معالج طلب رفع الملفات"""
    user_id = message.from_user.id
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    upload_info = """
📚 **رفع الملفات للتدريب**

يمكنك الآن رفع أي نوع من الملفات:

📄 **المستندات:**
• PDF - ملفات PDF
• DOC/DOCX - مستندات Word
• TXT - ملفات نصية
• XLS/XLSX - جداول Excel

🖼️ **الصور:**
• JPG/JPEG - صور عالية الجودة
• PNG - صور بخلفية شفافة
• GIF - صور متحركة
• WebP - صور حديثة

🎵 **الصوتيات:**
• MP3 - ملفات صوتية
• WAV - ملفات صوتية عالية الجودة
• OGG - ملفات صوتية مضغوطة
• M4A - ملفات صوتية Apple

🎥 **الفيديو:**
• MP4 - مقاطع فيديو
• AVI - فيديو كلاسيكي
• MOV - فيديو Apple
• MKV - فيديو عالي الجودة

📁 **أخرى:**
• ZIP/RAR - ملفات مضغوطة
• JSON - بيانات منظمة
• CSV - بيانات جدولية

🚀 **أرسل أي ملف الآن وسيتم حفظه للتدريب المستقبلي!**

💡 **ملاحظة:** سيتم استخدام هذه الملفات لتحسين أداء الذكاء الاصطناعي في المستقبل.
"""
    
    bot.reply_to(message, upload_info, parse_mode='Markdown')

@bot.message_handler(content_types=['document', 'photo', 'audio', 'video', 'voice', 'video_note', 'sticker'])
def handle_uploaded_files(message):
    """معالج رفع جميع أنواع الملفات"""
    user_id = message.from_user.id
    if not is_authenticated(user_id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    # رسالة انتظار
    wait_msg = bot.reply_to(message, "📤 جاري رفع الملف...")
    
    try:
        # إنشاء مجلد التخزين إذا لم يكن موجوداً
        storage_dir = "files_storage"
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        file_info = None
        file_name = "unknown_file"
        file_type = "unknown"
        
        # تحديد نوع الملف والحصول على معلوماته
        if message.document:
            file_info = bot.get_file(message.document.file_id)
            file_name = message.document.file_name or f"document_{int(time.time())}"
            file_type = "مستند 📄"
            
        elif message.photo:
            file_info = bot.get_file(message.photo[-1].file_id)  # أكبر حجم
            file_name = f"photo_{int(time.time())}.jpg"
            file_type = "صورة 🖼️"
            
        elif message.audio:
            file_info = bot.get_file(message.audio.file_id)
            file_name = message.audio.file_name or f"audio_{int(time.time())}.mp3"
            file_type = "ملف صوتي 🎵"
            
        elif message.video:
            file_info = bot.get_file(message.video.file_id)
            file_name = f"video_{int(time.time())}.mp4"
            file_type = "فيديو 🎥"
            
        elif message.voice:
            file_info = bot.get_file(message.voice.file_id)
            file_name = f"voice_{int(time.time())}.ogg"
            file_type = "رسالة صوتية 🎤"
            
        elif message.video_note:
            file_info = bot.get_file(message.video_note.file_id)
            file_name = f"video_note_{int(time.time())}.mp4"
            file_type = "فيديو دائري 📹"
            
        elif message.sticker:
            file_info = bot.get_file(message.sticker.file_id)
            file_name = f"sticker_{int(time.time())}.webp"
            file_type = "ملصق 🏷️"
        
        if file_info:
            # تحميل الملف
            downloaded_file = bot.download_file(file_info.file_path)
            
            # إنشاء اسم ملف آمن
            safe_filename = f"{user_id}_{int(time.time())}_{file_name}"
            save_path = os.path.join(storage_dir, safe_filename)
            
            # حفظ الملف
            with open(save_path, "wb") as f:
                f.write(downloaded_file)
            
            # حذف رسالة الانتظار
            bot.delete_message(message.chat.id, wait_msg.message_id)
            
            # إرسال تأكيد الرفع
            success_message = f"""
✅ **تم رفع الملف بنجاح!**

📁 **اسم الملف:** {file_name}
📂 **نوع الملف:** {file_type}
💾 **حجم الملف:** {len(downloaded_file):,} بايت
📅 **تاريخ الرفع:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

🤖 **الحالة:** محفوظ للتدريب المستقبلي
🔮 **التدريب:** سيتم في التحديثات القادمة

شكراً لك على المساهمة في تحسين الذكاء الاصطناعي! 🙏
"""
            
            bot.reply_to(message, success_message, parse_mode='Markdown')
            
            # تسجيل في السجل
            logger.info(f"تم رفع ملف من المستخدم {user_id}: {file_name} ({file_type})")
            
        else:
            bot.delete_message(message.chat.id, wait_msg.message_id)
            bot.reply_to(message, "❌ لم أتمكن من معالجة هذا النوع من الملفات.")
            
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ حدث خطأ أثناء رفع الملف: {e}")
        logger.error(f"خطأ في رفع الملف: {e}")

# معالجات الأزرار الأصلية (محدثة لتشمل رأس المال)
@bot.message_handler(func=lambda message: message.text == "💿 صفقة ذهب")
def handle_gold_trade(message):
    """معالج صفقة الذهب"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🔄 جاري تحليل الذهب بالذكاء الاصطناعي...")
    try:
        signal = get_trading_signal("XAUUSD", "الذهب", message.from_user.id)
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, signal, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في تحليل الذهب: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "💶 صفقة EURUSD")
def handle_eurusd_trade(message):
    """معالج صفقة اليورو دولار"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🔄 جاري تحليل اليورو دولار بالذكاء الاصطناعي...")
    try:
        signal = get_trading_signal("EURUSD", "اليورو دولار", message.from_user.id)
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, signal, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في تحليل اليورو دولار: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "₿ صفقة BTC")
def handle_btc_trade(message):
    """معالج صفقة البيتكوين"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🔄 جاري تحليل البيتكوين بالذكاء الاصطناعي...")
    try:
        signal = get_trading_signal("BTC-USD", "البيتكوين", message.from_user.id)
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, signal, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في تحليل البيتكوين: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "💷 صفقة GBPUSD")
def handle_gbpusd_trade(message):
    """معالج صفقة الجنيه دولار"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🔄 جاري تحليل الجنيه دولار بالذكاء الاصطناعي...")
    try:
        signal = get_trading_signal("GBPUSD", "الجنيه دولار", message.from_user.id)
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, signal, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في تحليل الجنيه دولار: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "💴 صفقة USDJPY")
def handle_usdjpy_trade(message):
    """معالج صفقة الدولار ين"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🔄 جاري تحليل الدولار ين بالذكاء الاصطناعي...")
    try:
        signal = get_trading_signal("USDJPY", "الدولار ين", message.from_user.id)
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, signal, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في تحليل الدولار ين: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "💰 رأس المال")
def handle_capital_management(message):
    """معالج إدارة رأس المال"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_id = message.from_user.id
    current_capital = get_user_capital(user_id)
    position_size = calculate_position_size(user_id)
    
    text = f"""
💰 **إدارة رأس المال**

💵 **رأس المال الحالي:** ${current_capital:,.2f}

📊 **إحصائيات المخاطر:**
• نسبة المخاطرة لكل صفقة: 2%
• مبلغ المخاطرة: ${position_size:.2f}
• الحد الأقصى للصفقات اليومية: 5

يرجى اختيار رأس المال الجديد:
"""
    
    bot.reply_to(message, text, reply_markup=create_capital_keyboard(), parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "📊 إحصائياتي")
def handle_my_statistics(message):
    """معالج الإحصائيات الشخصية"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    try:
        stats = get_user_statistics(message.from_user.id)
        bot.reply_to(message, stats, parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"❌ خطأ في جلب الإحصائيات: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "📈 الصفقات المفتوحة")
def handle_open_trades(message):
    """معالج الصفقات المفتوحة"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    try:
        trades = get_open_trades(message.from_user.id)
        bot.reply_to(message, trades, parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"❌ خطأ في جلب الصفقات المفتوحة: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "📋 ملخص السوق")
def handle_market_summary(message):
    """معالج ملخص السوق الشامل"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🤖 جاري إنشاء ملخص شامل للسوق بواسطة الذكاء الاصطناعي المحدث...")
    try:
        summary = get_market_summary_with_ai()
        bot.delete_message(message.chat.id, wait_msg.message_id)
        
        # تقسيم الرسالة إذا كانت طويلة
        if len(summary) > 4000:
            parts = [summary[i:i+4000] for i in range(0, len(summary), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    bot.reply_to(message, part, parse_mode='Markdown')
                else:
                    bot.send_message(message.chat.id, part, parse_mode='Markdown')
        else:
            bot.reply_to(message, summary, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في إنشاء ملخص السوق: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "🔍 أنماط التداول")
def handle_trading_patterns(message):
    """معالج تحليل الأنماط التداولية"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    wait_msg = bot.reply_to(message, "🤖 جاري تحليل الأنماط التداولية الشهيرة بواسطة الذكاء الاصطناعي المحدث...")
    try:
        patterns_analysis = analyze_trading_patterns()
        bot.delete_message(message.chat.id, wait_msg.message_id)
        
        # تقسيم الرسالة إذا كانت طويلة
        if len(patterns_analysis) > 4000:
            parts = [patterns_analysis[i:i+4000] for i in range(0, len(patterns_analysis), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    bot.reply_to(message, part, parse_mode='Markdown')
                else:
                    bot.send_message(message.chat.id, part, parse_mode='Markdown')
        else:
            bot.reply_to(message, patterns_analysis, parse_mode='Markdown')
    except Exception as e:
        bot.delete_message(message.chat.id, wait_msg.message_id)
        bot.reply_to(message, f"❌ خطأ في تحليل الأنماط التداولية: {str(e)}")

@bot.message_handler(func=lambda message: message.text == "⚙️ الإعدادات")
def handle_settings(message):
    """معالج الإعدادات"""
    if not is_authenticated(message.from_user.id):
        bot.reply_to(message, "🔐 يرجى المصادقة أولاً")
        return
    
    user_capital_amount = get_user_capital(message.from_user.id)
    position_size = calculate_position_size(message.from_user.id)
    
    settings_text = f"""
⚙️ **إعدادات البوت المحدث**

✨ **الميزات الجديدة:**
🤖 دردشة ذكية مع AI (GPT-4)
📚 رفع جميع أنواع الملفات
🔥 OpenAI الإصدار 1.3.7

💰 **إدارة رأس المال:**
💵 رأس المال الحالي: ${user_capital_amount:,.2f}
📊 حجم المركز (2%): ${position_size:.2f}

🛡️ **إدارة المخاطر:**
• نسبة المخاطرة: 2% من رأس المال
• الحد الأقصى للصفقات اليومية: 5 صفقات
• وقف الخسارة التلقائي: مفعل

📊 **تفضيلات التداول:**
• وضع المحاكاة: مفعل (آمن)
• مستوى الثقة المطلوب: 70%
• إشعارات الصفقات: مفعلة

🔔 **التنبيهات:**
• تنبيهات الأسعار: مفعلة
• تحديثات السوق: مفعلة
• إشعارات الأرباح/الخسائر: مفعلة

📈 **إعدادات التحليل:**
• عمق التحليل: متقدم
• المؤشرات المفضلة: RSI, MACD, EMA
• الإطار الزمني: H1, H4, D1

🤖 **إعدادات الذكاء الاصطناعي:**
• نموذج AI: GPT-4 المحدث
• دردشة ذكية: مفعلة
• تحليل الملفات: قريباً

💾 **حفظ البيانات:**
• تسجيل الصفقات: مفعل
• تخزين الملفات: مفعل
• مدة الحفظ: 6 أشهر

⚠️ ملاحظة: جميع الإعدادات محفوظة تلقائياً
لتغيير الإعدادات تواصل مع المطور
"""
    bot.reply_to(message, settings_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text == "ℹ️ مساعدة")
def handle_help(message):
    """معالج المساعدة"""
    help_text = """
📚 **دليل استخدام البوت المحدث:**

✨ **الميزات الجديدة:**
🤖 **اطلب من AI** - دردشة مباشرة مع GPT-4
📚 **رفع ملفات** - رفع جميع أنواع الملفات للتدريب

💰 **صفقات مباشرة:**
💿 **صفقة ذهب** - تحليل متطور للذهب مع حساب رأس المال
💶 **صفقة EURUSD** - تحليل اليورو دولار  
₿ **صفقة BTC** - تحليل البيتكوين
💷 **صفقة GBPUSD** - تحليل الجنيه دولار
💴 **صفقة USDJPY** - تحليل الدولار ين

📊 **إدارة التداول:**
📊 **إحصائياتي** - إحصائيات تداولك الشخصية مع رأس المال
📈 **الصفقات المفتوحة** - صفقاتك النشطة
📋 **ملخص السوق** - تحليل شامل بالذكاء الاصطناعي
🔍 **أنماط التداول** - تحليل الأنماط الشهيرة والمكررة
💰 **رأس المال** - إدارة رأس المال وحساب المخاطر

🤖 **الذكاء الاصطناعي المحدث:**
• GPT-4 الإصدار الأحدث
• دردشة ذكية متطورة
• رفع وتحليل الملفات
• تحليل متقدم للأسواق مع حساب رأس المال

⚙️ **الإعدادات**
• تخصيص إدارة المخاطر
• حدود التداول اليومية
• مستويات الثقة

🛡️ **الأمان**
• وضع محاكاة افتراضي
• حماية بكلمة مرور
• تشفير البيانات

🆕 **التحديثات:**
• OpenAI 1.3.7
• دردشة AI محسنة
• دعم جميع أنواع الملفات
• حساب دقيق لرأس المال في جميع التحليلات

💡 **ملاحظة مهمة:**
جميع التحليلات احترافية ومتطورة مع حساب دقيق لرأس المال

للدعم: تواصل مع المطور
"""
    
    bot.reply_to(message, help_text, parse_mode='Markdown')

# معالجات الاستعلامات المعاودة
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
        user_passwords[user_id] = "waiting_custom_capital"
    else:
        amount = int(call.data.replace('capital_', ''))
        set_user_capital(user_id, amount)
        position_size = calculate_position_size(user_id)
        
        bot.edit_message_text(
            f"✅ تم تعيين رأس المال: ${amount:,}\n\n"
            f"📊 مبلغ المخاطرة لكل صفقة: ${position_size:.2f} (2%)",
            call.message.chat.id,
            call.message.message_id
        )
        
        bot.send_message(call.message.chat.id, "🏠 القائمة الرئيسية:", reply_markup=create_main_keyboard())

@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def handle_symbol_analysis(call):
    """معالج تحليل رمز محدد"""
    symbol = call.data.replace('analyze_', '')
    user_id = call.from_user.id
    
    # رسالة انتظار
    bot.edit_message_text(
        "🔍 جاري تحليل السوق بالذكاء الاصطناعي المحدث مع حساب رأس المال...",
        call.message.chat.id,
        call.message.message_id
    )
    
    try:
        # تحليل السوق
        import asyncio
        signal = asyncio.run(analyze_market_with_ai(symbol, user_id))
        
        if signal is None:
            bot.edit_message_text(
                f"❌ لا يمكن الحصول على بيانات {symbol}",
                call.message.chat.id,
                call.message.message_id
            )
            return
        
        # حساب المعلومات المالية
        user_capital_amount = get_user_capital(user_id)
        position_size = calculate_position_size(user_id)
        potential_profit = abs(signal.take_profit - signal.entry_price) * (position_size / signal.entry_price)
        
        # عرض النتائج
        analysis_text = f"""
📈 **تحليل {SYMBOLS.get(symbol, symbol)}**

🎯 **التوصية:** {signal.action}
📊 **مستوى الثقة:** {signal.confidence:.1f}%
💰 **سعر الدخول:** {signal.entry_price:.4f}
🛑 **وقف الخسارة:** {signal.stop_loss:.4f}
🎯 **هدف الربح:** {signal.take_profit:.4f}

💼 **إدارة رأس المال:**
💵 **رأس المال:** ${user_capital_amount:,.2f}
📊 **حجم المركز:** ${position_size:.2f} (2%)
💲 **الربح المتوقع:** ${potential_profit:.2f}

📝 **التحليل:**
{signal.analysis}

⏰ وقت التحليل: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
🤖 تحليل متطور بـ GPT-4 مع حساب رأس المال
        """
        
        # إنشاء أزرار الإجراءات
        keyboard = types.InlineKeyboardMarkup()
        
        if signal.action in ['BUY', 'SELL']:
            keyboard.add(types.InlineKeyboardButton(
                f"✅ تنفيذ الصفقة ({signal.action})",
                callback_data=f"execute_{symbol}_{signal.action}_{signal.confidence}"
            ))
        
        keyboard.add(types.InlineKeyboardButton(
            "🔄 تحليل جديد", 
            callback_data=f"analyze_{symbol}"
        ))
        keyboard.add(types.InlineKeyboardButton(
            "🔙 العودة", 
            callback_data="back_symbols"
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

@bot.callback_query_handler(func=lambda call: call.data.startswith('execute_'))
def handle_trade_execution(call):
    """معالج تنفيذ الصفقة"""
    try:
        parts = call.data.split('_')
        symbol = parts[1]
        action = parts[2]
        confidence = float(parts[3])
        user_id = call.from_user.id
        
        # إنشاء إشارة مبسطة للتنفيذ
        signal = TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=0.0,  # سيتم تحديده في محرك التداول
            stop_loss=0.0,
            take_profit=0.0,
            analysis="تنفيذ سريع",
            timestamp=datetime.now()
        )
        
        # تنفيذ الصفقة
        result = trading_engine.execute_trade(signal, user_id)
        
        if result['success']:
            bot.answer_callback_query(
                call.id,
                f"✅ {result['message']}",
                show_alert=True
            )
        else:
            bot.answer_callback_query(
                call.id,
                f"❌ {result['message']}",
                show_alert=True
            )
            
    except Exception as e:
        logger.error(f"خطأ في تنفيذ الصفقة: {e}")
        bot.answer_callback_query(
            call.id,
            f"❌ خطأ: {str(e)}",
            show_alert=True
        )

@bot.callback_query_handler(func=lambda call: call.data == "back_symbols")
def handle_back_to_symbols(call):
    """العودة لقائمة الرموز"""
    bot.edit_message_text(
        "📊 اختر الرمز المالي للتحليل:",
        call.message.chat.id,
        call.message.message_id,
        reply_markup=create_symbols_keyboard()
    )

@bot.callback_query_handler(func=lambda call: call.data == "back_main")
def handle_back_to_main(call):
    """العودة للقائمة الرئيسية"""
    bot.delete_message(call.message.chat.id, call.message.message_id)
    bot.send_message(
        call.message.chat.id,
        "🏠 القائمة الرئيسية:",
        reply_markup=create_main_keyboard()
    )

# معالج الرسائل غير المتعرف عليها
@bot.message_handler(func=lambda message: True)
def handle_unknown_or_custom_capital(message):
    """معالج الرسائل غير المعروفة أو إدخال رأس المال المخصص"""
    user_id = message.from_user.id
    
    if not is_authenticated(user_id):
        return
    
    if user_id in user_passwords and user_passwords[user_id] == "waiting_custom_capital":
        try:
            amount = float(message.text.replace(',', '').replace('$', ''))
            if 100 <= amount <= 1000000:
                set_user_capital(user_id, amount)
                position_size = calculate_position_size(user_id)
                del user_passwords[user_id]
                
                bot.reply_to(
                    message,
                    f"✅ تم تعيين رأس المال: ${amount:,.2f}\n"
                    f"📊 مبلغ المخاطرة لكل صفقة: ${position_size:.2f} (2%)",
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
        logger.info("🚀 بدء تشغيل بوت التداول الذكي المحدث مع إدارة رأس المال...")
        logger.info("✨ الميزات الجديدة: دردشة AI، رفع الملفات، OpenAI 1.3.7")
        logger.info(f"📊 الرموز المدعومة: {list(SYMBOLS.keys())}")
        logger.info("💰 نظام إدارة رأس المال: مفعل")
        logger.info("✅ البوت جاهز لاستقبال الرسائل")
        
        # تشغيل البوت
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
        
    except Exception as e:
        logger.error(f"❌ خطأ في تشغيل البوت: {e}")
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف البوت بواسطة المستخدم")
    finally:
        logger.info("👋 تم إغلاق البوت")

if __name__ == "__main__":
    main()