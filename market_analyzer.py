import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from openai import OpenAI
from config import Config
from logger_config import market_logger
import talib

class MarketAnalyzer:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.symbol_mapping = {
            'XAUUSD': 'GC=F',  # Gold futures
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'BTCUSD': 'BTC-USD'
        }
        
    def get_market_data(self, symbol: str, period: str = "1d", interval: str = "1h") -> pd.DataFrame:
        """الحصول على بيانات السوق الحقيقية"""
        try:
            yf_symbol = self.symbol_mapping.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            
            # الحصول على البيانات التاريخية
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                market_logger.warning(f"لا توجد بيانات لـ {symbol}")
                return pd.DataFrame()
            
            market_logger.info(f"تم الحصول على بيانات {symbol}: {len(data)} سجل")
            return data
            
        except Exception as e:
            market_logger.error(f"خطأ في الحصول على بيانات {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """حساب المؤشرات التقنية"""
        if data.empty or len(data) < 20:
            return {}
        
        try:
            indicators = {}
            
            # أسعار الإغلاق والحجم
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            # المتوسطات المتحركة
            indicators['sma_20'] = talib.SMA(close_prices, timeperiod=20)[-1] if len(close_prices) >= 20 else 0
            indicators['ema_12'] = talib.EMA(close_prices, timeperiod=12)[-1] if len(close_prices) >= 12 else 0
            indicators['ema_26'] = talib.EMA(close_prices, timeperiod=26)[-1] if len(close_prices) >= 26 else 0
            
            # مؤشر القوة النسبية RSI
            rsi_values = talib.RSI(close_prices, timeperiod=14)
            indicators['rsi'] = rsi_values[-1] if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]) else 50
            
            # مؤشر MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            indicators['macd'] = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
            indicators['macd_signal'] = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
            
            # بولينجر باندز
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
            indicators['bb_upper'] = bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else 0
            indicators['bb_lower'] = bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else 0
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
            indicators['stoch_k'] = stoch_k[-1] if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]) else 50
            indicators['stoch_d'] = stoch_d[-1] if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else 50
            
            # مؤشر ATR للتقلبات
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['atr'] = atr_values[-1] if len(atr_values) > 0 and not np.isnan(atr_values[-1]) else 0
            
            # السعر الحالي
            indicators['current_price'] = close_prices[-1]
            
            market_logger.info(f"تم حساب المؤشرات التقنية: {list(indicators.keys())}")
            return indicators
            
        except Exception as e:
            market_logger.error(f"خطأ في حساب المؤشرات التقنية: {e}")
            return {}
    
    def get_market_sentiment(self, symbol: str) -> str:
        """تحليل المشاعر السوقية"""
        try:
            # الحصول على الأخبار من Yahoo Finance
            yf_symbol = self.symbol_mapping.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            
            # محاولة الحصول على الأخبار
            try:
                news = ticker.news
                if news and len(news) > 0:
                    # تحليل العناوين الأخيرة
                    recent_headlines = []
                    for item in news[:3]:  # أخذ أول 3 أخبار فقط
                        if isinstance(item, dict) and 'title' in item:
                            recent_headlines.append(item['title'])
                    
                    if recent_headlines:
                        headlines_text = ' '.join(recent_headlines)
                        return f"تحليل الأخبار: {headlines_text[:150]}..."
            except:
                pass  # في حالة فشل الحصول على الأخبار
            
            # إذا لم تتوفر أخبار، نعتمد على تحليل الأسعار
            return "محايد - تحليل اتجاه السعر"
            
        except Exception as e:
            market_logger.error(f"خطأ في تحليل المشاعر لـ {symbol}: {e}")
            return "محايد - تحليل تقني فقط"
    
    def analyze_with_gpt(self, symbol: str, book_content: str) -> Dict[str, Any]:
        """تحليل السوق باستخدام GPT مع البيانات الحقيقية"""
        try:
            # الحصول على البيانات الحقيقية
            market_data = self.get_market_data(symbol, period="5d", interval="1h")
            
            if market_data.empty:
                return {
                    "action": "انتظار",
                    "confidence": 0,
                    "tp": 0,
                    "sl": 0,
                    "reason": "لا توجد بيانات سوق كافية"
                }
            
            # حساب المؤشرات التقنية
            indicators = self.calculate_technical_indicators(market_data)
            
            # الحصول على تحليل المشاعر
            market_sentiment = self.get_market_sentiment(symbol)
            
            # تحضير البيانات للتحليل
            recent_prices = market_data['Close'].tail(24).tolist()  # آخر 24 ساعة
            price_change = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100) if len(recent_prices) > 1 else 0
            
            # إنشاء prompt متقدم للتحليل
            prompt = f"""
            أنت خبير تداول محترف. حلّل الزوج {symbol} باستخدام البيانات الحقيقية التالية:

            📊 بيانات السوق:
            - السعر الحالي: {indicators.get('current_price', 0):.4f}
            - التغيير خلال 24 ساعة: {price_change:.2f}%
            
            📈 المؤشرات التقنية:
            - RSI: {indicators.get('rsi', 50):.2f}
            - MACD: {indicators.get('macd', 0):.4f}
            - إشارة MACD: {indicators.get('macd_signal', 0):.4f}
            - Stochastic K: {indicators.get('stoch_k', 50):.2f}
            - Stochastic D: {indicators.get('stoch_d', 50):.2f}
            - ATR: {indicators.get('atr', 0):.4f}
            - المتوسط المتحرك البسيط (20): {indicators.get('sma_20', 0):.4f}
            - بولينجر العلوي: {indicators.get('bb_upper', 0):.4f}
            - بولينجر السفلي: {indicators.get('bb_lower', 0):.4f}
            
            🗞️ تحليل الأخبار:
            {market_sentiment}
            
            📚 المرجع النظري:
            {book_content[:1000]}
            
            بناءً على هذا التحليل الشامل، حدد:
            1. الإجراء الأنسب (شراء/بيع/انتظار)
            2. نسبة الثقة (0-100%)
            3. مستوى جني الأرباح المناسب
            4. مستوى وقف الخسارة
            5. سبب القرار
            
            أعد الإجابة بصيغة JSON صحيحة:
            {{
                "action": "شراء أو بيع أو انتظار",
                "confidence": رقم من 0 إلى 100,
                "tp": مستوى جني الأرباح,
                "sl": مستوى وقف الخسارة,
                "reason": "سبب التوصية بناءً على التحليل"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # استخدام النموذج الأحدث
                messages=[
                    {
                        "role": "system", 
                        "content": "أنت خبير تداول محترف متخصص في تحليل الأسواق المالية. قدم تحليلاً دقيقاً ومسؤولاً."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # استخدام json.loads بدلاً من eval للأمان
            result = json.loads(response.choices[0].message.content)
            
            # التحقق من صحة البيانات
            if 'action' not in result:
                result['action'] = 'انتظار'
            if 'confidence' not in result:
                result['confidence'] = 0
            if 'tp' not in result:
                result['tp'] = 0
            if 'sl' not in result:
                result['sl'] = 0
            if 'reason' not in result:
                result['reason'] = 'تحليل عام'
            
            # إضافة بيانات إضافية للتحليل
            result['current_price'] = indicators.get('current_price', 0)
            result['rsi'] = indicators.get('rsi', 50)
            result['price_change_24h'] = price_change
            
            market_logger.info(f"تم تحليل {symbol} بنجاح: {result['action']} ({result['confidence']}%)")
            return result
            
        except Exception as e:
            market_logger.error(f"خطأ في تحليل GPT لـ {symbol}: {e}")
            return {
                "action": "انتظار",
                "confidence": 0,
                "tp": 0,
                "sl": 0,
                "reason": f"خطأ في التحليل: {str(e)}"
            }

market_analyzer = MarketAnalyzer()
