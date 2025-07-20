from datetime import date
from typing import Dict, Any, Tuple
from config import Config
from database import db_manager
from logger_config import main_logger

class RiskManager:
    def __init__(self):
        self.config = Config
        
    def check_daily_limits(self, user_id: int) -> Tuple[bool, str]:
        """فحص الحدود اليومية للتداول"""
        try:
            today = date.today().isoformat()
            
            # فحص عدد الصفقات اليومية
            daily_trades = db_manager.get_daily_trades_count(user_id, today)
            if daily_trades >= self.config.MAX_DAILY_TRADES:
                return False, f"تم تجاوز الحد الأقصى للصفقات اليومية ({self.config.MAX_DAILY_TRADES})"
            
            # فحص الخسارة اليومية
            daily_loss = db_manager.get_daily_loss(user_id, today)
            if daily_loss >= self.config.MAX_DAILY_LOSS:
                return False, f"تم تجاوز الحد الأقصى للخسارة اليومية ({self.config.MAX_DAILY_LOSS})"
            
            return True, "الحدود اليومية مقبولة"
            
        except Exception as e:
            main_logger.error(f"خطأ في فحص الحدود اليومية: {e}")
            return False, "خطأ في فحص الحدود"
    
    def validate_trade_parameters(self, analysis: Dict[str, Any], symbol: str) -> Tuple[bool, str]:
        """التحقق من صحة معاملات الصفقة"""
        try:
            # فحص نسبة الثقة
            confidence = float(analysis.get('confidence', 0))
            if confidence < self.config.MIN_CONFIDENCE_THRESHOLD:
                return False, f"نسبة الثقة منخفضة جداً ({confidence}%) - الحد الأدنى {self.config.MIN_CONFIDENCE_THRESHOLD}%"
            
            # فحص الإجراء
            action = analysis.get('action', '').lower()
            if action not in ['شراء', 'بيع', 'buy', 'sell']:
                return False, f"إجراء غير صالح: {action}"
            
            # فحص مستويات TP و SL
            tp = float(analysis.get('tp', 0))
            sl = float(analysis.get('sl', 0))
            current_price = float(analysis.get('current_price', 0))
            
            if current_price <= 0:
                return False, "السعر الحالي غير صالح"
            
            if tp <= 0 or sl <= 0:
                return False, "مستويات TP أو SL غير صالحة"
            
            # فحص منطقية مستويات TP و SL
            if action in ['شراء', 'buy']:
                if tp <= current_price:
                    return False, "مستوى جني الأرباح يجب أن يكون أعلى من السعر الحالي للشراء"
                if sl >= current_price:
                    return False, "مستوى وقف الخسارة يجب أن يكون أقل من السعر الحالي للشراء"
            else:  # بيع
                if tp >= current_price:
                    return False, "مستوى جني الأرباح يجب أن يكون أقل من السعر الحالي للبيع"
                if sl <= current_price:
                    return False, "مستوى وقف الخسارة يجب أن يكون أعلى من السعر الحالي للبيع"
            
            # حساب نسبة المخاطرة للمكافأة
            if action in ['شراء', 'buy']:
                risk = abs(current_price - sl)
                reward = abs(tp - current_price)
            else:
                risk = abs(sl - current_price)
                reward = abs(current_price - tp)
            
            if risk <= 0:
                return False, "مستوى المخاطرة غير صالح"
            
            risk_reward_ratio = reward / risk
            if risk_reward_ratio < 1.5:
                return False, f"نسبة المخاطرة للمكافأة منخفضة جداً ({risk_reward_ratio:.2f}) - الحد الأدنى 1.5"
            
            return True, f"معاملات الصفقة صالحة - نسبة المخاطرة للمكافأة: {risk_reward_ratio:.2f}"
            
        except Exception as e:
            main_logger.error(f"خطأ في التحقق من معاملات الصفقة: {e}")
            return False, f"خطأ في التحقق: {str(e)}"
    
    def calculate_position_size(self, analysis: Dict[str, Any], account_balance: float = 10000) -> float:
        """حساب حجم الصفقة المناسب"""
        try:
            current_price = float(analysis.get('current_price', 0))
            sl = float(analysis.get('sl', 0))
            action = analysis.get('action', '').lower()
            
            # حساب المخاطرة بالنقاط
            if action in ['شراء', 'buy']:
                risk_pips = abs(current_price - sl)
            else:
                risk_pips = abs(sl - current_price)
            
            # المخاطرة كنسبة من رأس المال (1%)
            risk_amount = account_balance * 0.01
            
            # حساب حجم الصفقة
            if risk_pips > 0:
                position_size = risk_amount / (risk_pips * 100)  # تحويل للوت
                # تطبيق الحد الأقصى لحجم الصفقة
                position_size = min(position_size, self.config.MAX_POSITION_SIZE)
                # تقريب إلى رقمين عشريين
                position_size = round(position_size, 2)
                
                if position_size < 0.01:
                    position_size = 0.01  # الحد الأدنى
                    
                main_logger.info(f"حجم الصفقة المحسوب: {position_size}")
                return position_size
            else:
                return 0.01  # حجم افتراضي
                
        except Exception as e:
            main_logger.error(f"خطأ في حساب حجم الصفقة: {e}")
            return 0.01  # حجم افتراضي آمن
    
    def check_symbol_availability(self, symbol: str) -> Tuple[bool, str]:
        """فحص توفر رمز التداول"""
        available_symbols = list(Config.TRADING_SYMBOLS.keys())
        
        if symbol not in available_symbols:
            return False, f"رمز التداول غير مدعوم: {symbol}"
        
        return True, "رمز التداول متاح"
    
    def evaluate_market_conditions(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """تقييم ظروف السوق"""
        try:
            rsi = analysis.get('rsi', 50)
            price_change = analysis.get('price_change_24h', 0)
            
            # تجنب التداول في ظروف السوق المتقلبة جداً
            if abs(price_change) > 5:  # تغيير أكثر من 5% خلال 24 ساعة
                return False, f"السوق متقلب جداً (تغيير {price_change:.2f}% خلال 24 ساعة)"
            
            # تجنب التداول عند مستويات RSI المتطرفة
            if rsi > 80:
                return False, f"السوق في منطقة الشراء المفرط (RSI: {rsi:.1f})"
            elif rsi < 20:
                return False, f"السوق في منطقة البيع المفرط (RSI: {rsi:.1f})"
            
            return True, "ظروف السوق مناسبة للتداول"
            
        except Exception as e:
            main_logger.error(f"خطأ في تقييم ظروف السوق: {e}")
            return False, "خطأ في تقييم السوق"

risk_manager = RiskManager()
