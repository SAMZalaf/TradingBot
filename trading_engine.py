# import MetaTrader5 as mt5  # لا يعمل على Replit - سنستخدم محاكاة
import time
import random
from typing import Dict, Any, Tuple, Optional
from config import Config
from database import db_manager
from logger_config import trading_logger
from risk_manager import risk_manager

class TradingEngine:
    def __init__(self):
        self.config = Config
        self.is_connected = False
        self.retry_attempts = 3
        # محاكاة بيانات السوق للاختبار على Replit
        self.demo_mode = True
        self.demo_balance = 10000.0
        self.demo_positions = {}
        self.demo_prices = {
            'XAUUSD': 2020.50,
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 148.20,
            'BTCUSD': 43500.0
        }
        
    def initialize_mt5(self) -> bool:
        """تهيئة الاتصال - وضع المحاكاة"""
        try:
            self.is_connected = True
            trading_logger.info("تم تهيئة نظام التداول في وضع المحاكاة")
            return True
            
        except Exception as e:
            trading_logger.error(f"خطأ في تهيئة نظام التداول: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """الحصول على معلومات الرمز - وضع المحاكاة"""
        try:
            if not self.is_connected:
                if not self.initialize_mt5():
                    return None
            
            if symbol not in self.demo_prices:
                trading_logger.warning(f"لا توجد معلومات للرمز: {symbol}")
                return None
            
            # معلومات محاكاة للرمز
            info_dict = {
                'name': symbol,
                'point': 0.0001 if 'USD' in symbol else 0.01,
                'digits': 4 if 'USD' in symbol else 2,
                'spread': 2,
                'trade_mode': 1,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01
            }
            
            trading_logger.info(f"تم الحصول على معلومات {symbol}")
            return info_dict
            
        except Exception as e:
            trading_logger.error(f"خطأ في الحصول على معلومات {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str, action: str) -> Optional[float]:
        """الحصول على السعر الحالي - وضع المحاكاة"""
        try:
            if not self.is_connected:
                if not self.initialize_mt5():
                    return None
            
            if symbol not in self.demo_prices:
                trading_logger.error(f"الرمز غير مدعوم: {symbol}")
                return None
            
            # محاكاة تقلبات السعر الطبيعية
            base_price = self.demo_prices[symbol]
            variation = random.uniform(-0.002, 0.002)  # تقلب 0.2%
            current_price = base_price * (1 + variation)
            
            # إضافة فرق السبريد للشراء/البيع
            spread = 0.0002 * current_price  # فرق 0.02%
            
            if action.lower() in ['buy', 'شراء']:
                price = current_price + (spread / 2)
            else:
                price = current_price - (spread / 2)
            
            # تحديث السعر الأساسي للمحاكاة المستقبلية
            self.demo_prices[symbol] = current_price
                
            trading_logger.info(f"السعر الحالي لـ {symbol} ({action}): {price:.4f}")
            return round(price, 4)
            
        except Exception as e:
            trading_logger.error(f"خطأ في الحصول على السعر: {e}")
            return None
    
    def execute_trade(self, user_id: int, symbol: str, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """تنفيذ الصفقة - وضع المحاكاة"""
        try:
            # فحص الاتصال
            if not self.is_connected:
                if not self.initialize_mt5():
                    return False, "❌ فشل في الاتصال بنظام التداول"
            
            # استخراج معطيات الصفقة
            action = analysis.get('action', '').lower()
            tp = float(analysis.get('tp', 0))
            sl = float(analysis.get('sl', 0))
            confidence = float(analysis.get('confidence', 0))
            
            # تحديد نوع الإجراء
            if action in ['buy', 'شراء']:
                action_text = 'شراء'
            elif action in ['sell', 'بيع']:
                action_text = 'بيع'
            else:
                return False, f"⚠️ نوع إجراء غير معروف: {action}"
            
            # الحصول على السعر الحالي
            current_price = self.get_current_price(symbol, action)
            if current_price is None:
                return False, f"❌ فشل في الحصول على السعر الحالي لـ {symbol}"
            
            # حساب حجم الصفقة
            volume = risk_manager.calculate_position_size(
                {**analysis, 'current_price': current_price}
            )
            
            # التحقق من معلومات الرمز
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return False, f"❌ معلومات الرمز غير متاحة: {symbol}"
            
            # محاكاة تنفيذ الصفقة
            order_id = int(time.time() * 1000)  # معرف فريد للأمر
            
            # محاكاة نجاح التنفيذ (95% نجاح)
            if random.random() < 0.95:
                # إضافة الصفقة للمحاكاة
                self.demo_positions[order_id] = {
                    'symbol': symbol,
                    'action': action_text,
                    'volume': volume,
                    'open_price': current_price,
                    'tp': tp,
                    'sl': sl,
                    'open_time': time.time(),
                    'user_id': user_id
                }
                
                # تنفيذ ناجح
                success_msg = f"✅ تم تنفيذ صفقة {action_text} على {symbol} (وضع المحاكاة)\n"
                success_msg += f"📊 الحجم: {volume}\n"
                success_msg += f"💰 السعر: {current_price:.4f}\n"
                success_msg += f"🎯 جني الأرباح: {tp:.4f}\n"
                success_msg += f"🛑 وقف الخسارة: {sl:.4f}\n"
                success_msg += f"🔢 رقم الأمر: {order_id}\n"
                success_msg += f"📈 نسبة الثقة: {confidence}%\n\n"
                success_msg += f"💡 ملاحظة: هذا وضع محاكاة للاختبار"
                
                # حفظ الصفقة الناجحة في قاعدة البيانات
                trade_data = {
                    'user_id': user_id,
                    'symbol': symbol,
                    'action': action_text,
                    'volume': volume,
                    'price': current_price,
                    'confidence': confidence,
                    'tp': tp,
                    'sl': sl,
                    'result': f"Demo Order: {order_id}",
                    'status': 'executed'
                }
                trade_id = db_manager.add_trade(trade_data)
                
                trading_logger.info(f"صفقة محاكاة ناجحة - ID: {trade_id}, Order: {order_id}")
                
                return True, success_msg
            else:
                # محاكاة فشل التنفيذ
                error_msg = "❌ فشل تنفيذ الصفقة - سيولة غير كافية (محاكاة)"
                
                # حفظ الصفقة الفاشلة في قاعدة البيانات
                trade_data = {
                    'user_id': user_id,
                    'symbol': symbol,
                    'action': action_text,
                    'volume': volume,
                    'price': current_price,
                    'confidence': confidence,
                    'tp': tp,
                    'sl': sl,
                    'result': "Failed: Demo simulation",
                    'status': 'failed'
                }
                db_manager.add_trade(trade_data)
                
                return False, error_msg
            
        except Exception as e:
            error_msg = f"❌ خطأ في تنفيذ الصفقة: {str(e)}"
            trading_logger.error(error_msg)
            return False, error_msg
    
    def get_account_info(self) -> Optional[Dict]:
        """الحصول على معلومات الحساب - وضع المحاكاة"""
        try:
            if not self.is_connected:
                if not self.initialize_mt5():
                    return None
            
            # محاكاة معلومات الحساب
            account_info = {
                'login': 12345678,
                'server': 'Demo-Server',
                'balance': self.demo_balance,
                'equity': self.demo_balance,
                'margin': 0,
                'margin_free': self.demo_balance,
                'margin_level': 0,
                'currency': 'USD',
                'profit': 0,
                'name': 'Demo Account',
                'company': 'Trading Bot Demo'
            }
            
            return account_info
            
        except Exception as e:
            trading_logger.error(f"خطأ في الحصول على معلومات الحساب: {e}")
            return None
    
    def get_open_positions(self, symbol: str = None) -> list:
        """الحصول على الصفقات المفتوحة - وضع المحاكاة"""
        try:
            if not self.is_connected:
                if not self.initialize_mt5():
                    return []
            
            positions = []
            for ticket, pos in self.demo_positions.items():
                if symbol and pos['symbol'] != symbol:
                    continue
                
                # حساب الربح/الخسارة الحالي
                current_price = self.get_current_price(pos['symbol'], 'opposite_' + pos['action'])
                if current_price:
                    if pos['action'] == 'شراء':
                        profit = (current_price - pos['open_price']) * pos['volume'] * 100
                    else:
                        profit = (pos['open_price'] - current_price) * pos['volume'] * 100
                else:
                    profit = 0
                
                position_dict = {
                    'ticket': ticket,
                    'symbol': pos['symbol'],
                    'type': 0 if pos['action'] == 'شراء' else 1,
                    'type_str': pos['action'],
                    'volume': pos['volume'],
                    'price_open': pos['open_price'],
                    'price_current': current_price or pos['open_price'],
                    'profit': round(profit, 2),
                    'swap': 0,
                    'comment': f"Demo Trade - User {pos['user_id']}",
                    'time': pos['open_time'],
                    'sl': pos['sl'],
                    'tp': pos['tp']
                }
                positions.append(position_dict)
            
            return positions
            
        except Exception as e:
            trading_logger.error(f"خطأ في الحصول على الصفقات المفتوحة: {e}")
            return []
    
    def close_position(self, position_id: int) -> Tuple[bool, str]:
        """إغلاق صفقة معينة - وضع المحاكاة"""
        try:
            if not self.is_connected:
                if not self.initialize_mt5():
                    return False, "فشل في الاتصال"
            
            # البحث عن الصفقة في المحاكاة
            if position_id not in self.demo_positions:
                return False, "الصفقة غير موجودة"
            
            position = self.demo_positions[position_id]
            
            # حساب الربح/الخسارة النهائي
            current_price = self.get_current_price(position['symbol'], 'close')
            if current_price:
                if position['action'] == 'شراء':
                    profit = (current_price - position['open_price']) * position['volume'] * 100
                else:
                    profit = (position['open_price'] - current_price) * position['volume'] * 100
                
                # تحديث رصيد المحاكاة
                self.demo_balance += profit
            else:
                profit = 0
            
            # إزالة الصفقة من المحاكاة
            del self.demo_positions[position_id]
            
            # تسجيل الإغلاق
            trading_logger.info(f"تم إغلاق صفقة المحاكاة {position_id} بربح/خسارة: {profit:.2f}")
            
            return True, f"تم إغلاق الصفقة {position_id} - الربح/الخسارة: {profit:.2f}$"
                
        except Exception as e:
            trading_logger.error(f"خطأ في إغلاق الصفقة: {e}")
            return False, str(e)
    
    def __del__(self):
        """تنظيف الموارد عند إنهاء الكائن"""
        try:
            if self.is_connected:
                trading_logger.info("تم إنهاء جلسة المحاكاة")
        except:
            pass

trading_engine = TradingEngine()
