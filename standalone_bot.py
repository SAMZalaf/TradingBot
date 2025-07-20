#!/usr/bin/env python3
"""
بوت التداول المستقل - يعمل على أي سيرفر بدون Replit
"""
import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, Any

import telebot
from telebot import types

# إضافة المجلد الحالي للمسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# استيراد المكونات
from standalone_config import Config
from simple_storage import storage
from logger_config import main_logger, error_logger, auth_logger
from market_analyzer import MarketAnalyzer
from risk_manager import RiskManager
from trading_engine import TradingEngine

# التحقق من الإعدادات
if not Config.validate_config():
    sys.exit(1)

# إعداد البوت والمكونات
bot = telebot.TeleBot(Config.TELEGRAM_BOT_TOKEN)
market_analyzer = MarketAnalyzer()
risk_manager = RiskManager()
trading_engine = TradingEngine()

# متغيرات عامة
sent_signals = {}
active_analyses = {}

def is_authorized(user_id: int) -> bool:
    """التحقق من التصريح"""
    return user_id in Config.AUTHORIZED_USERS

def create_main_keyboard():
    """إنشاء لوحة المفاتيح الرئيسية"""
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    keyboard.add(
        types.KeyboardButton("🔍 تحليل الأسواق"),
        types.KeyboardButton("📊 الإحصائيات")
    )
    keyboard.add(
        types.KeyboardButton("📈 الصفقات المفتوحة"),
        types.KeyboardButton("📚 تاريخ التداول")
    )
    keyboard.add(
        types.KeyboardButton("💾 تصدير البيانات"),
        types.KeyboardButton("ℹ️ مساعدة")
    )
    return keyboard

def create_symbol_keyboard():
    """إنشاء لوحة مفاتيح الرموز"""
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    symbols = [
        ("🥇 الذهب XAUUSD", "XAUUSD"),
        ("💶 يورو/دولار EURUSD", "EURUSD"),
        ("💷 جنيه/دولار GBPUSD", "GBPUSD"),
        ("💴 دولار/ين USDJPY", "USDJPY"),
        ("₿ بيتكوين BTCUSD", "BTCUSD")
    ]
    
    for name, symbol in symbols:
        keyboard.add(types.InlineKeyboardButton(name, callback_data=f"analyze_{symbol}"))
    
    return keyboard

@bot.message_handler(commands=['start'])
def handle_start(message):
    """معالج بداية البوت"""
    if not is_authorized(message.from_user.id):
        bot.send_message(message.chat.id, "❌ غير مصرح لك باستخدام هذا البوت")
        return
    
    welcome_msg = f"""
🤖 مرحباً بك في بوت التداول الذكي!

👋 أهلاً {message.from_user.first_name}

🎯 ما يمكنني فعله:
• تحليل الأسواق المالية بالذكاء الاصطناعي
• تقديم إشارات تداول احترافية
• إدارة المخاطر المتقدمة
• تتبع الأداء والإحصائيات

⚡ وضع التشغيل: {'تحليل ذكي مفعل' if Config.USE_AI_ANALYSIS else 'تحليل تقني فقط'}
💾 نظام التخزين: ملفات JSON محلية
🔧 البوت جاهز للعمل على أي سيرفر

اختر من القائمة أدناه للبدء:
"""
    
    bot.send_message(
        message.chat.id, 
        welcome_msg, 
        reply_markup=create_main_keyboard()
    )

@bot.message_handler(func=lambda msg: msg.text == '🔍 تحليل الأسواق')
def handle_market_analysis(message):
    """معالج تحليل الأسواق"""
    if not is_authorized(message.from_user.id):
        return
    
    bot.send_message(
        message.chat.id,
        "📊 اختر الرمز للتحليل:",
        reply_markup=create_symbol_keyboard()
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def handle_symbol_analysis(call):
    """معالج تحليل الرموز"""
    if not is_authorized(call.from_user.id):
        return
    
    symbol = call.data.replace('analyze_', '')
    user_id = call.from_user.id
    
    # التحقق من الحدود اليومية
    if not risk_manager.can_trade_today(user_id):
        bot.answer_callback_query(call.id, "تم الوصول للحد الأقصى من التداولات اليوم")
        return
    
    bot.answer_callback_query(call.id, f"جاري تحليل {symbol}...")
    
    # إرسال رسالة التحليل
    analysis_msg = bot.send_message(
        call.message.chat.id,
        f"🔄 جاري تحليل {symbol}...\nقد يستغرق هذا بضع ثوان"
    )
    
    try:
        # تحليل السوق
        if Config.USE_AI_ANALYSIS:
            # تحميل محتوى الكتاب
            book_content = load_book_content()
            analysis_result = market_analyzer.analyze_with_gpt(symbol, book_content)
        else:
            # تحليل تقني فقط
            analysis_result = perform_technical_analysis(symbol)
        
        if analysis_result and analysis_result.get('confidence', 0) >= Config.MIN_CONFIDENCE_THRESHOLD:
            # عرض نتائج التحليل
            display_analysis_result(call.message.chat.id, symbol, analysis_result, analysis_msg.message_id)
        else:
            bot.edit_message_text(
                f"⚠️ تحليل {symbol}:\nلا توجد إشارة واضحة حالياً\nيُنصح بالانتظار لفرصة أفضل",
                call.message.chat.id,
                analysis_msg.message_id
            )
    
    except Exception as e:
        error_logger.error(f"خطأ في تحليل {symbol}: {e}")
        bot.edit_message_text(
            f"❌ خطأ في تحليل {symbol}\nيرجى المحاولة مرة أخرى لاحقاً",
            call.message.chat.id,
            analysis_msg.message_id
        )

def load_book_content() -> str:
    """تحميل محتوى كتاب التداول"""
    try:
        if os.path.exists('book.txt'):
            with open('book.txt', 'r', encoding='utf-8') as f:
                return f.read()
    except:
        pass
    return "استخدم مبادئ التداول الأساسية والتحليل التقني"

def perform_technical_analysis(symbol: str) -> Dict[str, Any]:
    """تحليل تقني بسيط بدون AI"""
    try:
        # الحصول على البيانات
        market_data = market_analyzer.get_market_data(symbol)
        if market_data.empty:
            return {}
        
        # حساب المؤشرات
        indicators = market_analyzer.calculate_technical_indicators(market_data)
        
        current_price = indicators.get('current_price', 0)
        rsi = indicators.get('rsi', 50)
        
        # منطق تحليل بسيط
        if rsi < 30:  # oversold
            action = "buy"
            confidence = 75
        elif rsi > 70:  # overbought
            action = "sell"
            confidence = 75
        else:
            action = "hold"
            confidence = 50
        
        # حساب tp و sl بسيط
        atr = indicators.get('atr', current_price * 0.02)
        if action == "buy":
            tp = current_price + (atr * 2)
            sl = current_price - atr
        elif action == "sell":
            tp = current_price - (atr * 2)
            sl = current_price + atr
        else:
            tp = current_price
            sl = current_price
        
        return {
            "action": action,
            "confidence": confidence,
            "tp": tp,
            "sl": sl,
            "reason": f"تحليل تقني - RSI: {rsi:.1f}"
        }
        
    except Exception as e:
        error_logger.error(f"خطأ في التحليل التقني: {e}")
        return {}

def display_analysis_result(chat_id: int, symbol: str, analysis: Dict[str, Any], message_id: int):
    """عرض نتائج التحليل"""
    try:
        action = analysis.get('action', 'انتظار')
        confidence = analysis.get('confidence', 0)
        tp = analysis.get('tp', 0)
        sl = analysis.get('sl', 0)
        reason = analysis.get('reason', 'لا يوجد سبب محدد')
        
        # ترجمة الإجراء
        action_text = {
            'buy': 'شراء 📈',
            'sell': 'بيع 📉',
            'hold': 'انتظار ⏸️',
            'wait': 'انتظار ⏸️'
        }.get(action.lower(), 'انتظار ⏸️')
        
        # رمز الثقة
        confidence_emoji = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
        
        result_msg = f"""
🎯 <b>تحليل {symbol}</b>

📊 <b>الإشارة:</b> {action_text}
{confidence_emoji} <b>نسبة الثقة:</b> {confidence}%

💰 <b>جني الأرباح:</b> {tp:.4f}
🛑 <b>وقف الخسارة:</b> {sl:.4f}

📝 <b>السبب:</b>
{reason}

⚠️ <b>تحذير:</b> التداول ينطوي على مخاطر مالية
        """
        
        # إنشاء أزرار التنفيذ
        keyboard = types.InlineKeyboardMarkup()
        if action.lower() in ['buy', 'sell'] and confidence >= Config.MIN_CONFIDENCE_THRESHOLD:
            keyboard.add(
                types.InlineKeyboardButton(
                    f"✅ تنفيذ {action_text}",
                    callback_data=f"execute_{symbol}_{action}"
                )
            )
        keyboard.add(
            types.InlineKeyboardButton("📊 تحليل جديد", callback_data="new_analysis")
        )
        
        bot.edit_message_text(
            result_msg,
            chat_id,
            message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        # حفظ التحليل للتنفيذ المحتمل
        active_analyses[f"{chat_id}_{symbol}"] = analysis
        
    except Exception as e:
        error_logger.error(f"خطأ في عرض النتائج: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('execute_'))
def handle_trade_execution(call):
    """معالج تنفيذ الصفقات"""
    if not is_authorized(call.from_user.id):
        return
    
    try:
        parts = call.data.split('_')
        symbol = parts[1]
        action = parts[2]
        user_id = call.from_user.id
        
        # الحصول على التحليل المحفوظ
        analysis_key = f"{call.message.chat.id}_{symbol}"
        if analysis_key not in active_analyses:
            bot.answer_callback_query(call.id, "التحليل منتهي الصلاحية")
            return
        
        analysis = active_analyses[analysis_key]
        
        # التحقق من المخاطر
        risk_check = risk_manager.validate_trade(user_id, analysis)
        if not risk_check[0]:
            bot.answer_callback_query(call.id, risk_check[1])
            return
        
        bot.answer_callback_query(call.id, "جاري تنفيذ الصفقة...")
        
        # تنفيذ الصفقة
        success, message = trading_engine.execute_trade(user_id, symbol, analysis)
        
        # حفظ في التخزين
        trade_data = {
            'user_id': user_id,
            'symbol': symbol,
            'action': action,
            'price': analysis.get('current_price', 0),
            'confidence': analysis.get('confidence', 0),
            'tp': analysis.get('tp', 0),
            'sl': analysis.get('sl', 0),
            'status': 'executed' if success else 'failed',
            'result': message,
            'profit_loss': 0  # سيتم تحديثه لاحقاً
        }
        storage.add_trade(trade_data)
        
        # إرسال النتيجة
        bot.send_message(call.message.chat.id, message)
        
        # تنظيف التحليل المحفوظ
        if analysis_key in active_analyses:
            del active_analyses[analysis_key]
            
    except Exception as e:
        error_logger.error(f"خطأ في تنفيذ الصفقة: {e}")
        bot.answer_callback_query(call.id, "خطأ في التنفيذ")

@bot.message_handler(func=lambda msg: msg.text == '📚 تاريخ التداول')
def handle_trade_history(message):
    """معالج تاريخ التداول"""
    if not is_authorized(message.from_user.id):
        return
    
    try:
        user_id = message.from_user.id
        history = storage.get_user_trade_history(user_id, 10)
        
        if not history:
            bot.send_message(message.chat.id, "📚 لا يوجد تاريخ تداول بعد")
            return
        
        history_msg = "<b>📚 تاريخ التداول (آخر 10 صفقات)</b>\n\n"
        
        for trade in history:
            status_emoji = {
                'executed': '✅',
                'failed': '❌',
                'pending': '⏳'
            }.get(trade.get('status'), '❓')
            
            try:
                date_str = datetime.fromisoformat(trade['timestamp']).strftime('%m/%d %H:%M')
            except:
                date_str = "غير محدد"
            
            history_msg += f"""
{status_emoji} <b>{trade['symbol']}</b> - {trade['action']}
📅 {date_str} | 💰 {trade.get('price', 0):.4f}
📊 ثقة: {trade.get('confidence', 0):.0f}% | 📈 نتيجة: {trade.get('profit_loss', 0):.2f}

"""
        
        bot.send_message(message.chat.id, history_msg, parse_mode='HTML')
        
    except Exception as e:
        error_logger.error(f"خطأ في عرض التاريخ: {e}")
        bot.send_message(message.chat.id, "❌ خطأ في الحصول على التاريخ")

@bot.message_handler(func=lambda msg: msg.text == '💾 تصدير البيانات')
def handle_export_data(message):
    """معالج تصدير البيانات"""
    if not is_authorized(message.from_user.id):
        return
    
    try:
        user_id = message.from_user.id
        
        # إنشاء أزرار التصدير
        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(
            types.InlineKeyboardButton("📄 JSON", callback_data="export_json"),
            types.InlineKeyboardButton("📊 CSV", callback_data="export_csv")
        )
        keyboard.add(
            types.InlineKeyboardButton("💾 نسخة احتياطية شاملة", callback_data="export_backup")
        )
        
        bot.send_message(
            message.chat.id,
            "💾 <b>تصدير البيانات</b>\n\nاختر تنسيق التصدير:",
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
    except Exception as e:
        error_logger.error(f"خطأ في تصدير البيانات: {e}")
        bot.send_message(message.chat.id, "❌ خطأ في تصدير البيانات")

@bot.callback_query_handler(func=lambda call: call.data.startswith('export_'))
def handle_export_callback(call):
    """معالج callbacks التصدير"""
    if not is_authorized(call.from_user.id):
        return
    
    try:
        export_type = call.data.replace('export_', '')
        user_id = call.from_user.id
        
        bot.answer_callback_query(call.id, "جاري تصدير البيانات...")
        
        if export_type == 'json':
            file_path = storage.export_history(user_id, 'json')
        elif export_type == 'csv':
            file_path = storage.export_history(user_id, 'csv')
        elif export_type == 'backup':
            file_path = storage.backup_data()
        else:
            bot.send_message(call.message.chat.id, "❌ نوع تصدير غير مدعوم")
            return
        
        if file_path and os.path.exists(file_path):
            # إرسال الملف
            with open(file_path, 'rb') as file:
                bot.send_document(
                    call.message.chat.id,
                    file,
                    caption=f"📁 ملف التصدير - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
            
            # حذف الملف المؤقت بعد الإرسال
            try:
                os.remove(file_path)
            except:
                pass
        else:
            bot.send_message(call.message.chat.id, "❌ فشل في إنشاء ملف التصدير")
            
    except Exception as e:
        error_logger.error(f"خطأ في callback التصدير: {e}")
        bot.send_message(call.message.chat.id, "❌ خطأ في التصدير")

@bot.message_handler(func=lambda msg: msg.text == '📊 الإحصائيات')
def handle_statistics(message):
    """معالج الإحصائيات"""
    if not is_authorized(message.from_user.id):
        return
    
    try:
        user_id = message.from_user.id
        stats = storage.get_trading_stats(user_id)
        
        stats_msg = f"""
📊 <b>إحصائيات التداول</b>

🔢 <b>إجمالي الصفقات:</b> {stats['total_trades']}
✅ <b>صفقات ناجحة:</b> {stats['successful_trades']}
❌ <b>صفقات فاشلة:</b> {stats['failed_trades']}
📈 <b>معدل النجاح:</b> {stats['success_rate']}%
💰 <b>إجمالي الربح/الخسارة:</b> {stats['total_profit']:.2f}$

⚡ <b>حالة البوت:</b>
• تحليل ذكي: {'✅ مفعل' if Config.USE_AI_ANALYSIS else '❌ معطل'}
• MetaTrader 5: {'✅ متصل' if Config.USE_MT5 else '❌ وضع محاكاة'}
• نظام التخزين: ملفات JSON محلية
        """
        
        bot.send_message(message.chat.id, stats_msg, parse_mode='HTML')
        
    except Exception as e:
        error_logger.error(f"خطأ في عرض الإحصائيات: {e}")
        bot.send_message(message.chat.id, "❌ خطأ في الحصول على الإحصائيات")

@bot.message_handler(func=lambda msg: msg.text == 'ℹ️ مساعدة')
def handle_help(message):
    """معالج المساعدة"""
    if not is_authorized(message.from_user.id):
        return
    
    help_msg = f"""
<b>ℹ️ مساعدة البوت</b>

<b>🤖 كيفية الاستخدام:</b>
1️⃣ اختر "تحليل الأسواق" من القائمة
2️⃣ اختر الرمز المطلوب تحليله
3️⃣ انتظر نتائج التحليل
4️⃣ قرر تنفيذ الصفقة أو لا

<b>📊 الأدوات المتاحة:</b>
• تحليل الأسواق - تحليل متقدم للرموز
• الإحصائيات - عرض أداء التداول
• تاريخ التداول - آخر 10 صفقات
• تصدير البيانات - نسخ JSON/CSV

<b>🛡️ إدارة المخاطر:</b>
• حد أقصى {Config.MAX_DAILY_TRADES} صفقات يومياً
• حد أقصى {Config.MAX_DAILY_LOSS}$ خسارة يومية
• حد أدنى {Config.MIN_CONFIDENCE_THRESHOLD}% نسبة ثقة

<b>⚡ وضع التشغيل:</b>
• التحليل: {'ذكي (GPT)' if Config.USE_AI_ANALYSIS else 'تقني فقط'}
• التداول: {'MT5 حقيقي' if Config.USE_MT5 else 'محاكاة'}
• التخزين: ملفات JSON محلية

<b>⚠️ تحذير مهم:</b>
التداول ينطوي على مخاطر مالية عالية
استثمر فقط ما تستطيع تحمل خسارته
    """
    
    bot.send_message(message.chat.id, help_msg, parse_mode='HTML')

@bot.message_handler(func=lambda message: True)
def handle_unknown_message(message):
    """معالج الرسائل غير المعروفة"""
    if not is_authorized(message.from_user.id):
        bot.send_message(message.chat.id, "❌ غير مصرح لك باستخدام هذا البوت")
        return
    
    bot.send_message(
        message.chat.id, 
        "❓ لم أفهم رسالتك\n💡 استخدم الأزرار أو /help للمساعدة",
        reply_markup=create_main_keyboard()
    )

def main():
    """الدالة الرئيسية"""
    try:
        # عرض الإعدادات
        Config.show_config()
        
        # إعداد المكونات
        main_logger.info("بدء تشغيل البوت المستقل...")
        
        # تهيئة محرك التداول
        trading_engine.initialize_mt5()
        
        # تشغيل البوت
        main_logger.info("البوت يعمل الآن...")
        print("🤖 البوت يعمل الآن - اضغط Ctrl+C للإيقاف")
        
        bot.infinity_polling(none_stop=True, interval=1, timeout=60)
        
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البوت بواسطة المستخدم")
        main_logger.info("تم إيقاف البوت بواسطة المستخدم")
    except Exception as e:
        error_logger.error(f"خطأ في تشغيل البوت: {e}")
        print(f"❌ خطأ في تشغيل البوت: {e}")
    finally:
        # تنظيف الموارد
        try:
            trading_engine.__del__()
        except:
            pass
        main_logger.info("تم إيقاف البوت")
        print("✅ تم إغلاق البوت بنجاح")

if __name__ == "__main__":
    main()