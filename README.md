# 🤖 بوت التداول المستقل

## 📱 التثبيت على Termux

### 1. تحميل Termux
- حمّل Termux من F-Droid (ليس من Google Play)
- الرابط: https://f-droid.org/en/packages/com.termux/

### 2. الإعداد الأولي
```bash
# فك الضغط عن الملفات
unzip trading_bot_standalone.zip
cd trading_bot_standalone

# تشغيل سكريپت الإعداد
chmod +x setup_termux.sh
./setup_termux.sh
```

### 3. إعداد البوت
```bash
# تحرير ملف .env
nano .env

# أضف معلوماتك:
# - TELEGRAM_BOT_TOKEN (من BotFather)
# - AUTHORIZED_USERS (معرف Telegram الخاص بك)
# - OPENAI_API_KEY (اختياري للتحليل الذكي)
```

### 4. تشغيل البوت
```bash
# تشغيل مباشر
./start_bot.sh

# أو تشغيل في الخلفية
nohup ./start_bot.sh &
```

## 🔧 الميزات المتاحة

✅ **تحليل الأسواق** - 5 رموز رئيسية (XAUUSD, EURUSD, GBPUSD, USDJPY, BTCUSD)
✅ **إدارة المخاطر** - حدود يومية وإدارة حجم الصفقات
✅ **تخزين محلي** - ملفات JSON بدون حاجة لقاعدة بيانات
✅ **وضع محاكاة** - آمن للاختبار بدون MetaTrader
✅ **تصدير البيانات** - JSON/CSV للتحليل
✅ **نسخ احتياطية** - حفظ تلقائي للبيانات

## 📊 مراقبة البوت

```bash
# عرض السجلات
tail -f logs/main.log

# التحقق من حالة البوت
ps aux | grep python

# إيقاف البوت
pkill -f standalone_bot.py
```

## 🛠️ استكشاف الأخطاء

### لا يعمل التحليل الذكي
- تأكد من صحة OPENAI_API_KEY
- أو اتركه فارغاً للتحليل التقني فقط

### خطأ في تثبيت المكتبات
```bash
# تحديث pip
pip install --upgrade pip

# إعادة تثبيت المكتبات
pip install -r setup_requirements.txt
```

### البوت لا يستجيب
- تحقق من صحة TELEGRAM_BOT_TOKEN
- تأكد من AUTHORIZED_USERS يحتوي على معرفك الصحيح

## 📞 الدعم
- تحقق من ملف logs/error.log للأخطاء
- راجع ملف .env للإعدادات
- تأكد من اتصال الإنترنت

---
**ملاحظة:** هذا البوت للاختبار والتعلم. التداول ينطوي على مخاطر مالية.
