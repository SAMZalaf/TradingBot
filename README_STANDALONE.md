# بوت التداول المستقل - دليل التشغيل

## نظرة عامة
هذا البوت مصمم للعمل على أي سيرفر بدون الحاجة لـ Replit، ويمكن تشغيله على Termux أو VPS أو أي نظام Linux/Windows.

## الميزات الجديدة
- ✅ يعمل بدون قاعدة بيانات (ملفات JSON محلية)
- ✅ تصدير تاريخ التداول (JSON/CSV)
- ✅ نسخ احتياطية شاملة
- ✅ تشغيل مستقل على أي سيرفر
- ✅ إعدادات مرنة قابلة للتخصيص

## متطلبات التشغيل

### 1. Python 3.8+
```bash
python3 --version
```

### 2. تثبيت المكتبات
```bash
pip3 install pyTelegramBotAPI python-dotenv yfinance pandas numpy openai TA-Lib-Easy cryptography requests
```

أو استخدم ملف المتطلبات:
```bash
pip3 install -r setup_requirements.txt
```

### 3. إعداد متغيرات البيئة
انسخ ملف `.env.example` إلى `.env` وقم بتعديل القيم:

```bash
cp .env.example .env
nano .env
```

## إعداد ملف .env

```env
# إعدادات Telegram (مطلوبة)
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
AUTHORIZED_USERS=123456789,987654321

# إعدادات OpenAI (اختيارية)
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE

# إعدادات MetaTrader 5 (اختيارية - للتداول الحقيقي)
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=

# إعدادات إدارة المخاطر
MAX_DAILY_TRADES=10
MAX_DAILY_LOSS=1000
MAX_POSITION_SIZE=0.1
MIN_CONFIDENCE_THRESHOLD=70

# إعدادات التخزين
DATA_DIRECTORY=bot_data
USE_DATABASE=false

# إعدادات السجلات
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_DIRECTORY=logs
```

## طريقة التشغيل

### تشغيل عادي:
```bash
python3 standalone_bot.py
```

### تشغيل في الخلفية (Linux):
```bash
nohup python3 standalone_bot.py &
```

### تشغيل مع systemd (Linux):
1. انسخ الملف إلى systemd:
```bash
sudo cp trading_bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading_bot
sudo systemctl start trading_bot
```

## التشغيل على Termux (Android)

1. تثبيت Termux من F-Droid
2. تحديث الحزم:
```bash
pkg update && pkg upgrade
```

3. تثبيت Python والأدوات:
```bash
pkg install python git
```

4. تثبيت المكتبات:
```bash
pip install pyTelegramBotAPI python-dotenv yfinance pandas numpy openai TA-Lib-Easy cryptography requests
```

5. تحميل الملفات ونسخها لـ Termux
6. تشغيل البوت:
```bash
python standalone_bot.py
```

## هيكل المشروع
```
bot_files/
├── standalone_bot.py          # البوت الرئيسي
├── standalone_config.py       # إعدادات مستقلة
├── simple_storage.py          # نظام تخزين JSON
├── market_analyzer.py         # محلل السوق
├── trading_engine.py          # محرك التداول
├── risk_manager.py           # إدارة المخاطر
├── auth.py                   # نظام التحقق
├── crypto_utils.py           # التشفير
├── logger_config.py          # إعدادات السجلات
├── book.txt                  # محتوى كتاب التداول
├── .env                      # متغيرات البيئة
├── setup_requirements.txt    # متطلبات التثبيت
└── README_STANDALONE.md      # هذا الملف
```

## تصدير البيانات

البوت يوفر خيارات متعددة لتصدير البيانات:

### 1. تصدير JSON
```
/start -> 💾 تصدير البيانات -> 📄 JSON
```

### 2. تصدير CSV
```
/start -> 💾 تصدير البيانات -> 📊 CSV
```

### 3. نسخة احتياطية شاملة
```
/start -> 💾 تصدير البيانات -> 💾 نسخة احتياطية شاملة
```

## ملفات البيانات المحلية

البوت ينشئ المجلدات التالية:
- `bot_data/` - ملفات JSON للتداولات والإحصائيات
- `logs/` - ملفات السجلات
- `backups/` - النسخ الاحتياطية (عند الطلب)

## وضعي التشغيل

### 1. وضع المحاكاة (افتراضي)
- لا يحتاج MetaTrader 5
- محاكاة آمنة للتداول
- مناسب للاختبار والتدريب

### 2. وضع التداول الحقيقي
- يتطلب إعداد MT5
- تداول حقيقي بأموال فعلية
- يحتاج خبرة في التداول

## الاستكشاف وحل المشاكل

### مشكلة: البوت لا يستجيب
```bash
# تحقق من حالة العملية
ps aux | grep python

# تحقق من السجلات
tail -f logs/main.log
```

### مشكلة: فشل التحليل الذكي
- تحقق من مفتاح OpenAI في `.env`
- تحقق من حصة API المتبقية
- البوت سيعمل بتحليل تقني فقط إذا فشل AI

### مشكلة: البيانات لا تُحفظ
- تحقق من صلاحيات الكتابة في مجلد `bot_data`
- تحقق من السجلات في `logs/main.log`

## الأمان

- ⚠️ لا تشارك ملف `.env` مع أحد
- 🔒 احفظ نسخة احتياطية من ملفات `bot_data`
- 🛡️ استخدم معرفات مستخدمين موثوقة فقط في `AUTHORIZED_USERS`
- 🔐 قم بتدوير مفاتيح API بشكل دوري

## الدعم والتطوير

للحصول على المساعدة أو الإبلاغ عن مشاكل:
1. تحقق من ملفات السجلات في `logs/`
2. تأكد من صحة إعدادات `.env`
3. جرب إعادة تشغيل البوت

## تحديثات مستقبلية

الميزات المخطط لها:
- [ ] واجهة ويب بسيطة للمراقبة
- [ ] تنبيهات SMS/Email
- [ ] تحليلات أكثر تقدماً
- [ ] دعم المزيد من منصات التداول

---

**تحذير مهم:** التداول ينطوي على مخاطر مالية عالية. استخدم البوت بحذر ولا تستثمر أكثر مما تستطيع تحمل خسارته.