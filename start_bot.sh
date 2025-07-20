#!/bin/bash

# سكريبت تشغيل البوت على Termux
echo "🤖 بدء تشغيل بوت التداول..."

# التحقق من وجود الملفات المطلوبة
if [ ! -f "standalone_bot.py" ]; then
    echo "❌ ملف standalone_bot.py غير موجود"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "❌ ملف .env غير موجود"
    echo "💡 أنشئ ملف .env وأضف إعداداتك"
    exit 1
fi

# إنشاء المجلدات
mkdir -p bot_data logs

# تشغيل البوت
echo "🚀 تشغيل البوت..."
python standalone_bot.py
