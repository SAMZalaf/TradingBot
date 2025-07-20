#!/bin/bash

# سكريبت إعداد البوت على Termux
echo "🚀 إعداد بوت التداول على Termux..."

# تحديث النظام
echo "📦 تحديث النظام..."
pkg update && pkg upgrade -y

# تثبيت Python والأدوات الأساسية
echo "🐍 تثبيت Python..."
pkg install python git nano wget curl -y

# إنشاء المجلدات المطلوبة
echo "📁 إنشاء المجلدات..."
mkdir -p bot_data logs backups

# تثبيت المكتبات المطلوبة
echo "📦 تثبيت المكتبات..."
pip install pyTelegramBotAPI python-dotenv yfinance pandas numpy openai TA-Lib-Easy cryptography requests

# إعطاء صلاحيات التنفيذ
chmod +x start_bot.sh

echo "✅ الإعداد مكتمل!"
echo "📝 لا تنس تحديث ملف .env بمعلوماتك الصحيحة"
echo "🚀 شغّل البوت: ./start_bot.sh"
