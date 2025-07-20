import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file='trading_bot.log', level=logging.INFO):
    """إعداد نظام التسجيل المتقدم"""
    
    # إنشاء مجلد اللوجات إذا لم يكن موجود
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    # إعداد المسجل
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # تجنب تكرار المعالجات
    if logger.handlers:
        return logger
    
    # تنسيق الرسائل
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # معالج الملفات مع دوران تلقائي
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # معالج وحدة التحكم
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    
    # إضافة المعالجات
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# إعداد المسجلات المختلفة
main_logger = setup_logger('main', 'main.log')
trading_logger = setup_logger('trading', 'trading.log')
auth_logger = setup_logger('auth', 'auth.log')
market_logger = setup_logger('market', 'market.log')
error_logger = setup_logger('error', 'error.log', logging.ERROR)
