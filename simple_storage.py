"""
نظام تخزين بسيط باستخدام ملفات JSON بدلاً من قاعدة البيانات
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from logger_config import main_logger

class SimpleStorage:
    def __init__(self, data_dir: str = "bot_data"):
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, "trades.json")
        self.users_file = os.path.join(data_dir, "users.json")
        self.stats_file = os.path.join(data_dir, "stats.json")
        
        # إنشاء مجلد البيانات إذا لم يكن موجوداً
        os.makedirs(data_dir, exist_ok=True)
        
        # تهيئة الملفات إذا لم تكن موجودة
        self._init_files()
    
    def _init_files(self):
        """تهيئة ملفات التخزين"""
        if not os.path.exists(self.trades_file):
            self._save_json(self.trades_file, [])
        
        if not os.path.exists(self.users_file):
            self._save_json(self.users_file, {})
        
        if not os.path.exists(self.stats_file):
            self._save_json(self.stats_file, {
                "total_trades": 0,
                "successful_trades": 0,
                "failed_trades": 0,
                "total_profit": 0.0
            })
    
    def _load_json(self, file_path: str) -> Any:
        """تحميل ملف JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            main_logger.error(f"خطأ في تحميل {file_path}: {e}")
            return [] if 'trades' in file_path else {}
    
    def _save_json(self, file_path: str, data: Any):
        """حفظ ملف JSON"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            main_logger.error(f"خطأ في حفظ {file_path}: {e}")
    
    def add_trade(self, trade_data: Dict[str, Any]) -> str:
        """إضافة صفقة جديدة"""
        try:
            trades = self._load_json(self.trades_file)
            
            # إضافة معرف وطابع زمني
            trade_data['id'] = str(len(trades) + 1)
            trade_data['timestamp'] = datetime.now().isoformat()
            
            trades.append(trade_data)
            self._save_json(self.trades_file, trades)
            
            # تحديث الإحصائيات
            self._update_stats(trade_data)
            
            main_logger.info(f"تم إضافة صفقة: {trade_data['id']}")
            return trade_data['id']
            
        except Exception as e:
            main_logger.error(f"خطأ في إضافة الصفقة: {e}")
            return ""
    
    def get_user_trade_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """الحصول على تاريخ تداول المستخدم"""
        try:
            trades = self._load_json(self.trades_file)
            
            # فلترة صفقات المستخدم
            user_trades = [t for t in trades if t.get('user_id') == user_id]
            
            # ترتيب حسب التاريخ الأحدث
            user_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return user_trades[:limit]
            
        except Exception as e:
            main_logger.error(f"خطأ في الحصول على تاريخ المستخدم: {e}")
            return []
    
    def get_trading_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """الحصول على إحصائيات التداول"""
        try:
            trades = self._load_json(self.trades_file)
            
            if user_id:
                trades = [t for t in trades if t.get('user_id') == user_id]
            
            total_trades = len(trades)
            successful_trades = len([t for t in trades if t.get('status') == 'executed'])
            failed_trades = len([t for t in trades if t.get('status') == 'failed'])
            
            total_profit = sum(t.get('profit_loss', 0) for t in trades)
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': failed_trades,
                'success_rate': round(success_rate, 2),
                'total_profit': round(total_profit, 2)
            }
            
        except Exception as e:
            main_logger.error(f"خطأ في الحصول على الإحصائيات: {e}")
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'success_rate': 0.0,
                'total_profit': 0.0
            }
    
    def _update_stats(self, trade_data: Dict[str, Any]):
        """تحديث الإحصائيات العامة"""
        try:
            stats = self._load_json(self.stats_file)
            
            stats['total_trades'] += 1
            
            if trade_data.get('status') == 'executed':
                stats['successful_trades'] += 1
            elif trade_data.get('status') == 'failed':
                stats['failed_trades'] += 1
            
            stats['total_profit'] += trade_data.get('profit_loss', 0)
            
            self._save_json(self.stats_file, stats)
            
        except Exception as e:
            main_logger.error(f"خطأ في تحديث الإحصائيات: {e}")
    
    def export_history(self, user_id: Optional[int] = None, format: str = 'json') -> str:
        """تصدير تاريخ التداول"""
        try:
            trades = self._load_json(self.trades_file)
            
            if user_id:
                trades = [t for t in trades if t.get('user_id') == user_id]
            
            if format == 'json':
                export_file = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                export_path = os.path.join(self.data_dir, export_file)
                self._save_json(export_path, trades)
                return export_path
            
            elif format == 'csv':
                import csv
                export_file = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                export_path = os.path.join(self.data_dir, export_file)
                
                if trades:
                    with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = trades[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(trades)
                
                return export_path
            
            return ""
            
        except Exception as e:
            main_logger.error(f"خطأ في تصدير التاريخ: {e}")
            return ""
    
    def backup_data(self) -> str:
        """نسخ احتياطي لجميع البيانات"""
        try:
            backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(self.data_dir, backup_file)
            
            backup_data = {
                'trades': self._load_json(self.trades_file),
                'users': self._load_json(self.users_file),
                'stats': self._load_json(self.stats_file),
                'backup_time': datetime.now().isoformat()
            }
            
            self._save_json(backup_path, backup_data)
            main_logger.info(f"تم إنشاء نسخة احتياطية: {backup_path}")
            return backup_path
            
        except Exception as e:
            main_logger.error(f"خطأ في النسخ الاحتياطي: {e}")
            return ""

# إنشاء مثيل عام للتخزين
storage = SimpleStorage()