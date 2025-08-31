import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TradingPredictor:
    def __init__(self, model_path='best_improved_model.keras'):
        """
        سیستم پیش‌بینی ترید با مدل آموزش دیده
        """
        print("🚀 در حال بارگذاری مدل...")
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()
        self.entry_threshold = 0.100
        self.exit_threshold = 0.100
        
        # آماده‌سازی scaler با داده‌های نمونه
        print("🔧 آماده‌سازی scaler...")
        self.prepare_scaler()
        print("✅ مدل آماده است!")
    
    def prepare_scaler(self):
        """آماده‌سازی scaler با استفاده از داده‌های آموزشی"""
        try:
            df = pd.read_csv('training_data.csv')
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'volatility_5m', 'volatility_15m',
                'volume_ratio', 'price_change_1m', 'price_change_5m',
                'price_change_15m', 'trend_strength', 'volatility_percentile',
                'volume_percentile'
            ]
            X = df[feature_columns].values
            self.scaler.fit(X)
        except Exception as e:
            print(f"⚠️ خطا در آماده‌سازی scaler: {e}")
    
    def predict_signals(self, market_data):
        """
        پیش‌بینی سیگنال‌های ورود و خروج
        
        Args:
            market_data: DataFrame یا dict شامل ویژگی‌های بازار
        
        Returns:
            dict: شامل پیش‌بینی‌ها و احتمالات
        """
        try:
            # تبدیل به DataFrame اگر dict باشد
            if isinstance(market_data, dict):
                market_data = pd.DataFrame([market_data])
            
            # انتخاب ویژگی‌ها
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'volatility_5m', 'volatility_15m',
                'volume_ratio', 'price_change_1m', 'price_change_5m',
                'price_change_15m', 'trend_strength', 'volatility_percentile',
                'volume_percentile'
            ]
            
            X = market_data[feature_columns].values
            
            # نرمال‌سازی داده‌ها
            X_scaled = self.scaler.transform(X)
            
            # پیش‌بینی
            predictions = self.model.predict(X_scaled, verbose=0)
            entry_probs = predictions[0]
            exit_probs = predictions[1]
            
            results = []
            for i in range(len(X)):
                entry_prob = float(entry_probs[i][0])
                exit_prob = float(exit_probs[i][0])
                
                result = {
                    'timestamp': market_data.iloc[i].get('timestamp', f'Sample_{i}'),
                    'price': float(market_data.iloc[i]['close']),
                    'entry_probability': entry_prob,
                    'exit_probability': exit_prob,
                    'entry_signal': entry_prob > self.entry_threshold,
                    'exit_signal': exit_prob > self.exit_threshold,
                    'confidence_level': self._calculate_confidence(entry_prob, exit_prob)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ خطا در پیش‌بینی: {e}")
            return None
    
    def _calculate_confidence(self, entry_prob, exit_prob):
        """محاسبه سطح اطمینان"""
        max_prob = max(entry_prob, exit_prob)
        if max_prob > 0.8:
            return "بالا"
        elif max_prob > 0.6:
            return "متوسط"
        else:
            return "پایین"
    
    def analyze_market_condition(self, predictions):
        """تحلیل شرایط بازار"""
        if not predictions:
            return "نامشخص"
        
        entry_signals = sum(1 for p in predictions if p['entry_signal'])
        exit_signals = sum(1 for p in predictions if p['exit_signal'])
        total = len(predictions)
        
        if entry_signals > total * 0.6:
            return "🟢 صعودی - فرصت خرید"
        elif exit_signals > total * 0.6:
            return "🔴 نزولی - فرصت فروش"
        else:
            return "🟡 خنثی - انتظار"

# مثال استفاده
if __name__ == "__main__":
    print("🎯 مثال استفاده از مدل پیش‌بینی ترید")
    print("="*50)
    
    # ایجاد predictor
    predictor = TradingPredictor()
    
    # داده‌های نمونه بازار
    sample_market_data = {
        'timestamp': '2024-01-01 10:00:00',
        'open': 0.2367,
        'high': 0.23688,
        'low': 0.23662,
        'close': 0.23687,
        'volume': 147931.0,
        'rsi': 56.97,
        'macd': -0.00021,
        'macd_signal': -0.00026,
        'macd_histogram': 0.000042,
        'bb_position': 0.543,
        'volatility_5m': 0.000115,
        'volatility_15m': 0.00017,
        'volume_ratio': 0.476,
        'price_change_1m': 0.0007,
        'price_change_5m': 0.00025,
        'price_change_15m': 0.00063,
        'trend_strength': -1.0,
        'volatility_percentile': 0.22,
        'volume_percentile': 0.34
    }
    
    # پیش‌بینی
    print("\n🔮 پیش‌بینی سیگنال‌ها...")
    predictions = predictor.predict_signals(sample_market_data)
    
    if predictions:
        for pred in predictions:
            print(f"\n📊 نتایج برای {pred['timestamp']}:")
            print(f"💰 قیمت: ${pred['price']:.5f}")
            print(f"📈 احتمال ورود: {pred['entry_probability']:.3f}")
            print(f"📉 احتمال خروج: {pred['exit_probability']:.3f}")
            print(f"🎯 سیگنال ورود: {'✅ بله' if pred['entry_signal'] else '❌ خیر'}")
            print(f"🎯 سیگنال خروج: {'✅ بله' if pred['exit_signal'] else '❌ خیر'}")
            print(f"🔒 سطح اطمینان: {pred['confidence_level']}")
        
        print(f"\n🌍 وضعیت کلی بازار: {predictor.analyze_market_condition(predictions)}")
    
    print("\n✅ مثال کامل شد!")
    print("\n📝 نکات مهم:")
    print("- مدل براساس داده‌های تاریخی آموزش دیده")
    print("- همیشه از ریسک منجمنت استفاده کنید")
    print("- این پیش‌بینی‌ها توصیه سرمایه‌گذاری نیستند") 