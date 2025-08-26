import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelDiagnostic:
    def __init__(self):
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
    def prepare_test_data(self):
        """آماده‌سازی داده تست"""
        print("📋 بارگذاری داده‌های تست...")
        
        # بارگذاری نمونه از training data
        df = pd.read_csv('training_data.csv', nrows=100)
        
        # انتخاب فیچرها
        features = df[self.feature_columns].fillna(0)
        
        # ایجاد scaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"✅ آماده‌سازی {len(features)} نمونه داده")
        return features_scaled, scaler
        
    def test_model(self, model_path, test_data, model_name):
        """تست یک مدل خاص"""
        print(f"\n🔍 تست مدل: {model_name}")
        print(f"📁 مسیر: {model_path}")
        
        try:
            # بارگذاری مدل
            model = tf.keras.models.load_model(model_path)
            print(f"✅ مدل بارگذاری شد")
            
            # نمایش ساختار مدل (ساده)
            print(f"📊 ساختار مدل:")
            try:
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
            except:
                print(f"   Model loaded successfully")
            
            # تست با نمونه داده
            sample_size = min(10, len(test_data))
            predictions = []
            
            for i in range(sample_size):
                sample = test_data[i:i+1]  # یک نمونه
                pred = model.predict(sample, verbose=0)
                predictions.append(pred)
            
            # تحلیل نتایج
            print(f"\n📈 نتایج پیش‌بینی ({sample_size} نمونه):")
            
            if isinstance(predictions[0], list):
                # چند خروجی
                num_outputs = len(predictions[0])
                print(f"   تعداد خروجی‌ها: {num_outputs}")
                
                for output_idx in range(num_outputs):
                    output_name = f"Output_{output_idx+1}"
                    if output_idx == 0:
                        output_name = "Entry"
                    elif output_idx == 1:
                        output_name = "Exit"
                    elif output_idx == 2:
                        output_name = "Profit"
                    
                    values = [pred[output_idx][0][0] for pred in predictions]
                    
                    print(f"   {output_name}:")
                    print(f"     Mean: {np.mean(values):.4f}")
                    print(f"     Std: {np.std(values):.4f}")
                    print(f"     Min/Max: {np.min(values):.4f}/{np.max(values):.4f}")
                    print(f"     Sample values: {[f'{v:.4f}' for v in values[:5]]}")
            else:
                # یک خروجی
                values = [pred[0][0] for pred in predictions]
                print(f"   Single Output:")
                print(f"     Mean: {np.mean(values):.4f}")
                print(f"     Std: {np.std(values):.4f}")
                print(f"     Min/Max: {np.min(values):.4f}/{np.max(values):.4f}")
                print(f"     Sample values: {[f'{v:.4f}' for v in values[:5]]}")
            
            return True, predictions
            
        except Exception as e:
            print(f"❌ خطا در تست مدل: {e}")
            return False, None
    
    def run_diagnostic(self):
        """اجرای تشخیص کامل"""
        print("🔬 شروع تشخیص مدل‌ها")
        print("="*50)
        
        # آماده‌سازی داده‌های تست
        test_data, scaler = self.prepare_test_data()
        
        # تست مدل‌های موجود
        models_to_test = [
            ('best_improved_model.h5', 'Improved Model'),
            ('best_trading_model.h5', 'Original Deep Learning Model')
        ]
        
        results = {}
        
        for model_path, model_name in models_to_test:
            success, predictions = self.test_model(model_path, test_data, model_name)
            results[model_name] = {
                'success': success,
                'predictions': predictions,
                'path': model_path
            }
        
        # خلاصه نتایج
        print(f"\n📊 خلاصه نتایج:")
        print("="*50)
        
        working_models = []
        for name, result in results.items():
            if result['success']:
                print(f"✅ {name}: کار می‌کند")
                working_models.append((name, result['path']))
            else:
                print(f"❌ {name}: مشکل دارد")
        
        if working_models:
            print(f"\n💡 پیشنهاد:")
            for name, path in working_models:
                print(f"   استفاده از {name}: {path}")
        else:
            print(f"\n⚠️ هیچ مدل سالمی یافت نشد! نیاز به آموزش مدل جدید")
        
        return results

if __name__ == "__main__":
    diagnostic = ModelDiagnostic()
    results = diagnostic.run_diagnostic() 