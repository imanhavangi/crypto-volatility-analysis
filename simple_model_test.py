import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def simple_test():
    print("🔬 تست ساده مدل")
    print("="*30)
    
    # بارگذاری مدل
    print("🚀 بارگذاری مدل...")
    model = tf.keras.models.load_model('best_improved_model.h5')
    print("✅ مدل بارگذاری شد")
    
    # بررسی ورودی مدل
    print(f"📊 ورودی مدل: {model.input_shape}")
    print(f"📊 خروجی مدل: {model.output_shape}")
    
    # تست با داده تصادفی
    print("\n🎲 تست با داده تصادفی:")
    random_input = np.random.randn(1, 19)  # 19 features based on model input
    pred_random = model.predict(random_input, verbose=0)
    print(f"Random Input Prediction: Entry={pred_random[0][0][0]:.4f}, Exit={pred_random[1][0][0]:.4f}")
    
    # تست با داده صفر
    print("\n0️⃣ تست با داده صفر:")
    zero_input = np.zeros((1, 19))
    pred_zero = model.predict(zero_input, verbose=0)
    print(f"Zero Input Prediction: Entry={pred_zero[0][0][0]:.4f}, Exit={pred_zero[1][0][0]:.4f}")
    
    # تست با داده یک
    print("\n1️⃣ تست با داده یک:")
    ones_input = np.ones((1, 19))
    pred_ones = model.predict(ones_input, verbose=0)
    print(f"Ones Input Prediction: Entry={pred_ones[0][0][0]:.4f}, Exit={pred_ones[1][0][0]:.4f}")
    
    # تست با training data واقعی
    print("\n📋 تست با training data:")
    try:
        df = pd.read_csv('training_data.csv', nrows=5)
        print(f"Training data columns: {list(df.columns)}")
        
        # انتخاب 19 فیچر اول (یا آنچه موجود است)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Available features: {len(available_features)} from {len(feature_columns)}")
        
        if len(available_features) >= 19:
            features = df[available_features[:19]].fillna(0).iloc[0].values
            features_scaled = StandardScaler().fit_transform(features.reshape(1, -1))
            
            pred_real = model.predict(features_scaled, verbose=0)
            print(f"Real Data Prediction: Entry={pred_real[0][0][0]:.4f}, Exit={pred_real[1][0][0]:.4f}")
        else:
            print(f"❌ تنها {len(available_features)} فیچر موجود است، 19 فیچر نیاز است")
            
    except Exception as e:
        print(f"❌ خطا در تست training data: {e}")
    
    # نتیجه‌گیری
    print(f"\n💡 نتیجه‌گیری:")
    print(f"اگر همه پیش‌بینی‌ها یکسان باشند، مدل درست آموزش نداده شده است.")

if __name__ == "__main__":
    simple_test() 