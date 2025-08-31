import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import ccxt
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_proper_scaler():
    """
    ایجاد Scaler مناسب بر اساس آمار training data - همان کد backtest
    """
    print("🔧 آماده‌سازی StandardScaler...")
    
    # بارگذاری نمونه‌ای از training data برای fit کردن scaler
    try:
        sample_data = pd.read_csv('training_data.csv', nrows=1000)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
        features = sample_data[feature_columns].fillna(0)
        scaler = StandardScaler()
        scaler.fit(features)
        print("✅ Scaler آماده شد!")
        return scaler, feature_columns
        
    except Exception as e:
        print(f"⚠️  خطا در ایجاد scaler: {e}")
        print("🔄 استفاده از scaler پیش‌فرض...")
        return StandardScaler(), []

def calculate_technical_indicators(df):
    """
    محاسبه شاخص‌های تکنیکال - همان کد backtest
    """
    print("🔢 محاسبه شاخص‌های تکنیکال...")
    
    df = df.copy()
    
    # RSI
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Bollinger Bands Position
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values)
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Volatility
    df['volatility_5m'] = df['close'].rolling(5).std()
    df['volatility_15m'] = df['close'].rolling(15).std()
    
    # Volume Ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Price Changes
    df['price_change_1m'] = df['close'].pct_change(1)
    df['price_change_5m'] = df['close'].pct_change(5)
    df['price_change_15m'] = df['close'].pct_change(15)
    
    # Trend Strength (simple)
    df['trend_strength'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
    
    # Percentiles
    df['volatility_percentile'] = df['volatility_5m'].rolling(100).rank(pct=True)
    df['volume_percentile'] = df['volume'].rolling(100).rank(pct=True)
    
    # پر کردن NaN
    df = df.fillna(method='ffill').fillna(0)
    
    print("✅ شاخص‌ها محاسبه شدند")
    return df

def prepare_features_for_model(df, feature_columns):
    """
    آماده‌سازی فیچرها برای مدل - همان کد backtest
    """
    features_df = df[feature_columns].copy()
    features_df = features_df.fillna(0)
    return features_df

def main():
    print("🔬 تست دقیق Input مدل - مشابه backtest")
    print("="*50)
    
    # بارگذاری مدل
    print("🚀 بارگذاری مدل...")
    model = tf.keras.models.load_model('best_improved_model.keras')
    
    # ایجاد scaler مشابه backtest
    scaler, feature_columns = create_proper_scaler()
    
    # شبیه‌سازی fetch داده (نمونه کوچک)
    print("📡 شبیه‌سازی دریافت داده...")
    
    # بارگذاری نمونه داده برای تست
    df_sample = pd.read_csv('training_data.csv', nrows=100)
    
    # اطمینان از وجود ستون‌های مورد نیاز
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(col in df_sample.columns for col in required_cols):
        print("✅ داده‌های OHLCV موجود")
        
        # محاسبه شاخص‌های تکنیکال
        df_with_indicators = calculate_technical_indicators(df_sample)
        
        # آماده‌سازی فیچرها
        features_df = prepare_features_for_model(df_with_indicators, feature_columns)
        
        print(f"📊 تست با {len(features_df)} نقطه داده")
        
        # تست چند نمونه
        predictions = []
        for i in range(min(10, len(features_df))):
            if i < 50:  # شرط backtest
                continue
                
            features = features_df.iloc[i].values
            
            # استفاده از scaler (مشابه backtest)
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # پیش‌بینی
            prediction = model.predict(features_scaled, verbose=0)
            
            entry_prob = prediction[0][0][0]
            exit_prob = prediction[1][0][0]
            
            predictions.append((entry_prob, exit_prob))
            
            if i < 5:  # چاپ اولین 5 نمونه
                print(f"   Sample {i}: Entry={entry_prob:.4f}, Exit={exit_prob:.4f}")
        
        if predictions:
            entry_probs = [p[0] for p in predictions]
            exit_probs = [p[1] for p in predictions]
            
            print(f"\n📊 خلاصه {len(predictions)} پیش‌بینی:")
            print(f"Entry - Mean: {np.mean(entry_probs):.4f}, Std: {np.std(entry_probs):.4f}, Range: {np.min(entry_probs):.4f}-{np.max(entry_probs):.4f}")
            print(f"Exit  - Mean: {np.mean(exit_probs):.4f}, Std: {np.std(exit_probs):.4f}, Range: {np.min(exit_probs):.4f}-{np.max(exit_probs):.4f}")
            
            # بررسی اگر همه مقادیر یکسان باشند
            if np.std(entry_probs) == 0:
                print("⚠️ همه Entry predictions یکسان هستند!")
            if np.std(exit_probs) == 0:
                print("⚠️ همه Exit predictions یکسان هستند!")
        
    else:
        print("❌ ستون‌های OHLCV موجود نیستند")

if __name__ == "__main__":
    main() 