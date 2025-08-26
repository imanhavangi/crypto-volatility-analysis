import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import ccxt
import talib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class LiveDataBacktester:
    def __init__(self, model_path='best_improved_model.h5', initial_balance=1000.0):
        """
        سیستم Backtesting با داده‌های جدید از API
        
        Args:
            model_path: مسیر مدل آموزش‌دیده
            initial_balance: بالانس اولیه (دلار)
        """
        print("🚀 بارگذاری مدل...")
        self.model = tf.keras.models.load_model(model_path)
        
        # تنظیمات Exchange
        self.exchange = ccxt.binance()
        
        # تنظیمات مالی
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_fee = 0.001  # 0.1% کارمزد
        
        # تنظیمات ترید
        self.position_size_ratio = 0.25  # 25% بالانس در هر معامله
        self.min_confidence = 0.15  # حداقل اطمینان برای ورود
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        
        print("✅ سیستم آماده است!")
    
    def fetch_ohlcv_data(self, symbol='BTC/USDT', timeframe='1m', start_date=None, end_date=None):
        """
        دریافت داده‌های OHLCV از exchange
        
        Args:
            symbol: نماد ارز (مثل BTC/USDT)
            timeframe: بازه زمانی ('1m', '5m', '1h', etc.)
            start_date: تاریخ شروع (string format: '2024-10-01')
            end_date: تاریخ پایان
        """
        print(f"📡 دریافت داده‌های {symbol} با timeframe {timeframe}...")
        
        # تبدیل تاریخ به timestamp
        if start_date:
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            start_timestamp = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            
        if end_date:
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        
        # دریافت داده‌ها در چندین مرحله
        all_ohlcv = []
        current_timestamp = start_timestamp
        limit = 1000  # حداکثر در هر درخواست
        
        while current_timestamp < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=current_timestamp, 
                    limit=limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + 1  # timestamp آخرین کندل + 1ms
                
                print(f"   دریافت شد: {len(ohlcv)} کندل (مجموع: {len(all_ohlcv)})")
                
            except Exception as e:
                print(f"❌ خطا در دریافت داده: {e}")
                break
        
        # تبدیل به DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        print(f"✅ مجموع {len(df)} کندل دریافت شد")
        return df
    
    def calculate_technical_indicators(self, df):
        """
        محاسبه indicators مشابه training data
        """
        print("🔧 محاسبه Technical Indicators...")
        
        # قیمت‌ها را به numpy array تبدیل کنیم
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Volatility (5m و 15m approximation)
        df['volatility_5m'] = df['close'].rolling(5).std()
        df['volatility_15m'] = df['close'].rolling(15).std()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price changes
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_5m'] = df['close'].pct_change(5)
        df['price_change_15m'] = df['close'].pct_change(15)
        
        # Trend strength (ADX approximation)
        adx = talib.ADX(high, low, close, timeperiod=14)
        df['trend_strength'] = np.where(adx > 25, 1, -1)
        
        # Percentiles
        df['volatility_percentile'] = df['volatility_5m'].rolling(100).rank(pct=True)
        df['volume_percentile'] = df['volume'].rolling(100).rank(pct=True)
        
        # حذف NaN values
        df = df.dropna()
        
        print(f"✅ Indicators محاسبه شد. داده‌های معتبر: {len(df)}")
        return df
    
    def prepare_features_for_model(self, df):
        """
        آماده‌سازی فیچرها برای مدل (مشابه training data)
        """
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
        # استخراج فیچرها و پر کردن NaN
        features_df = df[feature_columns].copy()
        features_df = features_df.fillna(0)
        
        return features_df
    
    def get_prediction(self, features):
        """دریافت پیش‌بینی از مدل"""
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))
        
        # پیش‌بینی
        prediction = self.model.predict(features_scaled, verbose=0)
        
        entry_prob = prediction[0][0][0]  # احتمال ورود
        exit_prob = prediction[1][0][0]   # احتمال خروج
        
        return entry_prob, exit_prob
    
    def execute_trade(self, row, action, confidence):
        """اجرای ترید"""
        price = row['close']
        timestamp = row.name
        
        if action == 'buy' and self.position is None:
            # ورود به معامله
            position_value = self.current_balance * self.position_size_ratio
            size = position_value / price
            fee = position_value * self.trading_fee
            
            if position_value + fee <= self.current_balance:
                self.current_balance -= (position_value + fee)
                self.position = {
                    'entry_price': price,
                    'size': size,
                    'entry_time': timestamp,
                    'entry_fee': fee,
                    'entry_confidence': confidence
                }
                print(f"📈 BUY @ {price:.5f} | Size: {size:.4f} | Confidence: {confidence:.3f}")
                return True
                
        elif action == 'sell' and self.position is not None:
            # خروج از معامله
            position_value = self.position['size'] * price
            fee = position_value * self.trading_fee
            profit = position_value - (self.position['size'] * self.position['entry_price'])
            net_profit = profit - self.position['entry_fee'] - fee
            
            self.current_balance += (position_value - fee)
            
            # ثبت تاریخچه
            trade_record = {
                'entry_time': self.position['entry_time'],
                'exit_time': timestamp,
                'entry_price': self.position['entry_price'],
                'exit_price': price,
                'size': self.position['size'],
                'profit': net_profit,
                'profit_pct': (net_profit / (self.position['size'] * self.position['entry_price'])) * 100,
                'entry_confidence': self.position['entry_confidence'],
                'exit_confidence': confidence,
                'duration_minutes': (timestamp - self.position['entry_time']).total_seconds() / 60
            }
            
            self.trades_history.append(trade_record)
            
            print(f"📉 SELL @ {price:.5f} | P&L: ${net_profit:.2f} ({trade_record['profit_pct']:.2f}%)")
            
            self.position = None
            return True
            
        return False
    
    def run_backtest(self, symbol='BTC/USDT', start_date='2024-10-01', days=7):
        """
        اجرای backtesting کامل
        
        Args:
            symbol: نماد ارز
            start_date: تاریخ شروع
            days: تعداد روزهای تست
        """
        print(f"\n🎯 شروع Backtesting برای {symbol}")
        print(f"📅 از {start_date} برای {days} روز")
        print(f"💰 بالانس اولیه: ${self.initial_balance}")
        print("="*60)
        
        # محاسبه تاریخ پایان
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=days)
        end_date = end_dt.strftime('%Y-%m-%d')
        
        # دریافت داده‌ها
        df = self.fetch_ohlcv_data(symbol, '1m', start_date, end_date)
        
        if len(df) == 0:
            print("❌ هیچ داده‌ای دریافت نشد!")
            return
        
        # محاسبه indicators
        df = self.calculate_technical_indicators(df)
        
        # آماده‌سازی فیچرها
        features_df = self.prepare_features_for_model(df)
        
        # Backtesting
        print(f"\n🔄 شروع شبیه‌سازی ترید...")
        trade_count = 0
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i % 1000 == 0:
                print(f"   پردازش: {i}/{len(df)} ({i/len(df)*100:.1f}%)")
            
            # استخراج فیچرها
            features = features_df.iloc[i].values
            
            # دریافت پیش‌بینی
            entry_prob, exit_prob = self.get_prediction(features)
            
            # تصمیم‌گیری ترید
            if self.position is None and entry_prob > self.min_confidence:
                if self.execute_trade(row, 'buy', entry_prob):
                    trade_count += 1
                    
            elif self.position is not None and exit_prob > self.min_confidence:
                if self.execute_trade(row, 'sell', exit_prob):
                    pass
        
        # بستن موقعیت باز در پایان
        if self.position is not None:
            last_row = df.iloc[-1]
            print("🔚 بستن موقعیت در پایان...")
            self.execute_trade(last_row, 'sell', 0.0)
        
        # نمایش نتایج
        self.display_results()
    
    def display_results(self):
        """نمایش نتایج نهایی"""
        print("\n" + "="*60)
        print("📈 نتایج نهایی BACKTESTING")
        print("="*60)
        
        # نتایج مالی
        total_return = self.current_balance - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        print(f"💰 بالانس نهایی: ${self.current_balance:.2f}")
        print(f"📊 سود/ضرر کل: ${total_return:.2f} ({return_pct:.2f}%)")
        
        if self.trades_history:
            trades_df = pd.DataFrame(self.trades_history)
            
            # آمار کلی
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_profit = trades_df['profit'].mean()
            max_profit = trades_df['profit'].max()
            max_loss = trades_df['profit'].min()
            
            print(f"\n🎯 آمار ترید:")
            print(f"   کل تریدها: {total_trades}")
            print(f"   نرخ برد: {win_rate:.1f}%")
            print(f"   متوسط سود: ${avg_profit:.2f}")
            print(f"   بیشترین سود: ${max_profit:.2f}")
            print(f"   بیشترین ضرر: ${max_loss:.2f}")
            
            # جدول آخرین تریدها
            print(f"\n📋 آخرین 10 ترید:")
            last_trades = trades_df.tail(10)[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'profit_pct']]
            print(last_trades.to_string(index=False))
            
            # نمودار ساده
            if len(trades_df) > 1:
                plt.figure(figsize=(12, 6))
                
                # نمودار تجمعی سود
                cumulative_pnl = trades_df['profit'].cumsum() + self.initial_balance
                plt.subplot(1, 2, 1)
                plt.plot(cumulative_pnl.values, linewidth=2)
                plt.title('تغییرات بالانس', fontsize=12, fontweight='bold')
                plt.ylabel('بالانس (دلار)')
                plt.grid(True, alpha=0.3)
                
                # نمودار توزیع سود/ضرر
                plt.subplot(1, 2, 2)
                profits = trades_df['profit']
                plt.hist(profits, bins=20, alpha=0.7, color='green' if profits.mean() > 0 else 'red')
                plt.title('توزیع سود/ضرر', fontsize=12, fontweight='bold')
                plt.xlabel('سود/ضرر (دلار)')
                plt.ylabel('تعداد')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('live_backtest_results.png', dpi=300, bbox_inches='tight')
                plt.show()
        else:
            print("\n⚠️ هیچ تریدی انجام نشد!")

def main():
    """اجرای اصلی"""
    print("🚀 سیستم Live Data Backtesting")
    print("="*50)
    
    # ایجاد backtester
    backtester = LiveDataBacktester(
        model_path='best_improved_model.h5',
        initial_balance=1000.0
    )
    
    # اجرای backtest روی BTC/USDT در اکتبر 2024 (3 ماه پیش)
    backtester.run_backtest(
        symbol='BTC/USDT',
        start_date='2024-10-15',  # 15 اکتبر 2024
        days=7  # یک هفته
    )

if __name__ == "__main__":
    main() 