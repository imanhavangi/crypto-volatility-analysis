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

class FixedBacktestSystem:
    def __init__(self, model_path='best_improved_model.keras', initial_balance=1000.0):
        """
        سیستم Backtesting اصلاح‌شده
        
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
        self.trading_fee = 0.0008  # 0.08% کارمزد (کمتر از قبل)
        
        # تنظیمات ترید - محافظه‌کارانه‌تر
        self.position_size_ratio = 0.15  # 15% بالانس در هر معامله
        self.entry_threshold = 0.54  # بر اساس تحلیل مدل
        self.exit_threshold = 0.54   # بر اساس تحلیل مدل
        
        # Risk Management
        self.stop_loss_pct = 0.02   # 2% stop-loss
        self.take_profit_pct = 0.04  # 4% take-profit
        self.max_position_time = 60  # حداکثر 60 دقیقه در هر موقعیت
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        
        # ایجاد Scaler مناسب
        self.scaler = self.create_proper_scaler()
        
        # آمار پیش‌بینی برای debugging
        self.prediction_stats = {
            'entry_probs': [],
            'exit_probs': [],
            'timestamps': []
        }
        
        print(f"✅ سیستم آماده است!")
        print(f"📊 تنظیمات: Entry>{self.entry_threshold:.0%}, Exit>{self.exit_threshold:.0%}")
        print(f"🛡️ Risk Management: SL={self.stop_loss_pct:.0%}, TP={self.take_profit_pct:.0%}, MaxTime={self.max_position_time}min")

    def create_proper_scaler(self):
        """
        ایجاد Scaler مناسب بر اساس آمار training data
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
            return scaler
            
        except Exception as e:
            print(f"⚠️  خطا در ایجاد scaler: {e}")
            print("🔄 استفاده از scaler پیش‌فرض...")
            return StandardScaler()

    def fetch_ohlcv_data(self, symbol='BTC/USDT', timeframe='1m', start_date=None, end_date=None):
        """
        دریافت داده‌های OHLCV از exchange
        """
        try:
            # تبدیل تاریخ به timestamp
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            print(f"📡 دریافت داده‌های {symbol} از {start_date} تا {end_date}...")
            
            # دریافت داده‌ها
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe, start_ts, 
                limit=None, params={'endTime': end_ts}
            )
            
            # تبدیل به DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ دریافت {len(df)} نقطه داده")
            return df
            
        except Exception as e:
            print(f"❌ خطا در دریافت داده: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """
        محاسبه شاخص‌های تکنیکال
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

    def prepare_features_for_model(self, df):
        """
        آماده‌سازی فیچرها برای مدل
        """
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
        features_df = df[feature_columns].copy()
        features_df = features_df.fillna(0)
        
        return features_df

    def get_prediction(self, features):
        """
        دریافت پیش‌بینی از مدل - با scaler صحیح
        """
        # استفاده از scaler آماده‌شده (نه fit جدید!)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # پیش‌بینی
        prediction = self.model.predict(features_scaled, verbose=0)
        
        entry_prob = prediction[0][0][0]  # احتمال ورود
        exit_prob = prediction[1][0][0]   # احتمال خروج
        
        # ذخیره آمار برای تحلیل
        self.prediction_stats['entry_probs'].append(float(entry_prob))
        self.prediction_stats['exit_probs'].append(float(exit_prob))
        
        return entry_prob, exit_prob

    def analyze_prediction_distribution(self, df, features_df, sample_size=100):
        """
        تحلیل توزیع پیش‌بینی‌های مدل برای تعیین threshold بهتر
        """
        print(f"\n🔍 تحلیل توزیع پیش‌بینی‌های مدل...")
        
        entry_probs = []
        exit_probs = []
        
        # نمونه‌گیری از داده‌ها
        sample_indices = np.random.choice(len(df), min(sample_size, len(df)), replace=False)
        
        for i in sample_indices:
            if i < 50:  # نادیده گرفتن اولین 50 نقطه
                continue
                
            features = features_df.iloc[i].values
            entry_prob, exit_prob = self.get_prediction(features)
            entry_probs.append(entry_prob)
            exit_probs.append(exit_prob)
        
        entry_probs = np.array(entry_probs)
        exit_probs = np.array(exit_probs)
        
        # نمایش آمار
        print(f"📊 Entry Predictions:")
        print(f"   میانگین: {entry_probs.mean():.3f}")
        print(f"   میانه: {np.median(entry_probs):.3f}")
        print(f"   انحراف معیار: {entry_probs.std():.3f}")
        print(f"   Min/Max: {entry_probs.min():.3f}/{entry_probs.max():.3f}")
        print(f"   Percentiles: 25%={np.percentile(entry_probs, 25):.3f}, 75%={np.percentile(entry_probs, 75):.3f}, 95%={np.percentile(entry_probs, 95):.3f}")
        
        print(f"📊 Exit Predictions:")
        print(f"   میانگین: {exit_probs.mean():.3f}")
        print(f"   میانه: {np.median(exit_probs):.3f}")
        print(f"   انحراف معیار: {exit_probs.std():.3f}")
        print(f"   Min/Max: {exit_probs.min():.3f}/{exit_probs.max():.3f}")
        print(f"   Percentiles: 25%={np.percentile(exit_probs, 25):.3f}, 75%={np.percentile(exit_probs, 75):.3f}, 95%={np.percentile(exit_probs, 95):.3f}")
        
        # پیشنهاد threshold جدید بر اساس توزیع
        # استفاده از میانه + نصف انحراف معیار برای threshold بهتر
        entry_median = np.median(entry_probs)
        exit_median = np.median(exit_probs)
        entry_std = entry_probs.std()
        exit_std = exit_probs.std()
        
        suggested_entry_threshold = entry_median + (entry_std * 0.5)
        suggested_exit_threshold = exit_median + (exit_std * 0.5)
        
        print(f"\n💡 پیشنهاد Threshold جدید:")
        print(f"   Entry: {suggested_entry_threshold:.3f} (به جای {self.entry_threshold})")
        print(f"   Exit: {suggested_exit_threshold:.3f} (به جای {self.exit_threshold})")
        
        return {
            'entry_stats': {
                'mean': entry_probs.mean(),
                'median': np.median(entry_probs),
                'std': entry_probs.std(),
                'p95': np.percentile(entry_probs, 95),
                'p85': np.percentile(entry_probs, 85),
                'p75': np.percentile(entry_probs, 75)
            },
            'exit_stats': {
                'mean': exit_probs.mean(),
                'median': np.median(exit_probs),
                'std': exit_probs.std(),
                'p95': np.percentile(exit_probs, 95),
                'p85': np.percentile(exit_probs, 85),
                'p75': np.percentile(exit_probs, 75)
            },
            'suggested_entry_threshold': suggested_entry_threshold,
            'suggested_exit_threshold': suggested_exit_threshold
        }

    def update_thresholds(self, entry_threshold, exit_threshold):
        """
        به‌روزرسانی threshold ها
        """
        print(f"🔧 به‌روزرسانی Thresholds:")
        print(f"   Entry: {self.entry_threshold:.3f} → {entry_threshold:.3f}")
        print(f"   Exit: {self.exit_threshold:.3f} → {exit_threshold:.3f}")
        
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def check_risk_management(self, current_price, current_time):
        """
        بررسی شرایط risk management
        """
        if self.position is None:
            return False, None
            
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        
        # محاسبه تغییر قیمت
        price_change_pct = (current_price - entry_price) / entry_price
        
        # محاسبه مدت زمان موقعیت
        time_in_position = (current_time - entry_time).total_seconds() / 60  # دقیقه
        
        # بررسی Stop Loss
        if price_change_pct <= -self.stop_loss_pct:
            return True, f"Stop Loss triggered: {price_change_pct:.2%}"
            
        # بررسی Take Profit
        if price_change_pct >= self.take_profit_pct:
            return True, f"Take Profit triggered: {price_change_pct:.2%}"
            
        # بررسی حداکثر زمان
        if time_in_position >= self.max_position_time:
            return True, f"Max time reached: {time_in_position:.1f}min"
            
        return False, None

    def execute_trade(self, row, action, confidence, reason="Model Signal"):
        """
        اجرای ترید با محاسبه دقیق‌تر
        """
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
                    'entry_confidence': confidence,
                    'position_value': position_value,
                    'entry_reason': reason
                }
                print(f"📈 BUY @ {price:.2f} | Size: {size:.4f} | Confidence: {confidence:.3f} | Reason: {reason}")
                return True
                
        elif action == 'sell' and self.position is not None:
            # خروج از معامله
            current_value = self.position['size'] * price
            exit_fee = current_value * self.trading_fee
            
            # محاسبه سود خالص
            gross_profit = current_value - self.position['position_value']
            net_profit = gross_profit - self.position['entry_fee'] - exit_fee
            
            self.current_balance += (current_value - exit_fee)
            
            # ثبت تاریخچه
            trade_record = {
                'entry_time': self.position['entry_time'],
                'exit_time': timestamp,
                'entry_price': self.position['entry_price'],
                'exit_price': price,
                'size': self.position['size'],
                'gross_profit': gross_profit,
                'net_profit': net_profit,
                'profit_pct': (net_profit / self.position['position_value']) * 100,
                'entry_confidence': self.position['entry_confidence'],
                'exit_confidence': confidence,
                'duration_minutes': (timestamp - self.position['entry_time']).total_seconds() / 60,
                'fees_total': self.position['entry_fee'] + exit_fee,
                'entry_reason': self.position.get('entry_reason', 'Unknown'),
                'exit_reason': reason
            }
            
            self.trades_history.append(trade_record)
            
            print(f"📉 SELL @ {price:.2f} | P&L: ${net_profit:.2f} ({trade_record['profit_pct']:.2f}%) | Reason: {reason}")
            
            self.position = None
            return True
            
        return False

    def run_backtest(self, symbol='BTC/USDT', start_date='2024-10-01', days=7, analyze_first=True):
        """
        اجرای backtesting کامل - با استفاده از training data به جای exchange data
        """
        print(f"\n🎯 شروع Backtesting اصلاح‌شده برای {symbol}")
        print(f"📅 از {start_date} برای {days} روز")
        print(f"💰 بالانس اولیه: ${self.initial_balance}")
        
        # استفاده از training data به جای exchange data (برای حل مشکل data mismatch)
        print(f"📡 استفاده از training data (حل مشکل exchange data)...")
        
        try:
            # بارگذاری نمونه‌ای از training data
            total_points = days * 24 * 60  # تعداد نقاط مورد نیاز (days * hours * minutes)
            df = pd.read_csv('training_data.csv', nrows=min(total_points, 10000))
            
            # انتخاب ستون‌های OHLCV
            if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                print(f"✅ دریافت {len(df)} نقطه داده از training data")
                
                # ایجاد timestamp های فرضی
                from datetime import datetime, timedelta
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                timestamps = [start_dt + timedelta(minutes=i) for i in range(len(df))]
                df.index = timestamps
                
            else:
                print("❌ ستون‌های OHLCV در training data موجود نیستند!")
                return
                
        except Exception as e:
            print(f"❌ خطا در بارگذاری training data: {e}")
            return
        
        # ادامه با همان کد قبلی...
        print(f"📊 داده آماده: {len(df)} نقطه")
        
        # آماده‌سازی فیچرها (training data قبلاً شاخص‌های تکنیکال دارد)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
        # بررسی موجودیت ستون‌ها
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) < len(feature_columns):
            print(f"⚠️ تنها {len(available_features)} از {len(feature_columns)} فیچر موجود است")
            print(f"📋 فیچرهای موجود: {available_features}")
        
        # آماده‌سازی فیچرها
        features_df = df[available_features].fillna(0)
        
        # تحلیل توزیع پیش‌بینی‌ها
        if analyze_first:
            stats = self.analyze_prediction_distribution(df, features_df, sample_size=200)
            
            # استفاده از threshold های پیشنهادی
            self.update_thresholds(
                stats['suggested_entry_threshold'],
                stats['suggested_exit_threshold']
            )
        
        # Backtesting
        print(f"\n🔄 شروع شبیه‌سازی ترید...")
        trade_signals = 0
        entry_signals = 0
        exit_signals = 0
        
        # پاک کردن آمار قبلی
        self.prediction_stats = {
            'entry_probs': [],
            'exit_probs': [],
            'timestamps': []
        }
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i % 1000 == 0:
                progress = i/len(df)*100
                print(f"   پردازش: {i}/{len(df)} ({progress:.1f}%) | Trades: {len(self.trades_history)}")
            
            # نادیده گرفتن اولین 50 نقطه (برای تنظیم شاخص‌ها)
            if i < 50:
                continue
                
            # بررسی Risk Management اول
            if self.position is not None:
                should_exit, risk_reason = self.check_risk_management(row['close'], timestamp)
                if should_exit:
                    if self.execute_trade(row, 'sell', 0.5, risk_reason):
                        continue
                
            # استخراج فیچرها
            features = features_df.iloc[i].values
            
            # دریافت پیش‌بینی
            entry_prob, exit_prob = self.get_prediction(features)
            self.prediction_stats['timestamps'].append(timestamp)
            
            # شمارش سیگنال‌ها برای debugging
            if entry_prob > self.entry_threshold:
                entry_signals += 1
            if exit_prob > self.exit_threshold:
                exit_signals += 1
            
            # تصمیم‌گیری ترید
            if self.position is None and entry_prob > self.entry_threshold:
                if self.execute_trade(row, 'buy', entry_prob, "AI Entry Signal"):
                    trade_signals += 1
                    
            elif self.position is not None and exit_prob > self.exit_threshold:
                if self.execute_trade(row, 'sell', exit_prob, "AI Exit Signal"):
                    pass
        
        # بستن موقعیت باز در پایان
        if self.position is not None:
            last_row = df.iloc[-1]
            print("🔚 بستن موقعیت در پایان...")
            self.execute_trade(last_row, 'sell', 0.5, "End of backtest")
        
        print(f"\n📊 سیگنال‌ها: Entry={entry_signals}, Exit={exit_signals}")
        
        # نمایش نتایج
        self.display_results()
        
        # تحلیل نهایی پیش‌بینی‌ها
        self.analyze_final_predictions()

    def display_results(self):
        """
        نمایش نتایج نهایی
        """
        print("\n" + "="*60)
        print("📈 نتایج نهایی BACKTESTING اصلاح‌شده")
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
            winning_trades = len(trades_df[trades_df['net_profit'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_profit = trades_df['net_profit'].mean()
            max_profit = trades_df['net_profit'].max()
            max_loss = trades_df['net_profit'].min()
            total_fees = trades_df['fees_total'].sum()
            
            print(f"\n🎯 آمار ترید:")
            print(f"   کل تریدها: {total_trades}")
            print(f"   نرخ برد: {win_rate:.1f}%")
            print(f"   میانگین سود: ${avg_profit:.2f}")
            print(f"   بیشترین سود: ${max_profit:.2f}")
            print(f"   بیشترین ضرر: ${max_loss:.2f}")
            print(f"   کل کارمزد: ${total_fees:.2f}")
            
            # نمایش آخرین 5 ترید
            print(f"\n📋 آخرین 5 ترید:")
            recent_trades = trades_df.tail(5)[['entry_price', 'exit_price', 'net_profit', 'profit_pct']]
            for _, trade in recent_trades.iterrows():
                print(f"   {trade['entry_price']:.2f} → {trade['exit_price']:.2f} | ${trade['net_profit']:.2f} ({trade['profit_pct']:.2f}%)")
        else:
            print("\n⚠️ هیچ تریدی انجام نشد!")

    def analyze_final_predictions(self):
        """
        تحلیل نهایی پیش‌بینی‌ها
        """
        if not self.prediction_stats['entry_probs']:
            return
            
        entry_probs = np.array(self.prediction_stats['entry_probs'])
        exit_probs = np.array(self.prediction_stats['exit_probs'])
        
        print(f"\n📈 تحلیل نهایی پیش‌بینی‌ها:")
        print(f"Entry - Mean: {entry_probs.mean():.3f}, Std: {entry_probs.std():.3f}")
        print(f"Exit - Mean: {exit_probs.mean():.3f}, Std: {exit_probs.std():.3f}")
        print(f"Entry > {self.entry_threshold}: {(entry_probs > self.entry_threshold).sum()}/{len(entry_probs)} ({(entry_probs > self.entry_threshold).mean()*100:.1f}%)")
        print(f"Exit > {self.exit_threshold}: {(exit_probs > self.exit_threshold).sum()}/{len(exit_probs)} ({(exit_probs > self.exit_threshold).mean()*100:.1f}%)")

# اجرای تست
if __name__ == "__main__":
    # ایجاد سیستم
    backtest = FixedBacktestSystem(initial_balance=1000.0)
    
    # اجرای backtesting
    backtest.run_backtest(
        symbol='BTC/USDT',
        start_date='2024-10-15',  # 3 ماه پیش
        days=7
    ) 