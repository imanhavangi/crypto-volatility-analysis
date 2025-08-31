import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss بهبود یافته برای حل مشکل class imbalance شدید
    """
    def __init__(self, alpha=0.99, gamma=3.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # افزایش alpha برای کلاس اقلیت
        self.gamma = gamma  # افزایش gamma برای تمرکز بیشتر بر نمونه‌های سخت

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        focal_loss = -alpha_t * tf.math.pow(1 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal_loss)

class OptimizedBacktester:
    """
    سیستم Backtesting بهینه‌شده با تنظیمات محافظه‌کارانه
    """
    
    def __init__(self, model_path='best_improved_model.keras', 
                 model_info_path='model_info.json', 
                 initial_balance=1000.0):
        """
        مقداردهی سیستم Backtesting بهینه‌شده
        """
        print("🚀 بارگذاری مدل بهبود یافته...")
        
        # بارگذاری مدل با custom objects
        self.model = tf.keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
        
        # بارگذاری اطلاعات مدل
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # تنظیمات داده
        self.data_file = 'training_data.csv'
        
        # تنظیمات مالی
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_fee = 0.0008  # 0.08% کارمزد
        
        # 🎯 تنظیمات ترید بهینه‌شده (محافظه‌کارانه‌تر)
        self.position_size_ratio = 0.15  # 15% بالانس در هر معامله (کاهش از 20%)
        
        # 🔧 Threshold های بهینه‌شده
        self.entry_threshold = 0.85   # افزایش قابل توجه (از 0.05 به 0.85)
        self.exit_threshold = 0.75    # افزایش (از 0.6 به 0.75)
        
        print(f"📊 Entry Threshold (بهینه): {self.entry_threshold}")
        print(f"📊 Exit Threshold (بهینه): {self.exit_threshold}")
        
        # Risk Management پیشرفته
        self.stop_loss_pct = 0.015    # 1.5% stop-loss (سخت‌تر)
        self.take_profit_pct = 0.03   # 3% take-profit (محافظه‌کارانه‌تر)
        self.max_position_time = 30   # حداکثر 30 دقیقه (کاهش از 90)
        self.min_volume_filter = 2000000  # حداقل volume بالاتر
        
        # فیلترهای اضافی برای کیفیت سیگنال
        self.min_confidence_gap = 0.1  # حداقل فاصله بین entry و exit probability
        self.consecutive_signal_limit = 3  # حداکثر 3 سیگنال متوالی
        self.cooldown_period = 5  # 5 کندل استراحت بعد از هر ترید
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        self.total_positions = 0
        self.last_trade_index = -10  # برای cooldown
        self.consecutive_signals = 0
        
        # ایجاد Scaler مناسب
        self.scaler = StandardScaler()
        self.feature_columns = self.model_info['feature_columns']
        
        print("✅ سیستم بهینه‌شده آماده است!")
        print(f"🎯 Features: {len(self.feature_columns)} فیچر")
        print("🔧 تنظیمات محافظه‌کارانه فعال شد")
    
    def load_historical_data(self, start_row=None, end_row=None):
        """
        بارگذاری داده‌های تاریخی از فایل CSV
        """
        print(f"📥 بارگذاری داده‌های تاریخی از {self.data_file}...")
        
        try:
            # بارگذاری داده‌ها
            df = pd.read_csv(self.data_file)
            
            # تبدیل timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # انتخاب بازه مشخص
            if start_row is not None and end_row is not None:
                df = df.iloc[start_row:end_row]
            elif start_row is not None:
                df = df.iloc[start_row:]
            elif end_row is not None:
                df = df.iloc[:end_row]
            
            print(f"✅ {len(df)} رکورد بارگذاری شد")
            if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                print(f"📅 از {df.index.min()} تا {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری داده‌ها: {e}")
            return None
    
    def prepare_data(self, df):
        """
        آماده‌سازی داده‌ها (فیچرها از قبل محاسبه شده‌اند)
        """
        print("🔧 آماده‌سازی داده‌ها...")
        
        # بررسی موجودیت فیچرهای مورد نیاز
        missing_features = []
        for feature in self.feature_columns:
            if feature not in df.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(f"⚠️ فیچرهای گمشده: {missing_features}")
        
        # پاک کردن NaN ها
        df_clean = df.dropna()
        
        print(f"✅ {len(df_clean)} رکورد آماده برای پردازش")
        return df_clean
    
    def prepare_features(self, df):
        """
        آماده‌سازی فیچرها برای مدل
        """
        # انتخاب فیچرهای مورد نیاز
        feature_data = df[self.feature_columns].copy()
        
        # Scaling
        if not hasattr(self.scaler, 'scale_'):
            # اگر scaler fit نشده، از آمار کلی استفاده می‌کنیم
            self.scaler.fit(feature_data)
        
        scaled_features = self.scaler.transform(feature_data)
        
        return scaled_features
    
    def get_trading_signals_optimized(self, df):
        """
        تولید سیگنال‌های ترید بهینه‌شده با فیلترهای اضافی
        """
        print("🎯 تولید سیگنال‌های ترید بهینه‌شده...")
        
        # آماده‌سازی فیچرها
        features = self.prepare_features(df)
        
        # پیش‌بینی مدل
        predictions = self.model.predict(features, verbose=0)
        entry_probs = predictions[0].flatten()
        exit_probs = predictions[1].flatten()
        
        # اعمال threshold های بهینه‌شده
        entry_signals_raw = (entry_probs > self.entry_threshold).astype(int)
        exit_signals_raw = (exit_probs > self.exit_threshold).astype(int)
        
        # 🔍 فیلترهای کیفیت سیگنال
        entry_signals_filtered = []
        exit_signals_filtered = []
        
        for i in range(len(entry_probs)):
            # فیلتر 1: اطمینان gap
            confidence_gap = abs(entry_probs[i] - exit_probs[i])
            
            # فیلتر 2: کیفیت سیگنال
            entry_quality = entry_probs[i] > (self.entry_threshold + 0.05)  # 5% اضافی
            exit_quality = exit_probs[i] > (self.exit_threshold + 0.05)    # 5% اضافی
            
            # Entry signal با فیلتر
            if (entry_signals_raw[i] == 1 and 
                confidence_gap > self.min_confidence_gap and 
                entry_quality):
                entry_signals_filtered.append(1)
            else:
                entry_signals_filtered.append(0)
            
            # Exit signal با فیلتر
            if (exit_signals_raw[i] == 1 and exit_quality):
                exit_signals_filtered.append(1)
            else:
                exit_signals_filtered.append(0)
        
        # اضافه کردن به DataFrame
        df_signals = df.copy()
        df_signals['entry_prob'] = entry_probs
        df_signals['exit_prob'] = exit_probs
        df_signals['entry_signal'] = entry_signals_filtered
        df_signals['exit_signal'] = exit_signals_filtered
        
        print(f"📊 Entry signals (خام): {entry_signals_raw.sum()}")
        print(f"📊 Entry signals (فیلتر شده): {sum(entry_signals_filtered)}")
        print(f"📊 Exit signals (خام): {exit_signals_raw.sum()}")
        print(f"📊 Exit signals (فیلتر شده): {sum(exit_signals_filtered)}")
        
        return df_signals
    
    def should_enter_trade(self, row, current_index):
        """
        بررسی شرایط ورود به ترید با فیلترهای اضافی
        """
        # شرایط اصلی
        if row['entry_signal'] != 1:
            return False
        
        # Cooldown period
        if current_index - self.last_trade_index < self.cooldown_period:
            return False
        
        # Volume filter
        if row['volume'] < self.min_volume_filter:
            return False
        
        # Consecutive signals limit
        if self.consecutive_signals >= self.consecutive_signal_limit:
            return False
        
        return True
    
    def execute_backtest_optimized(self, df_signals, symbol='DOGE/USDT'):
        """
        اجرای Backtest بهینه‌شده
        """
        print("🔄 اجرای Backtest بهینه‌شده...")
        
        self.current_balance = self.initial_balance
        self.position = None
        self.trades_history = []
        self.total_positions = 0
        self.last_trade_index = -10
        self.consecutive_signals = 0
        
        portfolio_values = []
        candle_index = 0
        
        for timestamp, row in df_signals.iterrows():
            current_price = row['close']
            current_volume = row['volume']
            
            # محاسبه ارزش فعلی پورتفولیو
            portfolio_value = self.current_balance
            if self.position:
                portfolio_value += self.position['quantity'] * current_price
            portfolio_values.append(portfolio_value)
            
            # مدیریت موقعیت موجود
            if self.position:
                self._manage_existing_position_optimized(row, candle_index, timestamp)
            
            # بررسی سیگنال ورود جدید
            if not self.position and self.should_enter_trade(row, candle_index):
                self._open_position_optimized(row, candle_index, timestamp, symbol)
                self.last_trade_index = candle_index
                self.consecutive_signals += 1
            else:
                # Reset consecutive signals if no entry
                if row['entry_signal'] != 1:
                    self.consecutive_signals = 0
            
            candle_index += 1
        
        # بستن موقعیت باقی‌مانده
        if self.position:
            final_row = df_signals.iloc[-1]
            self._close_position_optimized(final_row, len(df_signals)-1, 'End of Period')
        
        # محاسبه نتایج
        final_portfolio_value = portfolio_values[-1] if portfolio_values else self.initial_balance
        
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': final_portfolio_value,
            'total_return': (final_portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'total_trades': len(self.trades_history),
            'portfolio_values': portfolio_values,
            'trades_history': self.trades_history
        }
        
        return results
    
    def _manage_existing_position_optimized(self, row, current_index, current_timestamp):
        """
        مدیریت موقعیت موجود بهینه‌شده
        """
        if not self.position:
            return
            
        current_price = row['close']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        
        # محاسبه سود/ضرر فعلی
        if self.position['type'] == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # محاسبه مدت زمان (بر حسب کندل)
        time_in_position = current_index - entry_time
        
        # بررسی شرایط خروج (اولویت با Stop Loss)
        should_exit = False
        exit_reason = ''
        
        # Stop Loss (اولویت اول)
        if pnl_pct <= -self.stop_loss_pct:
            should_exit = True
            exit_reason = 'Stop Loss'
        
        # Take Profit (اولویت دوم)
        elif pnl_pct >= self.take_profit_pct:
            should_exit = True
            exit_reason = 'Take Profit'
        
        # Max Position Time (اولویت سوم)
        elif time_in_position >= self.max_position_time:
            should_exit = True
            exit_reason = 'Max Time'
        
        # Exit Signal (اولویت چهارم)
        elif row['exit_signal'] == 1:
            should_exit = True
            exit_reason = 'Exit Signal'
        
        if should_exit:
            self._close_position_optimized(row, current_index, exit_reason)
    
    def _open_position_optimized(self, row, index, timestamp, symbol):
        """
        باز کردن موقعیت جدید بهینه‌شده
        """
        current_price = row['close']
        position_value = self.current_balance * self.position_size_ratio
        
        # محاسبه کارمزد
        fee = position_value * self.trading_fee
        
        # محاسبه تعداد
        quantity = (position_value - fee) / current_price
        
        if quantity > 0:
            self.position = {
                'type': 'long',  # فعلاً فقط Long
                'entry_price': current_price,
                'quantity': quantity,
                'entry_time': index,
                'entry_timestamp': timestamp,
                'entry_fee': fee,
                'symbol': symbol
            }
            
            self.current_balance -= position_value
            self.total_positions += 1
            
            print(f"📈 Long opened at {current_price:.6f}, Qty: {quantity:.2f} (Quality Entry)")
    
    def _close_position_optimized(self, row, index, reason):
        """
        بستن موقعیت بهینه‌شده
        """
        if not self.position:
            return
            
        current_price = row['close']
        
        # محاسبه ارزش خروج
        exit_value = self.position['quantity'] * current_price
        exit_fee = exit_value * self.trading_fee
        net_exit_value = exit_value - exit_fee
        
        # محاسبه سود/ضرر
        total_fees = self.position['entry_fee'] + exit_fee
        net_pnl = net_exit_value - (self.position['quantity'] * self.position['entry_price'])
        pnl_pct = net_pnl / (self.position['quantity'] * self.position['entry_price']) * 100
        
        # ثبت ترید
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': index,
            'entry_price': self.position['entry_price'],
            'exit_price': current_price,
            'quantity': self.position['quantity'],
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'fees': total_fees,
            'reason': reason,
            'duration': index - self.position['entry_time']
        }
        
        self.trades_history.append(trade)
        
        # به‌روزرسانی بالانس
        self.current_balance += net_exit_value
        
        print(f"📉 Position closed at {current_price:.6f}, PnL: {net_pnl:.2f}$ ({pnl_pct:.2f}%), Reason: {reason}")
        
        # پاک کردن موقعیت
        self.position = None
        
        # Reset consecutive signals after trade completion
        self.consecutive_signals = 0
    
    def analyze_results_detailed(self, results):
        """
        تحلیل جامع نتایج Backtest
        """
        print("\n" + "="*70)
        print("📊 تحلیل جامع نتایج Backtest بهینه‌شده")
        print("="*70)
        
        # آمار کلی
        print(f"💰 بالانس اولیه: ${results['initial_balance']:,.2f}")
        print(f"💰 بالانس نهایی: ${results['final_balance']:,.2f}")
        
        total_return = results['total_return']
        if total_return > 0:
            print(f"📈 بازدهی کل: +{total_return:.2f}% 🎉")
        else:
            print(f"📉 بازدهی کل: {total_return:.2f}% ❌")
        
        print(f"🔄 تعداد کل تریدها: {results['total_trades']}")
        
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            
            # آمار تریدها
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            print(f"\n📊 آمار کیفیت تریدها:")
            print(f"✅ تریدهای سودآور: {len(winning_trades)} ({win_rate:.1f}%)")
            print(f"❌ تریدهای ضررده: {len(losing_trades)} ({100-win_rate:.1f}%)")
            print(f"💚 میانگین سود: ${avg_win:.2f}")
            print(f"💔 میانگین ضرر: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
                
                if profit_factor > 1.5:
                    print("   🟢 عالی!")
                elif profit_factor > 1.0:
                    print("   🟡 قابل قبول")
                else:
                    print("   🔴 نیاز به بهبود")
            
            # آمار مدت‌زمان
            avg_duration = trades_df['duration'].mean()
            max_duration = trades_df['duration'].max()
            min_duration = trades_df['duration'].min()
            
            print(f"\n⏱️ آمار زمانی:")
            print(f"   میانگین مدت ترید: {avg_duration:.1f} دقیقه")
            print(f"   حداکثر مدت ترید: {max_duration:.0f} دقیقه")
            print(f"   حداقل مدت ترید: {min_duration:.0f} دقیقه")
            
            # دلایل خروج
            print(f"\n🚪 دلایل خروج:")
            exit_reasons = trades_df['reason'].value_counts()
            for reason, count in exit_reasons.items():
                percentage = count/len(trades_df)*100
                print(f"   {reason}: {count} بار ({percentage:.1f}%)")
            
            # آمار سود/ضرر
            total_gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_gross_loss = losing_trades['pnl'].sum() if len(losing_trades) > 0 else 0
            net_profit = total_gross_profit + total_gross_loss
            
            print(f"\n💵 آمار مالی:")
            print(f"   کل سود ناخالص: ${total_gross_profit:.2f}")
            print(f"   کل ضرر ناخالص: ${total_gross_loss:.2f}")
            print(f"   سود خالص: ${net_profit:.2f}")
            
            # بهترین و بدترین ترید
            best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
            worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
            
            print(f"\n🏆 بهترین ترید: ${best_trade['pnl']:.2f} ({best_trade['pnl_pct']:.2f}%)")
            print(f"💔 بدترین ترید: ${worst_trade['pnl']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
        
        # توصیه‌های بهبود
        print(f"\n🔧 توصیه‌های بهبود:")
        if results['total_return'] < 0:
            print("   - افزایش threshold ها برای کاهش تریدهای ضعیف")
            print("   - بهبود فیلترهای کیفیت سیگنال")
            print("   - تنظیم مجدد Risk Management")
        elif results['total_trades'] < 10:
            print("   - کاهش threshold ها برای افزایش فرصت‌های ترید")
        else:
            print("   - تنظیمات فعلی مناسب است")
    
    def run_optimized_backtest(self, start_row=None, end_row=None):
        """
        اجرای کامل Backtest بهینه‌شده
        """
        print(f"🚀 شروع Backtest بهینه‌شده")
        print(f"💰 بالانس اولیه: ${self.initial_balance:,.2f}")
        print("🎯 تنظیمات محافظه‌کارانه فعال")
        
        # بارگذاری داده‌ها
        df = self.load_historical_data(start_row, end_row)
        if df is None:
            return None
        
        # آماده‌سازی داده‌ها
        df = self.prepare_data(df)
        
        # تولید سیگنال‌های بهینه‌شده
        df_signals = self.get_trading_signals_optimized(df)
        
        # اجرای backtest بهینه‌شده
        results = self.execute_backtest_optimized(df_signals, 'DOGE/USDT')
        
        # تحلیل جامع نتایج
        self.analyze_results_detailed(results)
        
        return results

if __name__ == "__main__":
    # اجرای Backtest بهینه‌شده
    backtester = OptimizedBacktester(
        model_path='best_improved_model.keras',
        model_info_path='model_info.json',
        initial_balance=1000.0
    )
    
    print("🎯 شروع Backtesting بهینه‌شده")
    print("🪙 ارز: DOGE/USDT (داده‌های تاریخی)")
    print("💰 بالانس اولیه: $1,000")
    print("🔧 تنظیمات محافظه‌کارانه و فیلترهای کیفیت")
    print("="*70)
    
    # اجرای backtest بر روی بازه‌ای از داده‌ها (آخرین 5000 رکورد)
    results = backtester.run_optimized_backtest(
        start_row=-5000,  # آخرین 5000 رکورد
        end_row=None
    )
    
    if results:
        print(f"\n🎉 Backtest بهینه‌شده کامل شد!")
        print(f"📈 بازدهی نهایی: {results['total_return']:+.2f}%")
        if results['total_return'] > 0:
            print("💰 سودآور بوده! 🎉")
        else:
            print("📉 هنوز نیاز به بهبود دارد")
        
        # نمایش جزئیات بیشتر
        print(f"🔄 تعداد تریدها: {results['total_trades']}")
        if results['total_trades'] > 0:
            profit = results['final_balance'] - results['initial_balance']
            print(f"💵 سود/ضرر مطلق: ${profit:+.2f}")
    else:
        print("❌ خطا در اجرای Backtest بهینه‌شده")



