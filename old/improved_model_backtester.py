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

class ImprovedModelBacktester:
    """
    سیستم Backtesting برای مدل بهبود یافته
    با threshold های بهینه و feature engineering دقیق
    """
    
    def __init__(self, model_path='best_improved_model.keras', 
                 model_info_path='model_info.json', 
                 initial_balance=1000.0):
        """
        مقداردهی سیستم Backtesting
        
        Args:
            model_path: مسیر مدل آموزش‌دیده
            model_info_path: مسیر اطلاعات مدل (thresholds, features)
            initial_balance: بالانس اولیه (دلار)
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
        
        # تنظیمات ترید بر اساس مدل بهبود یافته
        self.position_size_ratio = 0.20  # 20% بالانس در هر معامله
        self.entry_threshold = self.model_info['entry_threshold']  # 0.05
        self.exit_threshold = self.model_info['exit_threshold']    # 0.6
        
        print(f"📊 Entry Threshold: {self.entry_threshold}")
        print(f"📊 Exit Threshold: {self.exit_threshold}")
        
        # Risk Management بهبود یافته
        self.stop_loss_pct = 0.025    # 2.5% stop-loss
        self.take_profit_pct = 0.06   # 6% take-profit (بالاتر برای crypto)
        self.max_position_time = 90   # حداکثر 90 دقیقه در هر موقعیت
        self.min_volume_filter = 1000000  # حداقل volume برای ترید
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        self.total_positions = 0
        
        # ایجاد Scaler مناسب
        self.scaler = StandardScaler()
        self.feature_columns = self.model_info['feature_columns']
        
        print("✅ سیستم آماده است!")
        print(f"🎯 Features: {len(self.feature_columns)} فیچر")
    
    def load_historical_data(self, start_row=None, end_row=None):
        """
        بارگذاری داده‌های تاریخی از فایل CSV
        
        Args:
            start_row: ردیف شروع (اختیاری)
            end_row: ردیف پایان (اختیاری)
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
    
    def get_trading_signals(self, df):
        """
        تولید سیگنال‌های ترید از مدل
        """
        print("🎯 تولید سیگنال‌های ترید...")
        
        # آماده‌سازی فیچرها
        features = self.prepare_features(df)
        
        # پیش‌بینی مدل
        predictions = self.model.predict(features, verbose=0)
        entry_probs = predictions[0].flatten()
        exit_probs = predictions[1].flatten()
        
        # اعمال threshold ها
        entry_signals = (entry_probs > self.entry_threshold).astype(int)
        exit_signals = (exit_probs > self.exit_threshold).astype(int)
        
        # اضافه کردن به DataFrame
        df_signals = df.copy()
        df_signals['entry_prob'] = entry_probs
        df_signals['exit_prob'] = exit_probs
        df_signals['entry_signal'] = entry_signals
        df_signals['exit_signal'] = exit_signals
        
        print(f"📊 Entry signals: {entry_signals.sum()}")
        print(f"📊 Exit signals: {exit_signals.sum()}")
        
        return df_signals
    
    def execute_backtest(self, df_signals, symbol='DOGE/USDT'):
        """
        اجرای Backtest
        """
        print("🔄 اجرای Backtest...")
        
        self.current_balance = self.initial_balance
        self.position = None
        self.trades_history = []
        self.total_positions = 0
        
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
            
            # بررسی Volume Filter
            if current_volume < self.min_volume_filter:
                candle_index += 1
                continue
            
            # مدیریت موقعیت موجود
            if self.position:
                self._manage_existing_position(row, candle_index, timestamp)
            
            # بررسی سیگنال ورود جدید
            if not self.position and row['entry_signal'] == 1:
                self._open_position(row, candle_index, timestamp, symbol)
            
            candle_index += 1
        
        # بستن موقعیت باقی‌مانده
        if self.position:
            final_row = df_signals.iloc[-1]
            self._close_position(final_row, len(df_signals)-1, 'End of Period')
        
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
    
    def _manage_existing_position(self, row, current_index, current_timestamp):
        """
        مدیریت موقعیت موجود
        """
        if not self.position:
            return
            
        current_price = row['close']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        entry_timestamp = self.position['entry_timestamp']
        
        # محاسبه سود/ضرر فعلی
        if self.position['type'] == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # محاسبه مدت زمان (بر حسب کندل)
        time_in_position = current_index - entry_time
        
        # بررسی شرایط خروج
        should_exit = False
        exit_reason = ''
        
        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            should_exit = True
            exit_reason = 'Stop Loss'
        
        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            should_exit = True
            exit_reason = 'Take Profit'
        
        # Max Position Time (بر حسب کندل)
        elif time_in_position >= self.max_position_time:
            should_exit = True
            exit_reason = 'Max Time'
        
        # Exit Signal
        elif row['exit_signal'] == 1:
            should_exit = True
            exit_reason = 'Exit Signal'
        
        if should_exit:
            self._close_position(row, current_index, exit_reason)
    
    def _open_position(self, row, index, timestamp, symbol):
        """
        باز کردن موقعیت جدید
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
                'entry_time': index,  # index کندل
                'entry_timestamp': timestamp,  # زمان واقعی
                'entry_fee': fee,
                'symbol': symbol
            }
            
            self.current_balance -= position_value
            self.total_positions += 1
            
            print(f"📈 Long opened at {current_price:.6f}, Qty: {quantity:.2f}")
    
    def _close_position(self, row, index, reason):
        """
        بستن موقعیت
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
    
    def analyze_results(self, results):
        """
        تحلیل نتایج Backtest
        """
        print("\n" + "="*60)
        print("📊 تحلیل نتایج Backtest")
        print("="*60)
        
        # آمار کلی
        print(f"💰 بالانس اولیه: ${results['initial_balance']:,.2f}")
        print(f"💰 بالانس نهایی: ${results['final_balance']:,.2f}")
        print(f"📈 بازدهی کل: {results['total_return']:+.2f}%")
        print(f"🔄 تعداد کل تریدها: {results['total_trades']}")
        
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            
            # آمار تریدها
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            print(f"\n📊 آمار تریدها:")
            print(f"✅ تریدهای سودآور: {len(winning_trades)} ({win_rate:.1f}%)")
            print(f"❌ تریدهای ضررده: {len(losing_trades)} ({100-win_rate:.1f}%)")
            print(f"💚 میانگین سود: ${avg_win:.2f}")
            print(f"💔 میانگین ضرر: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
            
            # آمار مدت‌زمان
            avg_duration = trades_df['duration'].mean()
            print(f"⏱️ میانگین مدت ترید: {avg_duration:.1f} دقیقه")
            
            # دلایل خروج
            print(f"\n🚪 دلایل خروج:")
            exit_reasons = trades_df['reason'].value_counts()
            for reason, count in exit_reasons.items():
                print(f"   {reason}: {count} بار ({count/len(trades_df)*100:.1f}%)")
    
    def plot_results(self, results, df_signals):
        """
        رسم نمودارهای نتایج
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # نمودار قیمت و سیگنال‌ها
        ax1.plot(df_signals.index, df_signals['close'], label='Price', alpha=0.7)
        
        # نمایش نقاط ورود و خروج
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            entry_times = [df_signals.index[t] for t in trades_df['entry_time']]
            exit_times = [df_signals.index[t] for t in trades_df['exit_time']]
            entry_prices = trades_df['entry_price'].values
            exit_prices = trades_df['exit_price'].values
            
            ax1.scatter(entry_times, entry_prices, color='green', marker='^', s=100, label='Entry', alpha=0.8)
            ax1.scatter(exit_times, exit_prices, color='red', marker='v', s=100, label='Exit', alpha=0.8)
        
        ax1.set_title('قیمت و نقاط ورود/خروج')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # نمودار ارزش پورتفولیو
        ax2.plot(results['portfolio_values'])
        ax2.axhline(y=results['initial_balance'], color='r', linestyle='--', alpha=0.5, label='Initial Balance')
        ax2.set_title(f'ارزش پورتفولیو (بازدهی: {results["total_return"]:+.2f}%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # هیستوگرام سود/ضرر
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            ax3.hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax3.set_title('توزیع سود/ضرر تریدها')
            ax3.set_xlabel('سود/ضرر ($)')
            ax3.grid(True, alpha=0.3)
        
        # نمودار احتمال‌های مدل
        ax4.plot(df_signals.index, df_signals['entry_prob'], label='Entry Probability', alpha=0.7)
        ax4.plot(df_signals.index, df_signals['exit_prob'], label='Exit Probability', alpha=0.7)
        ax4.axhline(y=self.entry_threshold, color='g', linestyle='--', alpha=0.5, label=f'Entry Threshold ({self.entry_threshold})')
        ax4.axhline(y=self.exit_threshold, color='r', linestyle='--', alpha=0.5, label=f'Exit Threshold ({self.exit_threshold})')
        ax4.set_title('احتمال‌های پیش‌بینی مدل')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_model_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_backtest(self, start_row=None, end_row=None):
        """
        اجرای کامل Backtest
        
        Args:
            start_row: ردیف شروع داده‌ها (اختیاری)
            end_row: ردیف پایان داده‌ها (اختیاری)
        """
        print(f"🚀 شروع Backtest کامل")
        print(f"💰 بالانس اولیه: ${self.initial_balance:,.2f}")
        
        # بارگذاری داده‌ها
        df = self.load_historical_data(start_row, end_row)
        if df is None:
            return None
        
        # آماده‌سازی داده‌ها
        df = self.prepare_data(df)
        
        # تولید سیگنال‌ها
        df_signals = self.get_trading_signals(df)
        
        # اجرای backtest
        results = self.execute_backtest(df_signals, 'DOGE/USDT')
        
        # تحلیل نتایج
        self.analyze_results(results)
        
        # رسم نمودارها
        self.plot_results(results, df_signals)
        
        return results

if __name__ == "__main__":
    # اجرای Backtest
    backtester = ImprovedModelBacktester(
        model_path='best_improved_model.keras',
        model_info_path='model_info.json',
        initial_balance=1000.0
    )
    
    print("🎯 شروع Backtesting مدل بهبود یافته")
    print("🪙 ارز: DOGE/USDT (داده‌های تاریخی)")
    print("💰 بالانس اولیه: $1,000")
    print("="*60)
    
    # اجرای backtest بر روی بازه‌ای از داده‌ها (آخرین 5000 رکورد)
    results = backtester.run_full_backtest(
        start_row=-5000,  # آخرین 5000 رکورد
        end_row=None
    )
    
    if results:
        print(f"\n🎉 Backtest کامل شد!")
        print(f"📈 بازدهی نهایی: {results['total_return']:+.2f}%")
        if results['total_return'] > 0:
            print("💰 سودآور بوده!")
        else:
            print("📉 ضررده بوده!")
        
        # نمایش جزئیات بیشتر
        print(f"🔄 تعداد تریدها: {results['total_trades']}")
        if results['total_trades'] > 0:
            profit = results['final_balance'] - results['initial_balance']
            print(f"💵 سود/ضرر مطلق: ${profit:+.2f}")
    else:
        print("❌ خطا در اجرای Backtest")
