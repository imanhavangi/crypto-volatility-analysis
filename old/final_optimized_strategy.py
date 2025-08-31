import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

class FinalOptimizedStrategy:
    """
    استراتژی نهایی بهینه‌شده برای مدل پیشرفته
    با تمرکز بر سودآوری و کنترل ریسک
    """
    
    def __init__(self, entry_model_path='advanced_entry_model.keras',
                 exit_model_path='advanced_exit_model.keras',
                 model_info_path='advanced_model_info.json',
                 initial_balance=1000.0):
        """
        استراتژی نهایی بهینه‌شده
        """
        print("🚀 بارگذاری استراتژی نهایی...")
        
        # بارگذاری مدل‌ها
        self.entry_model = tf.keras.models.load_model(entry_model_path)
        self.exit_model = tf.keras.models.load_model(exit_model_path)
        
        # بارگذاری اطلاعات مدل
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # تنظیمات مالی
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_fee = 0.0008
        
        # 🎯 تنظیمات ترید بهینه‌شده (برای شما که تعداد زیاد ولی با کیفیت می‌خواهید)
        self.position_size_ratio = 0.08  # 8% بالانس (کاهش ریسک)
        
        # Threshold های بهینه بر اساس نتایج قبلی
        self.entry_threshold = 0.45   # متعادل بین کمیت و کیفیت
        self.exit_threshold = 0.55    # کمی محافظه‌کارانه‌تر
        
        # 🛡️ Risk Management پیشرفته
        self.stop_loss_pct = 0.015    # 1.5% stop-loss (سخت‌تر)
        self.take_profit_pct = 0.025  # 2.5% take-profit (واقعی‌تر)
        self.trailing_stop_pct = 0.008 # 0.8% trailing stop
        self.max_position_time = 35   # 35 دقیقه
        self.min_volume_filter = 800000
        
        # 📊 فیلترهای کیفیت پیشرفته
        self.min_confidence_diff = 0.05  # حداقل اختلاف احتمال
        self.trend_confirmation = True   # تأیید ترند
        self.volume_confirmation = True  # تأیید volume
        self.momentum_filter = True      # فیلتر momentum
        
        # Scalers
        self.entry_scaler = RobustScaler()
        self.exit_scaler = RobustScaler()
        self.feature_columns = self.model_info['feature_columns']
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        self.total_positions = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # حداکثر ضرر متوالی
        
        print("✅ استراتژی نهایی آماده است!")
        print(f"🎯 Target: تعداد زیاد ترید با کیفیت بالا")
    
    def enhanced_feature_engineering(self, df):
        """
        Feature engineering مطابق با مدل آموزش‌دیده
        """
        df_enhanced = df.copy()
        
        # Technical Indicators اضافی
        df_enhanced['rsi_divergence'] = df_enhanced['rsi'].diff()
        df_enhanced['macd_momentum'] = df_enhanced['macd'] - df_enhanced['macd'].shift(1)
        df_enhanced['volume_surge'] = df_enhanced['volume'] / df_enhanced['volume'].rolling(20).mean()
        
        # Price action patterns
        df_enhanced['price_momentum'] = df_enhanced['close'].pct_change(5)
        df_enhanced['volatility_regime'] = (df_enhanced['volatility_5m'] > df_enhanced['volatility_5m'].rolling(50).quantile(0.8)).astype(int)
        
        # Market microstructure
        df_enhanced['spread_proxy'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close']
        df_enhanced['volume_price_trend'] = df_enhanced['volume'] * df_enhanced['price_change_1m']
        
        # Advanced momentum
        df_enhanced['momentum_5'] = df_enhanced['close'] / df_enhanced['close'].shift(5) - 1
        df_enhanced['momentum_15'] = df_enhanced['close'] / df_enhanced['close'].shift(15) - 1
        df_enhanced['momentum_consistency'] = (df_enhanced['momentum_5'] * df_enhanced['momentum_15'] > 0).astype(int)
        
        # Regime detection
        df_enhanced['trend_regime'] = (df_enhanced['close'] > df_enhanced['close'].rolling(20).mean()).astype(int)
        df_enhanced['volatility_normalized'] = df_enhanced['volatility_5m'] / df_enhanced['volatility_5m'].rolling(100).mean()
        
        return df_enhanced
    
    def apply_quality_filters(self, df_signals):
        """
        اعمال فیلترهای کیفیت پیشرفته
        """
        print("🔍 اعمال فیلترهای کیفیت...")
        
        filtered_signals = df_signals.copy()
        
        # فیلتر 1: اختلاف احتمال
        confidence_diff = abs(filtered_signals['entry_prob'] - filtered_signals['exit_prob'])
        confidence_filter = confidence_diff > self.min_confidence_diff
        
        # فیلتر 2: تأیید ترند
        if self.trend_confirmation:
            trend_filter = (
                (filtered_signals['trend_regime'] == 1) &  # Uptrend
                (filtered_signals['momentum_consistency'] == 1)  # Consistent momentum
            )
        else:
            trend_filter = pd.Series(True, index=filtered_signals.index)
        
        # فیلتر 3: تأیید volume
        if self.volume_confirmation:
            volume_filter = filtered_signals['volume_surge'] > 1.2  # 20% بالاتر از میانگین
        else:
            volume_filter = pd.Series(True, index=filtered_signals.index)
        
        # فیلتر 4: momentum
        if self.momentum_filter:
            momentum_filter = (
                (filtered_signals['price_momentum'] > 0) &  # Positive momentum
                (filtered_signals['macd_momentum'] > 0)    # MACD improving
            )
        else:
            momentum_filter = pd.Series(True, index=filtered_signals.index)
        
        # اعمال همه فیلترها
        all_filters = confidence_filter & trend_filter & volume_filter & momentum_filter
        
        # فیلتر کردن سیگنال‌ها
        filtered_signals.loc[~all_filters, 'entry_signal'] = 0
        
        original_count = df_signals['entry_signal'].sum()
        filtered_count = filtered_signals['entry_signal'].sum()
        
        print(f"📊 سیگنال‌های اصلی: {original_count}")
        print(f"📊 سیگنال‌های فیلتر شده: {filtered_count}")
        print(f"📊 فیلتر شده: {((original_count - filtered_count) / original_count * 100):.1f}%")
        
        return filtered_signals
    
    def load_and_predict(self, start_row=-2000, end_row=None):
        """
        بارگذاری داده‌ها و پیش‌بینی
        """
        print(f"📥 بارگذاری و پیش‌بینی...")
        
        # بارگذاری داده‌ها
        df = pd.read_csv('training_data.csv')
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # انتخاب بازه
        if start_row is not None and end_row is not None:
            df = df.iloc[start_row:end_row]
        elif start_row is not None:
            df = df.iloc[start_row:]
        elif end_row is not None:
            df = df.iloc[:end_row]
        
        # Enhanced feature engineering
        df_enhanced = self.enhanced_feature_engineering(df)
        df_clean = df_enhanced.dropna()
        
        # آماده‌سازی features
        feature_data = df_clean[self.feature_columns].copy()
        
        # Scaling
        if not hasattr(self.entry_scaler, 'scale_'):
            self.entry_scaler.fit(feature_data)
        entry_features = self.entry_scaler.transform(feature_data)
        
        if not hasattr(self.exit_scaler, 'scale_'):
            self.exit_scaler.fit(feature_data)
        exit_features = self.exit_scaler.transform(feature_data)
        
        # پیش‌بینی
        entry_probs = self.entry_model.predict(entry_features, verbose=0).flatten()
        exit_probs = self.exit_model.predict(exit_features, verbose=0).flatten()
        
        # اعمال threshold ها
        entry_signals = (entry_probs > self.entry_threshold).astype(int)
        exit_signals = (exit_probs > self.exit_threshold).astype(int)
        
        # اضافه کردن پیش‌بینی‌ها
        df_signals = df_clean.copy()
        df_signals['entry_prob'] = entry_probs
        df_signals['exit_prob'] = exit_probs
        df_signals['entry_signal'] = entry_signals
        df_signals['exit_signal'] = exit_signals
        
        # اعمال فیلترهای کیفیت
        df_final = self.apply_quality_filters(df_signals)
        
        print(f"✅ {len(df_final)} رکورد آماده شد")
        
        return df_final
    
    def should_enter_trade(self, row, current_index):
        """
        بررسی شرایط ورود با منطق پیشرفته
        """
        # شرط اصلی
        if row['entry_signal'] != 1:
            return False
        
        # Volume filter
        if row['volume'] < self.min_volume_filter:
            return False
        
        # جلوگیری از ترید بعد از ضررهای متوالی
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        
        # اطمینان بالا برای entry
        if row['entry_prob'] < (self.entry_threshold + 0.1):
            return False
        
        return True
    
    def execute_optimized_backtest(self, df_signals):
        """
        اجرای Backtest بهینه‌شده
        """
        print("🔄 اجرای Backtest بهینه‌شده...")
        
        self.current_balance = self.initial_balance
        self.position = None
        self.trades_history = []
        self.total_positions = 0
        self.consecutive_losses = 0
        
        portfolio_values = []
        candle_index = 0
        
        for timestamp, row in df_signals.iterrows():
            current_price = row['close']
            
            # محاسبه ارزش فعلی پورتفولیو
            portfolio_value = self.current_balance
            if self.position:
                portfolio_value += self.position['quantity'] * current_price
            portfolio_values.append(portfolio_value)
            
            # مدیریت موقعیت موجود
            if self.position:
                self._manage_position_advanced(row, candle_index, timestamp)
            
            # بررسی سیگنال ورود جدید
            if not self.position and self.should_enter_trade(row, candle_index):
                self._open_position_advanced(row, candle_index, timestamp)
            
            candle_index += 1
        
        # بستن موقعیت باقی‌مانده
        if self.position:
            final_row = df_signals.iloc[-1]
            self._close_position_advanced(final_row, len(df_signals)-1, 'End of Period')
        
        # محاسبه نتایج
        final_portfolio_value = portfolio_values[-1] if portfolio_values else self.initial_balance
        
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': final_portfolio_value,
            'total_return': (final_portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'total_trades': len(self.trades_history),
            'portfolio_values': portfolio_values,
            'trades_history': self.trades_history.copy()
        }
        
        return results
    
    def _manage_position_advanced(self, row, current_index, current_timestamp):
        """
        مدیریت موقعیت با Trailing Stop
        """
        if not self.position:
            return
            
        current_price = row['close']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        
        # محاسبه سود/ضرر فعلی
        pnl_pct = (current_price - entry_price) / entry_price
        
        # به‌روزرسانی highest price برای trailing stop
        if 'highest_price' not in self.position:
            self.position['highest_price'] = current_price
        elif current_price > self.position['highest_price']:
            self.position['highest_price'] = current_price
        
        # محاسبه trailing stop
        trailing_stop_price = self.position['highest_price'] * (1 - self.trailing_stop_pct)
        
        # محاسبه مدت زمان
        time_in_position = current_index - entry_time
        
        # بررسی شرایط خروج
        should_exit = False
        exit_reason = ''
        
        # Stop Loss (اولویت اول)
        if pnl_pct <= -self.stop_loss_pct:
            should_exit = True
            exit_reason = 'Stop Loss'
        
        # Trailing Stop
        elif current_price <= trailing_stop_price and pnl_pct > 0.005:  # فقط اگر کمی سود داریم
            should_exit = True
            exit_reason = 'Trailing Stop'
        
        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            should_exit = True
            exit_reason = 'Take Profit'
        
        # Max Position Time
        elif time_in_position >= self.max_position_time:
            should_exit = True
            exit_reason = 'Max Time'
        
        # Exit Signal
        elif row['exit_signal'] == 1 and pnl_pct > -0.005:  # فقط اگر ضرر زیاد نداریم
            should_exit = True
            exit_reason = 'Exit Signal'
        
        if should_exit:
            self._close_position_advanced(row, current_index, exit_reason)
    
    def _open_position_advanced(self, row, index, timestamp):
        """
        باز کردن موقعیت با size dynamic
        """
        current_price = row['close']
        
        # اندازه موقعیت بر اساس consecutive losses
        size_multiplier = max(0.5, 1 - (self.consecutive_losses * 0.2))
        position_value = self.current_balance * self.position_size_ratio * size_multiplier
        
        # محاسبه کارمزد
        fee = position_value * self.trading_fee
        
        # محاسبه تعداد
        quantity = (position_value - fee) / current_price
        
        if quantity > 0:
            self.position = {
                'type': 'long',
                'entry_price': current_price,
                'quantity': quantity,
                'entry_time': index,
                'entry_timestamp': timestamp,
                'entry_fee': fee,
                'highest_price': current_price,
                'symbol': 'DOGE/USDT'
            }
            
            self.current_balance -= position_value
            self.total_positions += 1
            
            print(f"📈 Position opened: ${current_price:.6f}, Size: {size_multiplier:.1f}x")
    
    def _close_position_advanced(self, row, index, reason):
        """
        بستن موقعیت با tracking ضررهای متوالی
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
        
        # به‌روزرسانی consecutive losses
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
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
        
        print(f"📉 Position closed: ${current_price:.6f}, PnL: {net_pnl:.2f}$ ({pnl_pct:.2f}%), {reason}")
        
        # پاک کردن موقعیت
        self.position = None
    
    def analyze_final_results(self, results):
        """
        تحلیل نهایی نتایج
        """
        print("\n" + "="*70)
        print("📊 نتایج استراتژی نهایی بهینه‌شده")
        print("="*70)
        
        print(f"💰 بالانس اولیه: ${results['initial_balance']:,.2f}")
        print(f"💰 بالانس نهایی: ${results['final_balance']:,.2f}")
        
        total_return = results['total_return']
        profit = results['final_balance'] - results['initial_balance']
        
        if total_return > 0:
            print(f"📈 بازدهی: +{total_return:.2f}% 🎉")
            print(f"💵 سود مطلق: +${profit:.2f}")
        else:
            print(f"📉 بازدهی: {total_return:.2f}%")
            print(f"💵 ضرر مطلق: ${profit:.2f}")
        
        print(f"🔄 تعداد تریدها: {results['total_trades']}")
        
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            print(f"\n📊 آمار کیفیت:")
            print(f"✅ Win Rate: {win_rate:.1f}%")
            print(f"💚 میانگین سود: ${avg_win:.2f}")
            print(f"💔 میانگین ضرر: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
                
                if profit_factor > 1.2:
                    print("   🟢 عالی!")
                elif profit_factor > 1.0:
                    print("   🟡 قابل قبول")
                else:
                    print("   🔴 نیاز به بهبود")
            
            # آمار زمانی
            avg_duration = trades_df['duration'].mean()
            print(f"\n⏱️ میانگین مدت ترید: {avg_duration:.1f} دقیقه")
            
            # دلایل خروج
            print(f"\n🚪 دلایل خروج:")
            exit_reasons = trades_df['reason'].value_counts()
            for reason, count in exit_reasons.items():
                print(f"   {reason}: {count} ({count/len(trades_df)*100:.1f}%)")
            
            # بهترین و بدترین
            best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
            worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
            
            print(f"\n🏆 بهترین ترید: ${best_trade['pnl']:.2f} ({best_trade['pnl_pct']:.2f}%)")
            print(f"💔 بدترین ترید: ${worst_trade['pnl']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
        
        return results
    
    def run_final_test(self, start_row=-2000, end_row=None):
        """
        اجرای تست نهایی
        """
        print("🚀 اجرای استراتژی نهایی بهینه‌شده")
        print("🎯 هدف: تعداد زیاد ترید با کیفیت بالا و سودآوری")
        print("="*70)
        
        # بارگذاری و پیش‌بینی
        df_signals = self.load_and_predict(start_row, end_row)
        
        # اجرای backtest
        results = self.execute_optimized_backtest(df_signals)
        
        # تحلیل نتایج
        final_results = self.analyze_final_results(results)
        
        return final_results

if __name__ == "__main__":
    # اجرای استراتژی نهایی
    strategy = FinalOptimizedStrategy(initial_balance=1000.0)
    
    print("🎯 استراتژی نهایی برای شما")
    print("💡 ویژگی‌ها:")
    print("   - تعداد زیاد ترید ولی با کیفیت")
    print("   - فیلترهای پیشرفته کیفیت")
    print("   - Trailing Stop و Risk Management")
    print("   - Dynamic Position Sizing")
    print("   - Consecutive Loss Protection")
    
    results = strategy.run_final_test(start_row=-2000, end_row=None)
    
    print(f"\n🎉 تست نهایی کامل شد!")
    if results['total_return'] > 0:
        print("💰 استراتژی سودآور است! 🎉")
    else:
        print("📈 نیاز به تنظیم بیشتر دارد")
    
    print(f"🔄 تعداد تریدها: {results['total_trades']}")
    print(f"📈 بازدهی: {results['total_return']:+.2f}%")



