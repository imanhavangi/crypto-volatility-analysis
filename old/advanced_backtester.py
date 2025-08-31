import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktester:
    """
    سیستم Backtesting برای مدل پیشرفته ضد overfitting
    """
    
    def __init__(self, entry_model_path='advanced_entry_model.keras',
                 exit_model_path='advanced_exit_model.keras',
                 model_info_path='advanced_model_info.json',
                 initial_balance=1000.0):
        """
        مقداردهی سیستم Backtesting پیشرفته
        """
        print("🚀 بارگذاری مدل‌های پیشرفته...")
        
        # بارگذاری مدل‌ها
        self.entry_model = tf.keras.models.load_model(entry_model_path)
        self.exit_model = tf.keras.models.load_model(exit_model_path)
        
        # بارگذاری اطلاعات مدل
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # تنظیمات مالی
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_fee = 0.0008  # 0.08% کارمزد
        
        # تنظیمات ترید - برای تعداد زیاد ترید با کیفیت
        self.position_size_ratio = 0.10  # 10% بالانس در هر معامله
        
        # Threshold های مختلف برای تست
        self.test_thresholds = {
            'conservative': {'entry': 0.7, 'exit': 0.7},
            'balanced': {'entry': 0.6, 'exit': 0.6},
            'aggressive': {'entry': 0.5, 'exit': 0.5},
            'very_aggressive': {'entry': 0.4, 'exit': 0.4},
            'ultra_aggressive': {'entry': 0.3, 'exit': 0.3}
        }
        
        # Risk Management
        self.stop_loss_pct = 0.02     # 2% stop-loss
        self.take_profit_pct = 0.04   # 4% take-profit
        self.max_position_time = 45   # حداکثر 45 دقیقه
        self.min_volume_filter = 1000000
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        self.total_positions = 0
        
        # Scalers
        self.entry_scaler = RobustScaler()
        self.exit_scaler = RobustScaler()
        
        self.feature_columns = self.model_info['feature_columns']
        
        print("✅ سیستم پیشرفته آماده است!")
        print(f"🎯 Features: {len(self.feature_columns)} فیچر پیشرفته")
    
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
    
    def load_historical_data(self, start_row=None, end_row=None):
        """
        بارگذاری و آماده‌سازی داده‌های تاریخی
        """
        print(f"📥 بارگذاری داده‌های تاریخی...")
        
        # بارگذاری داده‌ها
        df = pd.read_csv('training_data.csv')
        
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
        
        # Enhanced feature engineering
        df_enhanced = self.enhanced_feature_engineering(df)
        
        # حذف NaN ها
        df_clean = df_enhanced.dropna()
        
        print(f"✅ {len(df_clean)} رکورد آماده شد")
        if hasattr(df_clean.index, 'min') and hasattr(df_clean.index, 'max'):
            print(f"📅 از {df_clean.index.min()} تا {df_clean.index.max()}")
        
        return df_clean
    
    def get_predictions(self, df, entry_threshold, exit_threshold):
        """
        دریافت پیش‌بینی‌های مدل
        """
        # آماده‌سازی features
        feature_data = df[self.feature_columns].copy()
        
        # Scaling برای entry
        if not hasattr(self.entry_scaler, 'scale_'):
            self.entry_scaler.fit(feature_data)
        entry_features = self.entry_scaler.transform(feature_data)
        
        # Scaling برای exit
        if not hasattr(self.exit_scaler, 'scale_'):
            self.exit_scaler.fit(feature_data)
        exit_features = self.exit_scaler.transform(feature_data)
        
        # پیش‌بینی
        entry_probs = self.entry_model.predict(entry_features, verbose=0).flatten()
        exit_probs = self.exit_model.predict(exit_features, verbose=0).flatten()
        
        # اعمال threshold ها
        entry_signals = (entry_probs > entry_threshold).astype(int)
        exit_signals = (exit_probs > exit_threshold).astype(int)
        
        # اضافه کردن به DataFrame
        df_signals = df.copy()
        df_signals['entry_prob'] = entry_probs
        df_signals['exit_prob'] = exit_probs
        df_signals['entry_signal'] = entry_signals
        df_signals['exit_signal'] = exit_signals
        
        return df_signals
    
    def execute_backtest(self, df_signals, strategy_name='balanced'):
        """
        اجرای Backtest
        """
        print(f"🔄 اجرای Backtest استراتژی {strategy_name}...")
        
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
                self._open_position(row, candle_index, timestamp, 'DOGE/USDT')
            
            candle_index += 1
        
        # بستن موقعیت باقی‌مانده
        if self.position:
            final_row = df_signals.iloc[-1]
            self._close_position(final_row, len(df_signals)-1, 'End of Period')
        
        # محاسبه نتایج
        final_portfolio_value = portfolio_values[-1] if portfolio_values else self.initial_balance
        
        results = {
            'strategy': strategy_name,
            'initial_balance': self.initial_balance,
            'final_balance': final_portfolio_value,
            'total_return': (final_portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'total_trades': len(self.trades_history),
            'portfolio_values': portfolio_values,
            'trades_history': self.trades_history.copy()
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
        
        # محاسبه سود/ضرر فعلی
        pnl_pct = (current_price - entry_price) / entry_price
        
        # محاسبه مدت زمان
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
        
        # Max Position Time
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
                'type': 'long',
                'entry_price': current_price,
                'quantity': quantity,
                'entry_time': index,
                'entry_timestamp': timestamp,
                'entry_fee': fee,
                'symbol': symbol
            }
            
            self.current_balance -= position_value
            self.total_positions += 1
    
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
        
        # پاک کردن موقعیت
        self.position = None
    
    def analyze_strategy_results(self, results):
        """
        تحلیل نتایج استراتژی
        """
        print(f"\n📊 نتایج استراتژی {results['strategy']}:")
        print("="*50)
        
        print(f"💰 بالانس اولیه: ${results['initial_balance']:,.2f}")
        print(f"💰 بالانس نهایی: ${results['final_balance']:,.2f}")
        
        total_return = results['total_return']
        if total_return > 0:
            print(f"📈 بازدهی: +{total_return:.2f}% ✅")
        else:
            print(f"📉 بازدهی: {total_return:.2f}% ❌")
        
        print(f"🔄 تعداد تریدها: {results['total_trades']}")
        
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            print(f"✅ Win Rate: {win_rate:.1f}%")
            print(f"💚 میانگین سود: ${avg_win:.2f}")
            print(f"💔 میانگین ضرر: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
            
            # دلایل خروج
            exit_reasons = trades_df['reason'].value_counts()
            print(f"\n🚪 دلایل خروج:")
            for reason, count in exit_reasons.items():
                print(f"   {reason}: {count} ({count/len(trades_df)*100:.1f}%)")
        
        return results
    
    def test_all_strategies(self, start_row=-3000, end_row=None):
        """
        تست تمام استراتژی‌ها
        """
        print("🚀 تست همه استراتژی‌ها با مدل پیشرفته")
        print("="*60)
        
        # بارگذاری داده‌ها
        df = self.load_historical_data(start_row, end_row)
        
        all_results = {}
        
        for strategy_name, thresholds in self.test_thresholds.items():
            print(f"\n🎯 تست استراتژی {strategy_name}")
            print(f"   Entry Threshold: {thresholds['entry']}")
            print(f"   Exit Threshold: {thresholds['exit']}")
            
            # دریافت سیگنال‌ها
            df_signals = self.get_predictions(df, thresholds['entry'], thresholds['exit'])
            
            print(f"   📊 Entry signals: {df_signals['entry_signal'].sum()}")
            print(f"   📊 Exit signals: {df_signals['exit_signal'].sum()}")
            
            # اجرای backtest
            results = self.execute_backtest(df_signals, strategy_name)
            
            # تحلیل نتایج
            results = self.analyze_strategy_results(results)
            
            all_results[strategy_name] = results
        
        # مقایسه نهایی
        self.compare_strategies(all_results)
        
        return all_results
    
    def compare_strategies(self, all_results):
        """
        مقایسه تمام استراتژی‌ها
        """
        print("\n" + "="*70)
        print("📊 مقایسه همه استراتژی‌ها")
        print("="*70)
        
        comparison_data = []
        for strategy_name, results in all_results.items():
            trades_df = pd.DataFrame(results['trades_history']) if results['trades_history'] else pd.DataFrame()
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Return (%)': results['total_return'],
                'Trades': results['total_trades'],
                'Win Rate (%)': win_rate,
                'Final Balance ($)': results['final_balance']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Return (%)', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # بهترین استراتژی
        best_strategy = comparison_df.iloc[0]
        print(f"\n🏆 بهترین استراتژی: {best_strategy['Strategy']}")
        print(f"   📈 بازدهی: {best_strategy['Return (%)']:.2f}%")
        print(f"   🔄 تعداد تریدها: {best_strategy['Trades']:.0f}")
        print(f"   ✅ Win Rate: {best_strategy['Win Rate (%)']:.1f}%")

if __name__ == "__main__":
    # تست مدل پیشرفته
    backtester = AdvancedBacktester(initial_balance=1000.0)
    
    print("🎯 تست کامل مدل پیشرفته ضد Overfitting")
    print("💰 بالانس اولیه: $1,000")
    print("🪙 ارز: DOGE/USDT")
    print("🔧 استراتژی‌های مختلف threshold")
    
    # تست همه استراتژی‌ها
    all_results = backtester.test_all_strategies(start_row=-3000, end_row=None)
    
    print(f"\n🎉 تست کامل شد!")
    print("💡 حالا می‌توانید بهترین threshold را انتخاب کنید")



