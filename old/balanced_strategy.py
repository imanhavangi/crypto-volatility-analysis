import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import json
import warnings
warnings.filterwarnings('ignore')

class BalancedStrategy:
    """
    استراتژی متعادل: تعداد زیاد ولی با کیفیت مناسب
    """
    
    def __init__(self, entry_model_path='advanced_entry_model.keras',
                 exit_model_path='advanced_exit_model.keras',
                 model_info_path='advanced_model_info.json',
                 initial_balance=1000.0):
        
        print("🎯 استراتژی متعادل برای شما")
        
        # بارگذاری مدل‌ها
        self.entry_model = tf.keras.models.load_model(entry_model_path)
        self.exit_model = tf.keras.models.load_model(exit_model_path)
        
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # تنظیمات مالی
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_fee = 0.0008
        
        # 🎯 تنظیمات متعادل برای تعداد زیاد ولی باکیفیت
        self.position_size_ratio = 0.12  # 12% بالانس
        
        # Threshold های متعادل
        self.entry_threshold = 0.40   # پایین‌تر برای تعداد بیشتر
        self.exit_threshold = 0.50    # متعادل
        
        # Risk Management معقول
        self.stop_loss_pct = 0.02     # 2% stop-loss
        self.take_profit_pct = 0.035  # 3.5% take-profit
        self.max_position_time = 40   # 40 دقیقه
        self.min_volume_filter = 500000  # کمتر از قبل
        
        # فیلترهای ساده‌تر
        self.min_confidence = 0.35    # کمتر از قبل
        self.trend_filter = True      # ساده
        self.volume_boost_required = 1.1  # فقط 10% volume بیشتر
        
        # Scalers و features
        self.entry_scaler = RobustScaler()
        self.exit_scaler = RobustScaler()
        self.feature_columns = self.model_info['feature_columns']
        
        # آمار ترید
        self.trades_history = []
        self.position = None
        self.total_positions = 0
        
        print("✅ آماده برای ترید با تعداد زیاد و کیفیت مناسب!")
    
    def enhanced_feature_engineering(self, df):
        """
        Feature engineering
        """
        df_enhanced = df.copy()
        
        # Technical Indicators
        df_enhanced['rsi_divergence'] = df_enhanced['rsi'].diff()
        df_enhanced['macd_momentum'] = df_enhanced['macd'] - df_enhanced['macd'].shift(1)
        df_enhanced['volume_surge'] = df_enhanced['volume'] / df_enhanced['volume'].rolling(20).mean()
        
        # Price action
        df_enhanced['price_momentum'] = df_enhanced['close'].pct_change(5)
        df_enhanced['volatility_regime'] = (df_enhanced['volatility_5m'] > df_enhanced['volatility_5m'].rolling(50).quantile(0.8)).astype(int)
        
        # Market microstructure
        df_enhanced['spread_proxy'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close']
        df_enhanced['volume_price_trend'] = df_enhanced['volume'] * df_enhanced['price_change_1m']
        
        # Momentum
        df_enhanced['momentum_5'] = df_enhanced['close'] / df_enhanced['close'].shift(5) - 1
        df_enhanced['momentum_15'] = df_enhanced['close'] / df_enhanced['close'].shift(15) - 1
        df_enhanced['momentum_consistency'] = (df_enhanced['momentum_5'] * df_enhanced['momentum_15'] > 0).astype(int)
        
        # Regime
        df_enhanced['trend_regime'] = (df_enhanced['close'] > df_enhanced['close'].rolling(20).mean()).astype(int)
        df_enhanced['volatility_normalized'] = df_enhanced['volatility_5m'] / df_enhanced['volatility_5m'].rolling(100).mean()
        
        return df_enhanced
    
    def apply_simple_filters(self, df_signals):
        """
        فیلترهای ساده برای حفظ تعداد مناسب
        """
        print("🔍 اعمال فیلترهای ساده...")
        
        filtered_signals = df_signals.copy()
        
        # فیلتر 1: حداقل اطمینان
        confidence_filter = filtered_signals['entry_prob'] > self.min_confidence
        
        # فیلتر 2: ترند ساده
        if self.trend_filter:
            simple_trend = filtered_signals['close'] > filtered_signals['close'].shift(5)
        else:
            simple_trend = pd.Series(True, index=filtered_signals.index)
        
        # فیلتر 3: Volume boost ساده
        volume_filter = filtered_signals['volume_surge'] > self.volume_boost_required
        
        # اعمال فیلترها
        all_filters = confidence_filter & simple_trend & volume_filter
        
        # فیلتر کردن
        filtered_signals.loc[~all_filters, 'entry_signal'] = 0
        
        original_count = df_signals['entry_signal'].sum()
        filtered_count = filtered_signals['entry_signal'].sum()
        filter_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
        
        print(f"📊 سیگنال‌های اصلی: {original_count}")
        print(f"📊 سیگنال‌های فیلتر شده: {filtered_count}")
        print(f"📊 فیلتر شده: {filter_pct:.1f}%")
        
        return filtered_signals
    
    def load_and_predict(self, start_row=-2000, end_row=None):
        """
        بارگذاری و پیش‌بینی
        """
        print(f"📥 بارگذاری داده‌ها...")
        
        # بارگذاری
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
        
        # Feature engineering
        df_enhanced = self.enhanced_feature_engineering(df)
        df_clean = df_enhanced.dropna()
        
        # Features
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
        
        # Threshold
        entry_signals = (entry_probs > self.entry_threshold).astype(int)
        exit_signals = (exit_probs > self.exit_threshold).astype(int)
        
        # اضافه کردن
        df_signals = df_clean.copy()
        df_signals['entry_prob'] = entry_probs
        df_signals['exit_prob'] = exit_probs
        df_signals['entry_signal'] = entry_signals
        df_signals['exit_signal'] = exit_signals
        
        # فیلترهای ساده
        df_final = self.apply_simple_filters(df_signals)
        
        print(f"✅ {len(df_final)} رکورد آماده شد")
        
        return df_final
    
    def should_enter_trade(self, row):
        """
        شرایط ورود ساده
        """
        if row['entry_signal'] != 1:
            return False
        
        if row['volume'] < self.min_volume_filter:
            return False
        
        return True
    
    def execute_backtest(self, df_signals):
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
            
            # Portfolio value
            portfolio_value = self.current_balance
            if self.position:
                portfolio_value += self.position['quantity'] * current_price
            portfolio_values.append(portfolio_value)
            
            # مدیریت موقعیت
            if self.position:
                self._manage_position(row, candle_index, timestamp)
            
            # ورود جدید
            if not self.position and self.should_enter_trade(row):
                self._open_position(row, candle_index, timestamp)
            
            candle_index += 1
        
        # بستن آخرین موقعیت
        if self.position:
            final_row = df_signals.iloc[-1]
            self._close_position(final_row, len(df_signals)-1, 'End of Period')
        
        # نتایج
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
    
    def _manage_position(self, row, current_index, current_timestamp):
        """
        مدیریت موقعیت
        """
        if not self.position:
            return
            
        current_price = row['close']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        
        # PnL
        pnl_pct = (current_price - entry_price) / entry_price
        time_in_position = current_index - entry_time
        
        # شرایط خروج
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
        
        # Max Time
        elif time_in_position >= self.max_position_time:
            should_exit = True
            exit_reason = 'Max Time'
        
        # Exit Signal
        elif row['exit_signal'] == 1:
            should_exit = True
            exit_reason = 'Exit Signal'
        
        if should_exit:
            self._close_position(row, current_index, exit_reason)
    
    def _open_position(self, row, index, timestamp):
        """
        باز کردن موقعیت
        """
        current_price = row['close']
        position_value = self.current_balance * self.position_size_ratio
        
        fee = position_value * self.trading_fee
        quantity = (position_value - fee) / current_price
        
        if quantity > 0:
            self.position = {
                'type': 'long',
                'entry_price': current_price,
                'quantity': quantity,
                'entry_time': index,
                'entry_timestamp': timestamp,
                'entry_fee': fee
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
        
        # Exit value
        exit_value = self.position['quantity'] * current_price
        exit_fee = exit_value * self.trading_fee
        net_exit_value = exit_value - exit_fee
        
        # PnL
        total_fees = self.position['entry_fee'] + exit_fee
        net_pnl = net_exit_value - (self.position['quantity'] * self.position['entry_price'])
        pnl_pct = net_pnl / (self.position['quantity'] * self.position['entry_price']) * 100
        
        # ثبت
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
        self.current_balance += net_exit_value
        self.position = None
    
    def analyze_results(self, results):
        """
        تحلیل نتایج
        """
        print("\n" + "="*60)
        print("📊 نتایج استراتژی متعادل")
        print("="*60)
        
        print(f"💰 بالانس اولیه: ${results['initial_balance']:,.2f}")
        print(f"💰 بالانس نهایی: ${results['final_balance']:,.2f}")
        
        total_return = results['total_return']
        profit = results['final_balance'] - results['initial_balance']
        
        if total_return > 0:
            print(f"📈 بازدهی: +{total_return:.2f}% 🎉")
        else:
            print(f"📉 بازدهی: {total_return:.2f}%")
        
        print(f"💵 سود/ضرر: ${profit:+.2f}")
        print(f"🔄 تعداد تریدها: {results['total_trades']}")
        
        if results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            avg_duration = trades_df['duration'].mean()
            
            print(f"\n📊 آمار عملکرد:")
            print(f"✅ Win Rate: {win_rate:.1f}%")
            print(f"💚 میانگین سود: ${avg_win:.2f}")
            print(f"💔 میانگین ضرر: ${avg_loss:.2f}")
            print(f"⏱️ میانگین مدت: {avg_duration:.1f} دقیقه")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
            
            # دلایل خروج
            print(f"\n🚪 دلایل خروج:")
            exit_reasons = trades_df['reason'].value_counts()
            for reason, count in exit_reasons.items():
                print(f"   {reason}: {count} ({count/len(trades_df)*100:.1f}%)")
        
        # نتیجه‌گیری
        print(f"\n💡 نتیجه‌گیری:")
        if results['total_trades'] >= 20 and total_return > -2:
            print("✅ تعداد و کیفیت مناسب!")
        elif results['total_trades'] < 10:
            print("⚠️ تعداد ترید کم - threshold کاهش دهید")
        elif total_return < -5:
            print("⚠️ ضرر زیاد - risk management بهبود دهید")
        else:
            print("🔧 نیاز به تنظیم بیشتر")
        
        return results
    
    def run_balanced_test(self, start_row=-2500, end_row=None):
        """
        اجرای تست متعادل
        """
        print("🚀 تست استراتژی متعادل")
        print("🎯 هدف: تعداد زیاد + کیفیت مناسب")
        print("="*50)
        
        # پیش‌بینی
        df_signals = self.load_and_predict(start_row, end_row)
        
        # Backtest
        results = self.execute_backtest(df_signals)
        
        # تحلیل
        final_results = self.analyze_results(results)
        
        return final_results

if __name__ == "__main__":
    # تست استراتژی متعادل
    strategy = BalancedStrategy(initial_balance=1000.0)
    
    print("🎯 استراتژی متعادل برای شما:")
    print("   - Entry Threshold: 0.40 (متعادل)")
    print("   - فیلترهای ساده (حفظ تعداد)")
    print("   - Risk Management معقول")
    print("   - Position Size: 12%")
    
    results = strategy.run_balanced_test(start_row=-2500, end_row=None)
    
    print(f"\n🎯 خلاصه نهایی:")
    print(f"🔄 تعداد تریدها: {results['total_trades']}")
    print(f"📈 بازدهی: {results['total_return']:+.2f}%")
    
    if results['total_trades'] >= 15 and results['total_return'] > -1:
        print("🎉 ترکیب مناسبی از تعداد و کیفیت!")
    else:
        print("🔧 هنوز می‌توان بهتر کرد")



