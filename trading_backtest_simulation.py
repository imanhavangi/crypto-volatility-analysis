import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingBacktestSimulator:
    def __init__(self, model_path='best_improved_model.h5', initial_balance=1000.0):
        """
        سیستم شبیه‌سازی backtesting با مدل deep learning
        
        Args:
            model_path: مسیر مدل آموزش‌دیده
            initial_balance: بالانس اولیه (دلار)
        """
        print("🚀 بارگذاری مدل...")
        self.model = tf.keras.models.load_model(model_path)
        
        # تنظیمات مالی
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_fee = 0.001  # 0.1% کارمزد
        
        # تنظیمات ترید
        self.position_size_ratio = 0.25  # 25% بالانس در هر معامله
        self.min_confidence = 0.15  # حداقل اطمینان برای ورود
        self.max_daily_trades = 10  # حداکثر تعداد ترید در روز
        
        # آمار ترید
        self.trades_history = []
        self.daily_pnl = []
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
        # موقعیت فعلی
        self.position = None  # {'type': 'long', 'entry_price': float, 'size': float, 'entry_time': datetime}
        
        print("✅ سیستم آماده است!")
    
    def prepare_features(self, data):
        """آماده‌سازی فیچرها مانند مدل اصلی"""
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m',
            'price_change_15m', 'trend_strength', 'volatility_percentile',
            'volume_percentile'
        ]
        
        # بررسی و جایگزینی NaN
        features_data = data[feature_columns].copy()
        features_data = features_data.fillna(0)
        
        return features_data.values
    
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
    
    def calculate_position_size(self, price):
        """محاسبه اندازه موقعیت"""
        available_balance = self.current_balance
        position_value = available_balance * self.position_size_ratio
        
        # اندازه موقعیت (تعداد coin)
        size = position_value / price
        
        return size, position_value
    
    def execute_entry(self, row, entry_prob):
        """اجرای ورود به معامله"""
        if self.position is not None:
            return False  # در حال حاضر موقعیت داریم
        
        price = row['close']
        size, position_value = self.calculate_position_size(price)
        
        # محاسبه کارمزد
        fee = position_value * self.trading_fee
        
        # بررسی بالانس کافی
        if position_value + fee > self.current_balance:
            return False
        
        # اجرای ترید
        self.current_balance -= (position_value + fee)
        
        self.position = {
            'type': 'long',
            'entry_price': price,
            'size': size,
            'entry_time': row['timestamp'],
            'entry_confidence': entry_prob,
            'entry_fee': fee
        }
        
        print(f"📈 LONG @ {price:.5f} | Size: {size:.2f} | Confidence: {entry_prob:.3f} | Fee: ${fee:.2f}")
        
        return True
    
    def execute_exit(self, row, exit_prob):
        """اجرای خروج از معامله"""
        if self.position is None:
            return False  # موقعیت نداریم
        
        price = row['close']
        position_value = self.position['size'] * price
        
        # محاسبه کارمزد
        fee = position_value * self.trading_fee
        
        # محاسبه سود/ضرر
        profit = position_value - (self.position['size'] * self.position['entry_price'])
        net_profit = profit - self.position['entry_fee'] - fee
        
        # بازگشت پول به بالانس
        self.current_balance += (position_value - fee)
        
        # ثبت تاریخچه ترید
        trade_record = {
            'entry_time': self.position['entry_time'],
            'exit_time': row['timestamp'],
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'size': self.position['size'],
            'profit': net_profit,
            'profit_pct': (net_profit / (self.position['size'] * self.position['entry_price'])) * 100,
            'entry_confidence': self.position['entry_confidence'],
            'exit_confidence': exit_prob,
            'duration': (pd.to_datetime(row['timestamp']) - pd.to_datetime(self.position['entry_time'])).total_seconds() / 60  # دقیقه
        }
        
        self.trades_history.append(trade_record)
        
        print(f"📉 EXIT @ {price:.5f} | P&L: ${net_profit:.2f} ({trade_record['profit_pct']:.2f}%) | Duration: {trade_record['duration']:.1f}min")
        
        # پاک کردن موقعیت
        self.position = None
        
        return True
    
    def run_backtest(self, test_data_file='training_data.csv', start_date='2025-07-26', end_date='2025-08-02'):
        """اجرای backtesting روی بازه مشخص"""
        print(f"\n🎯 شروع backtesting از {start_date} تا {end_date}")
        print(f"💰 بالانس اولیه: ${self.initial_balance}")
        print("="*60)
        
        # بارگذاری داده‌ها
        df = pd.read_csv(test_data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # فیلتر بازه زمانی
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        test_df = df.loc[mask].copy().reset_index(drop=True)
        
        print(f"📊 تعداد کل نقاط داده: {len(test_df)}")
        
        # شبیه‌سازی ترید
        for i, row in test_df.iterrows():
            # آماده‌سازی فیچرها
            features = self.prepare_features(row)
            
            # دریافت پیش‌بینی
            entry_prob, exit_prob = self.get_prediction(features)
            
            # تصمیم‌گیری ترید
            if self.position is None and entry_prob > self.min_confidence:
                # سیگنال ورود
                self.execute_entry(row, entry_prob)
            
            elif self.position is not None and exit_prob > self.min_confidence:
                # سیگنال خروج
                self.execute_exit(row, exit_prob)
            
            # بررسی Stop Loss (اختیاری)
            if self.position is not None:
                current_price = row['close']
                unrealized_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
                
                # Stop Loss در -5%
                if unrealized_pnl < -0.05:
                    print(f"🛑 STOP LOSS @ {current_price:.5f}")
                    self.execute_exit(row, 0.0)
            
            # ثبت روزانه
            if i % 1440 == 0:  # هر 1440 دقیقه (یک روز)
                self.daily_pnl.append(self.current_balance)
                
                # محاسبه Max Drawdown
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance
                else:
                    drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                    self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # بستن موقعیت باز در پایان
        if self.position is not None:
            last_row = test_df.iloc[-1]
            print("🔚 بستن موقعیت در پایان backtesting...")
            self.execute_exit(last_row, 0.0)
        
        # نمایش نتایج
        self.display_results()
    
    def display_results(self):
        """نمایش نتایج کامل backtesting"""
        print("\n" + "="*60)
        print("📈 نتایج نهایی BACKTESTING")
        print("="*60)
        
        # نتایج مالی
        total_return = self.current_balance - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        print(f"💰 بالانس نهایی: ${self.current_balance:.2f}")
        print(f"📊 سود/ضرر کل: ${total_return:.2f} ({return_pct:.2f}%)")
        print(f"📉 حداکثر افت (Max Drawdown): {self.max_drawdown*100:.2f}%")
        
        # آمار ترید
        if self.trades_history:
            trades_df = pd.DataFrame(self.trades_history)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            losing_trades = len(trades_df[trades_df['profit'] <= 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_profit = trades_df['profit'].mean()
            avg_winning_profit = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
            avg_losing_profit = trades_df[trades_df['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
            
            print(f"\n🎯 آمار ترید:")
            print(f"   کل تریدها: {total_trades}")
            print(f"   تریدهای برنده: {winning_trades} ({win_rate:.1f}%)")
            print(f"   تریدهای بازنده: {losing_trades} ({100-win_rate:.1f}%)")
            print(f"   متوسط سود هر ترید: ${avg_profit:.2f}")
            print(f"   متوسط سود برنده: ${avg_winning_profit:.2f}")
            print(f"   متوسط ضرر بازنده: ${avg_losing_profit:.2f}")
            
            # نمایش جدول تریدها
            print(f"\n📋 جزئیات تریدها:")
            print(trades_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'profit_pct', 'duration']].to_string(index=False))
            
            # رسم نمودار
            self.plot_results(trades_df)
        else:
            print("\n⚠️ هیچ تریدی انجام نشد!")
    
    def plot_results(self, trades_df):
        """رسم نمودارهای نتایج"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # نمودار تغییرات بالانس
        balance_history = [self.initial_balance]
        running_balance = self.initial_balance
        
        for _, trade in trades_df.iterrows():
            running_balance += trade['profit']
            balance_history.append(running_balance)
        
        ax1.plot(balance_history, marker='o', linewidth=2)
        ax1.set_title('📈 تغییرات بالانس در طول زمان', fontsize=12, fontweight='bold')
        ax1.set_ylabel('بالانس (دلار)')
        ax1.grid(True, alpha=0.3)
        
        # توزیع سود/ضرر
        ax2.hist(trades_df['profit'], bins=20, alpha=0.7, color='green' if trades_df['profit'].mean() > 0 else 'red')
        ax2.set_title('📊 توزیع سود/ضرر تریدها', fontsize=12, fontweight='bold')
        ax2.set_xlabel('سود/ضرر (دلار)')
        ax2.set_ylabel('تعداد')
        ax2.grid(True, alpha=0.3)
        
        # Win/Loss نمودار دایره‌ای
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        
        labels = ['برنده', 'بازنده']
        sizes = [winning_trades, losing_trades]
        colors = ['green', 'red']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('🎯 نسبت تریدهای برنده/بازنده', fontsize=12, fontweight='bold')
        
        # مدت زمان تریدها
        ax4.scatter(trades_df['duration'], trades_df['profit'], alpha=0.6, 
                   c=['green' if p > 0 else 'red' for p in trades_df['profit']])
        ax4.set_title('⏱️ رابطه مدت زمان و سودآوری', fontsize=12, fontweight='bold')
        ax4.set_xlabel('مدت زمان (دقیقه)')
        ax4.set_ylabel('سود/ضرر (دلار)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """اجرای اصلی backtesting"""
    print("🚀 سیستم Backtesting مدل Deep Learning")
    print("="*50)
    
    # ایجاد simulator
    simulator = TradingBacktestSimulator(
        model_path='best_improved_model.h5',
        initial_balance=1000.0
    )
    
    # اجرای backtesting روی هفته اول
    simulator.run_backtest(
        test_data_file='training_data.csv',
        start_date='2025-07-26 00:00:00',
        end_date='2025-08-02 23:59:59'
    )

if __name__ == "__main__":
    main() 