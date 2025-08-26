import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeepLearningTradingSystem:
    def __init__(self, model_path='deep_trading_model.h5', sequence_length=60):
        """
        Deep Learning Trading System for Real-time Predictions
        
        Args:
            model_path: Path to the trained model
            sequence_length: Sequence length used for training
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
            'volatility_5m', 'volatility_15m', 'volume_ratio',
            'price_change_1m', 'price_change_5m', 'price_change_15m',
            'trend_strength', 'volatility_percentile', 'volume_percentile',
            'hour', 'day_of_week', 'minute'
        ]
        self.is_trained = False
        self.last_sequence = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def prepare_scaler(self, training_data_path):
        """Prepare the scaler using training data"""
        print("Preparing feature scaler...")
        df = pd.read_csv(training_data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        
        # Handle missing values
        df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill').fillna(0)
        
        # Fit scaler
        X = df[self.feature_columns].values
        self.scaler.fit(X)
        self.is_trained = True
        print("✅ Scaler prepared successfully")
        
    def preprocess_data(self, df):
        """
        Preprocess new data for prediction
        """
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        
        # Handle missing values
        df[self.feature_columns] = df[self.feature_columns].fillna(method='ffill').fillna(0)
        
        # Extract features
        X = df[self.feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def create_sequence(self, X):
        """Create sequence for LSTM prediction"""
        if len(X) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # Take the last sequence_length points
        sequence = X[-self.sequence_length:]
        return np.expand_dims(sequence, axis=0)  # Add batch dimension
    
    def predict_signals(self, df):
        """
        Predict trading signals for new data
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self.is_trained:
            raise ValueError("Scaler not prepared. Call prepare_scaler() first.")
        
        # Preprocess data
        X_scaled = self.preprocess_data(df)
        
        # Create sequence
        X_seq = self.create_sequence(X_scaled)
        
        # Make prediction
        predictions = self.model.predict(X_seq, verbose=0)
        entry_prob, exit_prob, profit_pred = predictions
        
        # Store last sequence for future use
        self.last_sequence = X_scaled
        
        return {
            'entry_probability': float(entry_prob[0][0]),
            'exit_probability': float(exit_prob[0][0]),
            'profit_prediction': float(profit_pred[0][0]),
            'entry_signal': bool(entry_prob[0][0] > 0.5),
            'exit_signal': bool(exit_prob[0][0] > 0.5),
            'confidence_entry': float(abs(entry_prob[0][0] - 0.5) * 2),  # 0-1 scale
            'confidence_exit': float(abs(exit_prob[0][0] - 0.5) * 2)     # 0-1 scale
        }
    
    def get_trading_recommendation(self, df, current_position=None, risk_threshold=0.6):
        """
        Get trading recommendation based on model predictions
        
        Args:
            df: DataFrame with market data
            current_position: 'long', 'short', or None
            risk_threshold: Minimum confidence required for action
        """
        signals = self.predict_signals(df)
        
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': '',
            'signals': signals
        }
        
        entry_confidence = signals['confidence_entry']
        exit_confidence = signals['confidence_exit']
        entry_signal = signals['entry_signal']
        exit_signal = signals['exit_signal']
        profit_pred = signals['profit_prediction']
        
        # Current position logic
        if current_position is None:  # No position
            if entry_signal and entry_confidence > risk_threshold and profit_pred > 0:
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = entry_confidence
                recommendation['reason'] = f'Strong entry signal (confidence: {entry_confidence:.2f}, predicted profit: {profit_pred:.4f})'
        
        elif current_position == 'long':  # Currently in long position
            if exit_signal and exit_confidence > risk_threshold:
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = exit_confidence
                recommendation['reason'] = f'Strong exit signal (confidence: {exit_confidence:.2f})'
            elif profit_pred < -0.001:  # Predicted loss
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = min(0.8, abs(profit_pred) * 100)
                recommendation['reason'] = f'Predicted loss: {profit_pred:.4f}'
        
        return recommendation
    
    def backtest_on_data(self, df, initial_balance=1000, transaction_fee=0.001):
        """
        Backtest the model on historical data
        """
        print("Starting backtest...")
        
        balance = initial_balance
        position = None
        position_size = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Need at least sequence_length + 1 points
        for i in range(self.sequence_length, len(df)):
            current_data = df.iloc[:i+1].copy()
            current_price = df.iloc[i]['close']
            timestamp = df.iloc[i]['timestamp']
            
            try:
                # Get recommendation
                recommendation = self.get_trading_recommendation(
                    current_data, 
                    current_position=position,
                    risk_threshold=0.6
                )
                
                action = recommendation['action']
                confidence = recommendation['confidence']
                
                # Execute trades
                if action == 'BUY' and position is None:
                    # Enter long position
                    position = 'long'
                    position_size = (balance * 0.95) / current_price  # Use 95% of balance
                    entry_price = current_price
                    balance -= position_size * current_price * (1 + transaction_fee)
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': current_price,
                        'size': position_size,
                        'confidence': confidence,
                        'balance': balance
                    })
                
                elif action == 'SELL' and position == 'long':
                    # Exit long position
                    sale_value = position_size * current_price * (1 - transaction_fee)
                    balance += sale_value
                    
                    profit = sale_value - (position_size * entry_price)
                    profit_pct = (current_price - entry_price) / entry_price
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': current_price,
                        'size': position_size,
                        'confidence': confidence,
                        'balance': balance,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    
                    position = None
                    position_size = 0
                    entry_price = 0
                
                # Calculate current equity
                if position == 'long':
                    current_equity = balance + (position_size * current_price)
                else:
                    current_equity = balance
                
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'price': current_price
                })
                
            except Exception as e:
                print(f"Error at index {i}: {e}")
                continue
        
        # Close final position if any
        if position == 'long':
            final_price = df.iloc[-1]['close']
            final_value = position_size * final_price * (1 - transaction_fee)
            balance += final_value
            
            trades.append({
                'timestamp': df.iloc[-1]['timestamp'],
                'action': 'SELL',
                'price': final_price,
                'size': position_size,
                'confidence': 0.5,
                'balance': balance,
                'profit': final_value - (position_size * entry_price),
                'profit_pct': (final_price - entry_price) / entry_price
            })
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance,
            'num_trades': len([t for t in trades if t['action'] == 'BUY'])
        }
    
    def plot_backtest_results(self, backtest_results, df):
        """Plot backtest results"""
        trades = pd.DataFrame(backtest_results['trades'])
        equity_curve = pd.DataFrame(backtest_results['equity_curve'])
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price chart with trades
        axes[0].plot(df['timestamp'], df['close'], label='Price', alpha=0.7)
        
        if not trades.empty:
            buy_trades = trades[trades['action'] == 'BUY']
            sell_trades = trades[trades['action'] == 'SELL']
            
            axes[0].scatter(buy_trades['timestamp'], buy_trades['price'], 
                           color='green', marker='^', s=100, label='Buy', zorder=5)
            axes[0].scatter(sell_trades['timestamp'], sell_trades['price'], 
                           color='red', marker='v', s=100, label='Sell', zorder=5)
        
        axes[0].set_title('Price Chart with Trading Signals')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Equity curve
        if not equity_curve.empty:
            axes[1].plot(equity_curve['timestamp'], equity_curve['equity'], 
                        label='Portfolio Value', color='blue')
            axes[1].set_title('Portfolio Equity Curve')
            axes[1].set_ylabel('Portfolio Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Trade profits
        if not trades.empty:
            profitable_trades = trades[trades['action'] == 'SELL']
            if not profitable_trades.empty:
                axes[2].bar(range(len(profitable_trades)), profitable_trades['profit'], 
                           color=['green' if p > 0 else 'red' for p in profitable_trades['profit']])
                axes[2].set_title('Individual Trade Profits')
                axes[2].set_ylabel('Profit/Loss')
                axes[2].set_xlabel('Trade Number')
                axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("📊 BACKTEST SUMMARY")
        print("="*50)
        print(f"Initial Balance: ${backtest_results['final_balance']:.2f}")
        print(f"Final Balance: ${backtest_results['final_balance']:.2f}")
        print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
        print(f"Number of Trades: {backtest_results['num_trades']}")
        
        if not trades.empty:
            profitable_trades = trades[(trades['action'] == 'SELL') & (trades['profit'] > 0)]
            losing_trades = trades[(trades['action'] == 'SELL') & (trades['profit'] <= 0)]
            
            if len(profitable_trades) > 0 or len(losing_trades) > 0:
                win_rate = len(profitable_trades) / (len(profitable_trades) + len(losing_trades))
                print(f"Win Rate: {win_rate*100:.2f}%")
                
                if len(profitable_trades) > 0:
                    avg_win = profitable_trades['profit'].mean()
                    print(f"Average Win: ${avg_win:.2f}")
                
                if len(losing_trades) > 0:
                    avg_loss = losing_trades['profit'].mean()
                    print(f"Average Loss: ${avg_loss:.2f}")

def main():
    """
    Main function to demonstrate the trading system
    """
    print("🤖 Deep Learning Trading System")
    print("="*50)
    
    # Initialize trading system
    trading_system = DeepLearningTradingSystem()
    
    # Check if model exists, if not, train it first
    try:
        if not trading_system.load_model():
            print("❌ Model not found. Please train the model first using deep_learning_trading_model.py")
            return
        
        # Prepare scaler
        trading_system.prepare_scaler('training_data.csv')
        
        # Load test data
        print("Loading test data...")
        df = pd.read_csv('training_data.csv')
        
        # Use last 20% of data for testing
        test_start = int(len(df) * 0.8)
        test_df = df.iloc[test_start:].copy().reset_index(drop=True)
        
        print(f"Testing on {len(test_df)} data points")
        
        # Run backtest
        backtest_results = trading_system.backtest_on_data(test_df)
        
        # Plot results
        trading_system.plot_backtest_results(backtest_results, test_df)
        
        print("\n✅ Backtest completed successfully!")
        print("📈 Results saved as 'backtest_results.png'")
        
        # Example of real-time prediction
        print("\n🔮 Example Real-time Prediction:")
        print("-" * 30)
        
        # Use last 100 points for prediction
        recent_data = df.tail(100).copy()
        signals = trading_system.predict_signals(recent_data)
        
        print(f"Entry Probability: {signals['entry_probability']:.3f}")
        print(f"Exit Probability: {signals['exit_probability']:.3f}")
        print(f"Profit Prediction: {signals['profit_prediction']:.6f}")
        print(f"Entry Signal: {'BUY' if signals['entry_signal'] else 'NO BUY'}")
        print(f"Exit Signal: {'SELL' if signals['exit_signal'] else 'NO SELL'}")
        print(f"Entry Confidence: {signals['confidence_entry']:.3f}")
        print(f"Exit Confidence: {signals['confidence_exit']:.3f}")
        
        # Get recommendation
        recommendation = trading_system.get_trading_recommendation(recent_data)
        print(f"\n🎯 Recommendation: {recommendation['action']}")
        print(f"Confidence: {recommendation['confidence']:.3f}")
        print(f"Reason: {recommendation['reason']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 