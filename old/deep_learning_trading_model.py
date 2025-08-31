import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DeepTradingModel:
    def __init__(self, input_features=20, lstm_units=128, dense_units=256, dropout_rate=0.3):
        """
        Deep Learning Model for Trading Prediction
        
        Args:
            input_features: Number of input features
            lstm_units: Number of LSTM units
            dense_units: Number of dense layer units
            dropout_rate: Dropout rate for regularization
        """
        self.input_features = input_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_and_preprocess_data(self, training_data_path, optimal_trades_path=None):
        """
        Load and preprocess training data
        """
        print("Loading training data...")
        df = pd.read_csv(training_data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        
        # Define feature columns (excluding timestamp and target variables)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
            'volatility_5m', 'volatility_15m', 'volume_ratio',
            'price_change_1m', 'price_change_5m', 'price_change_15m',
            'trend_strength', 'volatility_percentile', 'volume_percentile',
            'hour', 'day_of_week', 'minute'
        ]
        
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(method='ffill').fillna(0)
        
        # Extract features and targets
        X = df[feature_columns].values
        y_entry = df['is_optimal_entry'].astype(int).values
        y_exit = df['is_optimal_exit'].astype(int).values
        y_profit = df['future_profit_potential'].values
        
        print(f"Data shape: {X.shape}")
        print(f"Entry points: {y_entry.sum()} out of {len(y_entry)} ({y_entry.mean()*100:.2f}%)")
        print(f"Exit points: {y_exit.sum()} out of {len(y_exit)} ({y_exit.mean()*100:.2f}%)")
        print(f"Average profit potential: {y_profit.mean():.6f}")
        
        return X, y_entry, y_exit, y_profit, feature_columns
        
    def create_sequence_data(self, X, y_entry, y_exit, y_profit, sequence_length=60):
        """
        Create sequence data for LSTM
        """
        print(f"Creating sequences with length {sequence_length}...")
        
        X_seq, y_entry_seq, y_exit_seq, y_profit_seq = [], [], [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_entry_seq.append(y_entry[i])
            y_exit_seq.append(y_exit[i])
            y_profit_seq.append(y_profit[i])
            
        return (np.array(X_seq), np.array(y_entry_seq), 
                np.array(y_exit_seq), np.array(y_profit_seq))
    
    def build_model(self, sequence_length=60, feature_count=22):
        """
        Build multi-task deep learning model
        """
        print("Building model architecture...")
        
        # Input layer
        inputs = keras.Input(shape=(sequence_length, feature_count))
        
        # LSTM layers with attention
        lstm1 = layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(inputs)
        lstm1 = layers.BatchNormalization()(lstm1)
        
        lstm2 = layers.LSTM(self.lstm_units//2, return_sequences=True, dropout=self.dropout_rate)(lstm1)
        lstm2 = layers.BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention = layers.Attention()([lstm2, lstm2])
        
        # Global pooling
        global_avg = layers.GlobalAveragePooling1D()(attention)
        global_max = layers.GlobalMaxPooling1D()(attention)
        
        # Concatenate pooled features
        concat = layers.Concatenate()([global_avg, global_max])
        
        # Dense layers with residual connections
        dense1 = layers.Dense(self.dense_units, activation='relu')(concat)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        
        dense2 = layers.Dense(self.dense_units//2, activation='relu')(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(self.dropout_rate)(dense2)
        
        # Add residual connection
        if self.dense_units == self.dense_units//2:
            residual = layers.Add()([dense1, dense2])
        else:
            residual = dense2
        
        # Task-specific outputs
        # Entry prediction (binary classification)
        entry_dense = layers.Dense(64, activation='relu', name='entry_dense')(residual)
        entry_output = layers.Dense(1, activation='sigmoid', name='entry_prediction')(entry_dense)
        
        # Exit prediction (binary classification)
        exit_dense = layers.Dense(64, activation='relu', name='exit_dense')(residual)
        exit_output = layers.Dense(1, activation='sigmoid', name='exit_prediction')(exit_dense)
        
        # Profit prediction (regression)
        profit_dense = layers.Dense(64, activation='relu', name='profit_dense')(residual)
        profit_output = layers.Dense(1, activation='linear', name='profit_prediction')(profit_dense)
        
        # Create model
        self.model = keras.Model(
            inputs=inputs,
            outputs=[entry_output, exit_output, profit_output],
            name='DeepTradingModel'
        )
        
        # Compile model with different loss functions for each task
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'entry_prediction': 'binary_crossentropy',
                'exit_prediction': 'binary_crossentropy', 
                'profit_prediction': 'mse'
            },
            loss_weights={
                'entry_prediction': 1.0,
                'exit_prediction': 1.0,
                'profit_prediction': 0.5
            },
            metrics={
                'entry_prediction': ['accuracy', 'precision', 'recall'],
                'exit_prediction': ['accuracy', 'precision', 'recall'],
                'profit_prediction': ['mae']
            }
        )
        
        print(self.model.summary())
        return self.model
    
    def train_model(self, X_train, y_entry_train, y_exit_train, y_profit_train,
                    X_val, y_entry_val, y_exit_val, y_profit_val, 
                    epochs=100, batch_size=32):
        """
        Train the model
        """
        print("Training model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_trading_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train,
            {
                'entry_prediction': y_entry_train,
                'exit_prediction': y_exit_train,
                'profit_prediction': y_profit_train
            },
            validation_data=(
                X_val,
                {
                    'entry_prediction': y_entry_val,
                    'exit_prediction': y_exit_val,
                    'profit_prediction': y_profit_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_entry_test, y_exit_test, y_profit_test):
        """
        Evaluate model performance
        """
        print("Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        entry_pred, exit_pred, profit_pred = predictions
        
        # Convert probabilities to binary predictions
        entry_pred_binary = (entry_pred > 0.5).astype(int)
        exit_pred_binary = (exit_pred > 0.5).astype(int)
        
        # Entry point evaluation
        print("\n=== ENTRY POINT PREDICTION ===")
        print(classification_report(y_entry_test, entry_pred_binary))
        
        # Exit point evaluation
        print("\n=== EXIT POINT PREDICTION ===")
        print(classification_report(y_exit_test, exit_pred_binary))
        
        # Profit prediction evaluation
        print("\n=== PROFIT PREDICTION ===")
        profit_mae = np.mean(np.abs(profit_pred.flatten() - y_profit_test))
        profit_rmse = np.sqrt(np.mean((profit_pred.flatten() - y_profit_test) ** 2))
        print(f"MAE: {profit_mae:.6f}")
        print(f"RMSE: {profit_rmse:.6f}")
        
        return {
            'entry_predictions': entry_pred,
            'exit_predictions': exit_pred,
            'profit_predictions': profit_pred,
            'entry_binary': entry_pred_binary,
            'exit_binary': exit_pred_binary
        }
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Entry accuracy
        axes[0, 1].plot(self.history.history['entry_prediction_accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_entry_prediction_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Entry Prediction Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Exit accuracy
        axes[1, 0].plot(self.history.history['exit_prediction_accuracy'], label='Training Accuracy')
        axes[1, 0].plot(self.history.history['val_exit_prediction_accuracy'], label='Validation Accuracy')
        axes[1, 0].set_title('Exit Prediction Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Profit MAE
        axes[1, 1].plot(self.history.history['profit_prediction_mae'], label='Training MAE')
        axes[1, 1].plot(self.history.history['val_profit_prediction_mae'], label='Validation MAE')
        axes[1, 1].set_title('Profit Prediction MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_signals(self, X_new):
        """
        Predict trading signals for new data
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        predictions = self.model.predict(X_new)
        entry_prob, exit_prob, profit_pred = predictions
        
        return {
            'entry_probability': entry_prob.flatten(),
            'exit_probability': exit_prob.flatten(),
            'profit_prediction': profit_pred.flatten(),
            'entry_signal': (entry_prob > 0.5).flatten(),
            'exit_signal': (exit_prob > 0.5).flatten()
        }
    
    def save_model(self, filepath='deep_trading_model.h5'):
        """
        Save the trained model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath='deep_trading_model.h5'):
        """
        Load a trained model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def main():
    """
    Main training pipeline
    """
    print("🚀 Starting Deep Learning Trading Model Training")
    print("=" * 50)
    
    # Initialize model
    model = DeepTradingModel(
        lstm_units=128,
        dense_units=256,
        dropout_rate=0.3
    )
    
    # Load and preprocess data
    X, y_entry, y_exit, y_profit, feature_columns = model.load_and_preprocess_data(
        'training_data.csv'
    )
    
    # Scale features
    X_scaled = model.scaler.fit_transform(X)
    
    # Create sequences
    sequence_length = 60
    X_seq, y_entry_seq, y_exit_seq, y_profit_seq = model.create_sequence_data(
        X_scaled, y_entry, y_exit, y_profit, sequence_length
    )
    
    print(f"Sequence data shape: {X_seq.shape}")
    
    # Split data
    test_size = 0.2
    val_size = 0.2
    
    # First split: train+val vs test
    X_temp, X_test, y_entry_temp, y_entry_test, y_exit_temp, y_exit_test, y_profit_temp, y_profit_test = train_test_split(
        X_seq, y_entry_seq, y_exit_seq, y_profit_seq, 
        test_size=test_size, random_state=42, stratify=y_entry_seq
    )
    
    # Second split: train vs val
    X_train, X_val, y_entry_train, y_entry_val, y_exit_train, y_exit_val, y_profit_train, y_profit_val = train_test_split(
        X_temp, y_entry_temp, y_exit_temp, y_profit_temp,
        test_size=val_size/(1-test_size), random_state=42, stratify=y_entry_temp
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    model.build_model(sequence_length, X_seq.shape[2])
    
    # Train model
    history = model.train_model(
        X_train, y_entry_train, y_exit_train, y_profit_train,
        X_val, y_entry_val, y_exit_val, y_profit_val,
        epochs=100,
        batch_size=32
    )
    
    # Evaluate model
    results = model.evaluate_model(X_test, y_entry_test, y_exit_test, y_profit_test)
    
    # Plot training history
    model.plot_training_history()
    
    # Save model
    model.save_model('deep_trading_model.h5')
    
    print("\n🎉 Training completed successfully!")
    print("Model saved as 'deep_trading_model.h5'")
    print("Training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main() 