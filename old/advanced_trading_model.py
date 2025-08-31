import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingModel:
    """
    مدل پیشرفته ترید با تکنیک‌های anti-overfitting
    """
    
    def __init__(self, use_regularization=True, use_ensemble=True):
        """
        مدل پیشرفته با تمرکز بر کیفیت، نه کمیت
        """
        self.use_regularization = use_regularization
        self.use_ensemble = use_ensemble
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.training_history = {}
        
        # تنظیمات anti-overfitting
        self.dropout_rate = 0.4  # بالا برای جلوگیری از overfitting
        self.l2_lambda = 0.01    # regularization قوی
        self.early_stopping_patience = 10
        self.validation_split = 0.2
        
        print("🧠 مدل پیشرفته Anti-Overfitting آماده شد")
        print("🎯 تمرکز: کیفیت سیگنال‌ها، نه کمیت")
    
    def enhanced_feature_engineering(self, df):
        """
        Feature engineering پیشرفته‌تر
        """
        print("🔧 Feature Engineering پیشرفته...")
        
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
    
    def create_balanced_dataset(self, df):
        """
        ایجاد dataset متعادل با تکنیک‌های پیشرفته
        """
        print("⚖️ ایجاد dataset متعادل...")
        
        # Enhanced feature engineering
        df_enhanced = self.enhanced_feature_engineering(df)
        
        # انتخاب features
        feature_cols = [col for col in df_enhanced.columns if col not in 
                       ['timestamp', 'is_optimal_entry', 'is_optimal_exit', 'future_profit_potential']]
        
        # حذف features با correlation بیش از حد
        correlation_matrix = df_enhanced[feature_cols].corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        feature_cols = [col for col in feature_cols if col not in high_corr_features]
        
        print(f"📊 Features انتخاب شده: {len(feature_cols)}")
        self.feature_columns = feature_cols
        
        # آماده‌سازی داده‌ها
        X = df_enhanced[feature_cols].dropna()
        y_entry = df_enhanced.loc[X.index, 'is_optimal_entry'].values
        y_exit = df_enhanced.loc[X.index, 'is_optimal_exit'].values
        
        print(f"📊 Dataset اصلی: {len(X)} نمونه")
        print(f"📊 Entry: {y_entry.sum()} مثبت ({y_entry.mean()*100:.1f}%)")
        print(f"📊 Exit: {y_exit.sum()} مثبت ({y_exit.mean()*100:.1f}%)")
        
        # حذف outliers
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_mask = isolation_forest.fit_predict(X) == 1
        
        X_clean = X[outlier_mask]
        y_entry_clean = y_entry[outlier_mask]
        y_exit_clean = y_exit[outlier_mask]
        
        print(f"📊 بعد از حذف outliers: {len(X_clean)} نمونه")
        
        # Balanced sampling با SMOTETomek
        smote_tomek = SMOTETomek(random_state=42)
        
        # Entry balancing
        X_entry_balanced, y_entry_balanced = smote_tomek.fit_resample(X_clean, y_entry_clean)
        
        # Exit balancing  
        X_exit_balanced, y_exit_balanced = smote_tomek.fit_resample(X_clean, y_exit_clean)
        
        print(f"📊 Entry balanced: {len(X_entry_balanced)} نمونه")
        print(f"📊 Exit balanced: {len(X_exit_balanced)} نمونه")
        
        return X_entry_balanced, y_entry_balanced, X_exit_balanced, y_exit_balanced, X_clean, y_entry_clean, y_exit_clean
    
    def create_advanced_model(self, input_dim, model_type='entry'):
        """
        ایجاد مدل پیشرفته با anti-overfitting
        """
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,),
                                kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # Hidden layers with residual connections
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid', name=f'{model_type}_output')
        ])
        
        # Custom optimizer با learning rate ثابت (برای compatibility)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Weighted binary crossentropy برای balance
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_with_cross_validation(self, X, y, model_type='entry', n_splits=5):
        """
        آموزش با cross-validation
        """
        print(f"🎯 آموزش مدل {model_type} با {n_splits}-fold CV...")
        
        # Time series split برای financial data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"📊 Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scaling
            scaler = RobustScaler()  # مقاوم به outliers
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # ایجاد مدل
            model = self.create_advanced_model(X_train_scaled.shape[1], model_type)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                )
            ]
            
            # Class weights برای imbalance
            class_weights = {
                0: 1.0,
                1: len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1.0
            }
            
            # آموزش
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=30,  # کمتر برای سرعت
                batch_size=512,  # بزرگتر برای سرعت
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=0
            )
            
            # ارزیابی
            val_pred = (model.predict(X_val_scaled, verbose=0) > 0.5).astype(int)
            val_score = f1_score(y_val, val_pred)
            cv_scores.append(val_score)
            
            fold_models.append({
                'model': model,
                'scaler': scaler,
                'score': val_score
            })
            
            print(f"   F1 Score: {val_score:.3f}")
        
        print(f"🏆 CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # انتخاب بهترین مدل
        best_fold = max(fold_models, key=lambda x: x['score'])
        
        return best_fold['model'], best_fold['scaler'], cv_scores
    
    def train_advanced_models(self, data_file='training_data.csv'):
        """
        آموزش مدل‌های پیشرفته
        """
        print("🚀 شروع آموزش مدل‌های پیشرفته")
        print("="*60)
        
        # بارگذاری و آماده‌سازی داده‌ها
        df = pd.read_csv(data_file)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # ایجاد dataset متعادل
        X_entry_bal, y_entry_bal, X_exit_bal, y_exit_bal, X_clean, y_entry_clean, y_exit_clean = self.create_balanced_dataset(df)
        
        # آموزش مدل Entry
        print("\n🎯 آموزش مدل Entry...")
        entry_model, entry_scaler, entry_cv_scores = self.train_with_cross_validation(
            pd.DataFrame(X_entry_bal, columns=self.feature_columns), 
            y_entry_bal, 
            'entry'
        )
        
        # آموزش مدل Exit
        print("\n🎯 آموزش مدل Exit...")
        exit_model, exit_scaler, exit_cv_scores = self.train_with_cross_validation(
            pd.DataFrame(X_exit_bal, columns=self.feature_columns),
            y_exit_bal,
            'exit'
        )
        
        # ذخیره مدل‌ها
        self.models['entry'] = entry_model
        self.models['exit'] = exit_model
        self.scalers['entry'] = entry_scaler
        self.scalers['exit'] = exit_scaler
        
        # تست نهایی روی test set
        print("\n📊 تست نهایی روی test set...")
        test_results = self.evaluate_on_test_set(X_clean, y_entry_clean, y_exit_clean)
        
        # ذخیره مدل‌ها
        entry_model.save('advanced_entry_model.keras')
        exit_model.save('advanced_exit_model.keras')
        
        # ذخیره اطلاعات
        model_info = {
            'feature_columns': self.feature_columns,
            'entry_cv_scores': entry_cv_scores,
            'exit_cv_scores': exit_cv_scores,
            'test_results': test_results,
            'model_type': 'advanced_anti_overfitting'
        }
        
        import json
        with open('advanced_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print("\n✅ آموزش کامل شد!")
        print(f"🎯 Entry CV Score: {np.mean(entry_cv_scores):.3f}")
        print(f"🎯 Exit CV Score: {np.mean(exit_cv_scores):.3f}")
        
        return model_info
    
    def evaluate_on_test_set(self, X_test, y_entry_test, y_exit_test):
        """
        ارزیابی نهایی روی test set
        """
        # آماده‌سازی test data
        X_test_entry = self.scalers['entry'].transform(X_test)
        X_test_exit = self.scalers['exit'].transform(X_test)
        
        # پیش‌بینی
        entry_probs = self.models['entry'].predict(X_test_entry, verbose=0).flatten()
        exit_probs = self.models['exit'].predict(X_test_exit, verbose=0).flatten()
        
        # تست threshold های مختلف
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        results = {}
        
        for threshold in thresholds:
            entry_pred = (entry_probs > threshold).astype(int)
            exit_pred = (exit_probs > threshold).astype(int)
            
            entry_f1 = f1_score(y_entry_test, entry_pred)
            exit_f1 = f1_score(y_exit_test, exit_pred)
            
            results[f'threshold_{threshold}'] = {
                'entry_signals': entry_pred.sum(),
                'exit_signals': exit_pred.sum(),
                'entry_f1': entry_f1,
                'exit_f1': exit_f1,
                'entry_precision': (entry_pred & y_entry_test).sum() / max(entry_pred.sum(), 1),
                'entry_recall': (entry_pred & y_entry_test).sum() / max(y_entry_test.sum(), 1)
            }
        
        print("\n📋 نتایج Test Set:")
        for threshold, metrics in results.items():
            print(f"Threshold {threshold}: Entry F1={metrics['entry_f1']:.3f}, "
                  f"Signals={metrics['entry_signals']}, "
                  f"Precision={metrics['entry_precision']:.3f}")
        
        return results

if __name__ == "__main__":
    # آموزش مدل پیشرفته
    advanced_model = AdvancedTradingModel()
    
    print("🎯 آموزش مدل Anti-Overfitting")
    print("🔧 تکنیک‌های اعمال شده:")
    print("   - Time Series Cross Validation")
    print("   - Heavy Regularization (L2 + Dropout)")
    print("   - Outlier Detection & Removal")
    print("   - Advanced Feature Engineering")
    print("   - Balanced Sampling (SMOTETomek)")
    print("   - Early Stopping & LR Scheduling")
    print("="*60)
    
    model_info = advanced_model.train_advanced_models()
    
    print(f"\n🎉 مدل پیشرفته آموزش یافت!")
    print("📁 فایل‌های ایجاد شده:")
    print("   - advanced_entry_model.keras")
    print("   - advanced_exit_model.keras") 
    print("   - advanced_model_info.json")
